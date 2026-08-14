#!/usr/bin/env python3
"""Generate MiniMax H3 N-Sync negatives through a local ComfyUI workflow.

The input workflow must be exported from ComfyUI in API format and use the
local MiniMax H3 nodes. Hosted/partner MiniMax API nodes are rejected. For each
positive media file, this script:

1. reads the matching caption from ``<stem>.txt`` or ``captions.json``;
2. removes user-supplied target/style text from the generation prompt;
3. queues a copy of the local H3 workflow through ComfyUI's HTTP API;
4. downloads the workflow output; and
5. uses ffmpeg to match the positive's dimensions, 24 fps duration/frame count,
   media type, and audio presence.

The output keeps the positive filename stem, which is how diffusion-pipe pairs
positive and negative examples for N-Sync training. Training should reuse the
positive caption directory via ``caption_path``; generation prompts are not
written as training captions.

Full guide: ``docs/minimax_h3_nsync_negative_generation.md``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


H3_FPS = 24.0
H3_DIMENSION_MULTIPLE = 32
H3_MIN_LENGTH = 5
H3_MAX_LENGTH = 3600

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

LOCAL_H3_NODE_TYPES = {
    "EmptyMiniMaxH3LatentAV",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3SigmaShift",
}
HOSTED_MINIMAX_NODE_PREFIXES = (
    "MinimaxHailuo",
    "MinimaxTextToVideoNode",
    "MinimaxImageToVideoNode",
    "MinimaxSubjectToVideoNode",
)
OUTPUT_NODE_TYPES = {
    "SaveVideo",
    "SaveWEBM",
    "VHS_VideoCombine",
    "SaveImage",
    "SaveAnimatedWEBP",
}


class GenerationError(RuntimeError):
    """A recoverable, user-facing generation error."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    kind: str
    width: int
    height: int
    source_frames: int
    source_fps: float
    target_frames: int
    has_audio: bool

    @property
    def target_duration(self) -> float:
        if self.kind == "image":
            return 0.0
        return self.target_frames / H3_FPS


@dataclass(frozen=True)
class WorkflowBinding:
    prompt_node: str
    prompt_input: str
    shape_node: str
    output_node: str
    default_width: int
    default_height: int


@dataclass(frozen=True)
class WorkItem:
    positive: Path
    output: Path
    caption: str
    generation_prompt: str
    media: MediaInfo
    generation_width: int
    generation_height: int
    generation_length: int
    seed: int


def _parse_fraction(value: Any) -> float:
    if value in (None, "", "0/0", "N/A"):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    numerator, separator, denominator = str(value).partition("/")
    if separator:
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def _parse_positive_int(value: Any) -> int:
    if value in (None, "", "N/A"):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def probe_media(path: Path, ffprobe: str = "ffprobe") -> MediaInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,"
            "nb_frames,nb_read_frames:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise GenerationError(f"ffprobe was not found: {ffprobe}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise GenerationError(f"ffprobe could not read {path}: {detail}") from error

    try:
        metadata = json.loads(result.stdout)
        streams = metadata.get("streams", [])
        video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
    except (json.JSONDecodeError, StopIteration) as error:
        raise GenerationError(f"No readable video/image stream found in {path}") from error

    width = _parse_positive_int(video_stream.get("width"))
    height = _parse_positive_int(video_stream.get("height"))
    if not width or not height:
        raise GenerationError(f"Could not determine dimensions for {path}")

    suffix = path.suffix.lower()
    kind = "image" if suffix in IMAGE_EXTENSIONS else "video"
    if kind == "image":
        return MediaInfo(path, kind, width, height, 1, 0.0, 1, False)

    fps = _parse_fraction(video_stream.get("avg_frame_rate"))
    if fps <= 0:
        fps = _parse_fraction(video_stream.get("r_frame_rate"))
    source_frames = _parse_positive_int(video_stream.get("nb_read_frames"))
    if not source_frames:
        source_frames = _parse_positive_int(video_stream.get("nb_frames"))
    duration = _parse_fraction(metadata.get("format", {}).get("duration"))
    if not source_frames and duration > 0 and fps > 0:
        source_frames = max(1, round(duration * fps))
    if fps <= 0 and source_frames > 0 and duration > 0:
        fps = source_frames / duration
    if fps <= 0 or source_frames <= 1:
        raise GenerationError(f"Could not determine frame count/frame rate for video {path}")

    # Match PreprocessMediaFile.convert_framerate(), which floors this value.
    target_frames = max(2, int(source_frames * H3_FPS / fps))
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    return MediaInfo(path, kind, width, height, source_frames, fps, target_frames, has_audio)


def enumerate_media(directory: Path) -> list[Path]:
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    stems: dict[str, Path] = {}
    for path in files:
        if path.stem in stems:
            raise GenerationError(
                f"Positive files must have unique stems, but both {stems[path.stem].name} "
                f"and {path.name} use {path.stem!r}"
            )
        stems[path.stem] = path
    return files


def load_caption_data(directory: Path) -> dict[str, Any] | None:
    captions_json = directory / "captions.json"
    if not captions_json.exists():
        return None
    try:
        with captions_json.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationError(f"Could not read {captions_json}: {error}") from error
    if not isinstance(data, dict):
        raise GenerationError(f"{captions_json} must contain a JSON object")
    return data


def read_caption(path: Path, caption_data: dict[str, Any] | None, caption_index: int) -> str:
    if caption_data is not None:
        captions = caption_data.get(path.name)
        if not isinstance(captions, list) or not captions:
            raise GenerationError(f"captions.json has no non-empty caption list for {path.name}")
        if caption_index >= len(captions):
            raise GenerationError(
                f"Caption index {caption_index} is out of range for {path.name} "
                f"({len(captions)} captions)"
            )
        caption = captions[caption_index]
        if not isinstance(caption, str):
            raise GenerationError(f"Caption {caption_index} for {path.name} is not a string")
    else:
        caption_path = path.with_suffix(".txt")
        if not caption_path.exists():
            raise GenerationError(f"Missing caption file: {caption_path}")
        caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption.strip():
        raise GenerationError(f"Caption for {path.name} is empty")
    return caption.strip()


def remove_target_text(caption: str, fragments: list[str]) -> str:
    prompt = caption
    for fragment in fragments:
        if fragment:
            prompt = re.sub(re.escape(fragment), "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"\s+([,.;:!?])", r"\1", prompt)
    prompt = re.sub(r"([,;])(?:\s*[,;])+", r"\1", prompt)
    prompt = prompt.strip(" \t\r\n,;:-")
    return prompt


def make_generation_prompt(
    caption: str,
    fragments: list[str],
    template: str,
    path: Path,
    media: MediaInfo,
) -> str:
    sanitized = remove_target_text(caption, fragments)
    if not sanitized:
        raise GenerationError(
            f"Removing target text made the generation prompt empty for {path.name}; "
            "provide a more descriptive caption or prompt template"
        )
    values = {
        "caption": sanitized,
        "stem": path.stem,
        "width": media.width,
        "height": media.height,
        "frames": media.target_frames,
        "seconds": media.target_duration,
    }
    try:
        prompt = template.format_map(values).strip()
    except (KeyError, ValueError) as error:
        raise GenerationError(f"Invalid --prompt-template: {error}") from error
    if not prompt:
        raise GenerationError(f"Generation prompt for {path.name} is empty")
    return prompt


def stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=4).digest()
    return (base_seed + int.from_bytes(digest, "big")) & 0xFFFFFFFF


def fit_generation_dimensions(width: int, height: int, pixels: int) -> tuple[int, int]:
    ratio = width / height
    generated_width = math.sqrt(pixels * ratio)
    generated_height = math.sqrt(pixels / ratio)
    generated_width = max(
        H3_DIMENSION_MULTIPLE,
        round(generated_width / H3_DIMENSION_MULTIPLE) * H3_DIMENSION_MULTIPLE,
    )
    generated_height = max(
        H3_DIMENSION_MULTIPLE,
        round(generated_height / H3_DIMENSION_MULTIPLE) * H3_DIMENSION_MULTIPLE,
    )
    return generated_width, generated_height


def load_api_workflow(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as file:
            workflow = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationError(f"Could not read workflow {path}: {error}") from error
    if not isinstance(workflow, dict) or not workflow:
        raise GenerationError("The workflow must be a non-empty JSON object")
    if "nodes" in workflow and isinstance(workflow["nodes"], list):
        raise GenerationError(
            "This is a UI-format workflow. In ComfyUI, export/save it in API format instead."
        )
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
            raise GenerationError(f"Workflow node {node_id!r} is not in ComfyUI API format")
        if not isinstance(node.get("inputs"), dict):
            raise GenerationError(f"Workflow node {node_id!r} has no inputs object")
    return workflow


def _node_title(node: dict[str, Any]) -> str:
    meta = node.get("_meta")
    return str(meta.get("title", "")) if isinstance(meta, dict) else ""


def _choose_node(
    workflow: dict[str, dict[str, Any]],
    explicit_id: str | None,
    candidates: list[str],
    label: str,
) -> str:
    if explicit_id is not None:
        if explicit_id not in workflow:
            raise GenerationError(f"Unknown {label} node ID: {explicit_id}")
        return explicit_id
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise GenerationError(f"Could not find a {label} node in the workflow")
    details = ", ".join(
        f"{node_id} ({_node_title(workflow[node_id]) or workflow[node_id]['class_type']})"
        for node_id in candidates
    )
    raise GenerationError(f"Multiple {label} nodes found: {details}; select one explicitly")


def bind_local_h3_workflow(
    workflow: dict[str, dict[str, Any]],
    conditioning_node: str | None = None,
    prompt_node: str | None = None,
    shape_node: str | None = None,
    output_node: str | None = None,
) -> WorkflowBinding:
    class_types = {node["class_type"] for node in workflow.values()}
    hosted = sorted(
        class_type
        for class_type in class_types
        if class_type.startswith(HOSTED_MINIMAX_NODE_PREFIXES)
    )
    if hosted:
        raise GenerationError(
            "Hosted MiniMax/Comfy API nodes are not allowed; use the local MiniMax H3 model nodes. "
            f"Found: {', '.join(hosted)}"
        )
    if not class_types.intersection(LOCAL_H3_NODE_TYPES):
        raise GenerationError("Workflow does not contain any local MiniMax H3 nodes")

    conditioning_candidates = [
        node_id
        for node_id, node in workflow.items()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
        and {"prompt", "width", "height", "length"}.issubset(node["inputs"])
    ]
    if conditioning_node is not None or conditioning_candidates:
        selected = _choose_node(
            workflow,
            conditioning_node,
            conditioning_candidates,
            "MiniMax H3 conditioning",
        )
        node = workflow[selected]
        if node["class_type"] != "MiniMaxH3ImageToVideo":
            raise GenerationError(
                f"Conditioning node {selected} must be MiniMaxH3ImageToVideo, not {node['class_type']}"
            )
        for reference_input in ("first_frame", "last_frame"):
            if node["inputs"].get(reference_input) is not None:
                raise GenerationError(
                    f"Conditioning node {selected} has {reference_input} connected. N-Sync negatives "
                    "must be text-to-video so the positive target/style is not copied into the negative."
                )
        prompt_id = shape_id = selected
        prompt_input = "prompt"
    else:
        shape_candidates = [
            node_id
            for node_id, node in workflow.items()
            if node["class_type"] == "EmptyMiniMaxH3LatentAV"
            and {"width", "height", "length"}.issubset(node["inputs"])
        ]
        shape_id = _choose_node(workflow, shape_node, shape_candidates, "MiniMax H3 latent/shape")
        prompt_candidates = [
            node_id
            for node_id, node in workflow.items()
            if node["class_type"] == "CLIPTextEncode" and "text" in node["inputs"]
        ]
        if prompt_node is None and len(prompt_candidates) > 1:
            preferred = [
                node_id
                for node_id in prompt_candidates
                if re.search(r"positive|prompt", _node_title(workflow[node_id]), flags=re.IGNORECASE)
                and not re.search(r"negative", _node_title(workflow[node_id]), flags=re.IGNORECASE)
            ]
            if len(preferred) == 1:
                prompt_candidates = preferred
        prompt_id = _choose_node(workflow, prompt_node, prompt_candidates, "positive prompt")
        prompt_input = "text"

    if output_node is not None:
        if output_node not in workflow:
            raise GenerationError(f"Unknown output node ID: {output_node}")
        output_id = output_node
    else:
        save_video_candidates = [
            node_id for node_id, node in workflow.items() if node["class_type"] == "SaveVideo"
        ]
        output_candidates = save_video_candidates or [
            node_id for node_id, node in workflow.items() if node["class_type"] in OUTPUT_NODE_TYPES
        ]
        output_id = _choose_node(workflow, None, output_candidates, "saved-media output")

    shape_inputs = workflow[shape_id]["inputs"]
    try:
        default_width = int(shape_inputs["width"])
        default_height = int(shape_inputs["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise GenerationError(
            f"Shape node {shape_id} must have literal integer width and height inputs"
        ) from error
    if default_width <= 0 or default_height <= 0:
        raise GenerationError(f"Shape node {shape_id} has invalid dimensions")
    return WorkflowBinding(
        prompt_node=prompt_id,
        prompt_input=prompt_input,
        shape_node=shape_id,
        output_node=output_id,
        default_width=default_width,
        default_height=default_height,
    )


def prepare_workflow(
    template: dict[str, dict[str, Any]],
    binding: WorkflowBinding,
    item: WorkItem,
) -> dict[str, dict[str, Any]]:
    workflow = copy.deepcopy(template)
    workflow[binding.prompt_node]["inputs"][binding.prompt_input] = item.generation_prompt
    shape_inputs = workflow[binding.shape_node]["inputs"]
    shape_inputs["width"] = item.generation_width
    shape_inputs["height"] = item.generation_height
    shape_inputs["length"] = item.generation_length

    for node in workflow.values():
        for input_name in ("seed", "noise_seed"):
            value = node["inputs"].get(input_name)
            if isinstance(value, int) and not isinstance(value, bool):
                node["inputs"][input_name] = item.seed

    output_inputs = workflow[binding.output_node]["inputs"]
    if isinstance(output_inputs.get("filename_prefix"), str):
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.positive.stem)
        output_inputs["filename_prefix"] = f"nsync/{safe_stem}_{item.seed}"
    return workflow


class ComfyClient:
    def __init__(self, base_url: str, timeout: float, poll_interval: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = str(uuid.uuid4())

    def _json_request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        request_timeout: float = 60.0,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise GenerationError(f"ComfyUI returned HTTP {error.code} for {path}: {body}") from error
        except urllib.error.URLError as error:
            raise GenerationError(f"Could not reach local ComfyUI at {self.base_url}: {error}") from error
        except json.JSONDecodeError as error:
            raise GenerationError(f"ComfyUI returned invalid JSON for {path}") from error

    def queue(self, workflow: dict[str, dict[str, Any]]) -> str:
        response = self._json_request(
            "/prompt",
            {"prompt": workflow, "client_id": self.client_id},
        )
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise GenerationError(f"ComfyUI did not return a prompt ID: {response}")
        return str(prompt_id)

    def wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        encoded_id = urllib.parse.quote(prompt_id, safe="")
        while time.monotonic() < deadline:
            history = self._json_request(f"/history/{encoded_id}")
            entry = history.get(prompt_id)
            if entry is not None:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise GenerationError(f"ComfyUI workflow failed: {json.dumps(messages)}")
                if status.get("completed", True):
                    return entry
            time.sleep(self.poll_interval)
        raise GenerationError(f"Timed out after {self.timeout:.0f}s waiting for ComfyUI prompt {prompt_id}")

    def download(self, resource: dict[str, Any], destination: Path) -> None:
        query = urllib.parse.urlencode(
            {
                "filename": resource["filename"],
                "subfolder": resource.get("subfolder", ""),
                "type": resource.get("type", "output"),
            }
        )
        request = urllib.request.Request(f"{self.base_url}/view?{query}")
        try:
            with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as file:
                shutil.copyfileobj(response, file)
        except (OSError, urllib.error.URLError) as error:
            raise GenerationError(f"Could not download ComfyUI output {resource}: {error}") from error


def _collect_resources(value: Any, resources: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str):
            resources.append(value)
            return
        for child in value.values():
            _collect_resources(child, resources)
    elif isinstance(value, list):
        for child in value:
            _collect_resources(child, resources)


def find_output_resource(history: dict[str, Any], output_node: str) -> dict[str, Any]:
    outputs = history.get("outputs", {})
    node_output = outputs.get(output_node)
    if node_output is None:
        available = ", ".join(sorted(outputs)) or "none"
        raise GenerationError(
            f"ComfyUI history has no output for node {output_node}; available output nodes: {available}"
        )
    resources: list[dict[str, Any]] = []
    _collect_resources(node_output, resources)
    unique_resources = {
        (resource.get("filename"), resource.get("subfolder", ""), resource.get("type", "output")): resource
        for resource in resources
    }
    resources = list(unique_resources.values())
    if len(resources) != 1:
        raise GenerationError(
            f"Expected exactly one saved media file from output node {output_node}, found {len(resources)}"
        )
    return resources[0]


def _run_ffmpeg(command: list[str], source: Path, destination: Path) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise GenerationError(f"ffmpeg was not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise GenerationError(
            f"ffmpeg could not normalize {source.name} to {destination.name}: {detail}"
        ) from error


def normalize_output(
    generated: Path,
    destination: Path,
    positive: MediaInfo,
    ffmpeg: str,
    ffprobe: str,
) -> None:
    generated_info = probe_media(generated, ffprobe)
    scale_crop = (
        f"scale={positive.width}:{positive.height}:force_original_aspect_ratio=increase,"
        f"crop={positive.width}:{positive.height}"
    )
    temporary = destination.with_name(f".{destination.stem}.part{destination.suffix}")
    if temporary.exists():
        temporary.unlink()

    if positive.kind == "image":
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(generated),
            "-map",
            "0:v:0",
            "-vf",
            scale_crop,
            "-frames:v",
            "1",
            str(temporary),
        ]
    else:
        if positive.has_audio and not generated_info.has_audio:
            raise GenerationError(
                f"{positive.path.name} has audio, but the ComfyUI output does not. "
                "Connect the H3 audio decode to the saved video in the local workflow."
            )
        duration = positive.target_duration
        video_filter = (
            f"fps={H3_FPS:g},{scale_crop},"
            f"tpad=stop_mode=clone:stop_duration={duration:.6f}"
        )
        pixel_format = "yuv420p" if positive.width % 2 == 0 and positive.height % 2 == 0 else "yuv444p"
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(generated),
            "-map",
            "0:v:0",
        ]
        if positive.has_audio:
            command.extend(["-map", "0:a:0", "-af", f"apad=whole_dur={duration:.6f}"])
        command.extend(
            [
                "-vf",
                video_filter,
                "-frames:v",
                str(positive.target_frames),
                "-t",
                f"{duration:.9f}",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-pix_fmt",
                pixel_format,
            ]
        )
        if positive.has_audio:
            command.extend(["-c:a", "aac", "-ar", "32000", "-ac", "2"])
        command.append(str(temporary))

    try:
        _run_ffmpeg(command, generated, destination)
        normalized = probe_media(temporary, ffprobe)
        if normalized.width != positive.width or normalized.height != positive.height:
            raise GenerationError(
                f"Normalized output has {normalized.width}x{normalized.height}, expected "
                f"{positive.width}x{positive.height}"
            )
        if positive.kind == "image" and normalized.kind != "image":
            raise GenerationError(f"Expected an image output for {positive.path.name}")
        if positive.kind == "video" and normalized.target_frames != positive.target_frames:
            raise GenerationError(
                f"Normalized output has {normalized.target_frames} frames at 24 fps, expected "
                f"{positive.target_frames}"
            )
        if normalized.has_audio != positive.has_audio:
            raise GenerationError(
                f"Normalized output audio presence ({normalized.has_audio}) does not match positive "
                f"({positive.has_audio})"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def output_is_valid(path: Path, positive: MediaInfo, ffprobe: str) -> bool:
    try:
        output = probe_media(path, ffprobe)
    except GenerationError:
        return False
    if output.width != positive.width or output.height != positive.height:
        return False
    if positive.kind != output.kind or positive.has_audio != output.has_audio:
        return False
    if positive.kind == "video" and output.target_frames != positive.target_frames:
        return False
    return True


def write_manifest(path: Path, records: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paired MiniMax H3 N-Sync negatives with a local ComfyUI workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Guide: docs/minimax_h3_nsync_negative_generation.md",
    )
    parser.add_argument("positive_dir", type=Path, help="Directory containing positive media and captions")
    parser.add_argument("negative_dir", type=Path, help="Directory in which paired negatives are written")
    parser.add_argument("--workflow", required=True, type=Path, help="Local H3 workflow exported in API format")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="Local ComfyUI server URL")
    parser.add_argument(
        "--remove-text",
        action="append",
        default=[],
        metavar="TEXT",
        help="Literal target/trigger/style text to remove from generation captions; repeat as needed",
    )
    parser.add_argument(
        "--allow-unchanged-prompt",
        action="store_true",
        help="Allow generation without --remove-text (only safe when captions already omit the target)",
    )
    parser.add_argument(
        "--prompt-template",
        default="{caption}",
        help=(
            "Generation prompt template; fields: {caption}, {stem}, {width}, {height}, "
            "{frames}, {seconds}"
        ),
    )
    parser.add_argument("--caption-index", type=int, default=0, help="Caption to use from each captions.json list")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; each filename receives a stable offset")
    parser.add_argument(
        "--generation-megapixels",
        type=float,
        help="Generation canvas area; by default the workflow's H3 canvas area is preserved",
    )
    parser.add_argument("--conditioning-node", help="Explicit MiniMaxH3ImageToVideo node ID")
    parser.add_argument("--prompt-node", help="Explicit CLIPTextEncode node ID for legacy H3 workflows")
    parser.add_argument("--shape-node", help="Explicit EmptyMiniMaxH3LatentAV node ID for legacy H3 workflows")
    parser.add_argument("--output-node", help="Explicit saved-media output node ID")
    parser.add_argument("--timeout", type=float, default=3600, help="Maximum seconds to wait for each generation")
    parser.add_argument("--poll-interval", type=float, default=2, help="History polling interval in seconds")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    parser.add_argument("--limit", type=int, help="Only process the first N positives")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate and replace existing valid outputs")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs and print planned jobs without queuing ComfyUI")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.positive_dir.is_dir():
        raise GenerationError(f"Positive directory does not exist: {args.positive_dir}")
    if args.positive_dir.resolve() == args.negative_dir.resolve():
        raise GenerationError("Positive and negative directories must be different")
    if not args.workflow.is_file():
        raise GenerationError(f"Workflow does not exist: {args.workflow}")
    args.remove_text = [fragment.strip() for fragment in args.remove_text if fragment.strip()]
    if not args.remove_text and not args.allow_unchanged_prompt:
        raise GenerationError(
            "At least one --remove-text is required so generated negatives omit the target concept/style. "
            "Use --allow-unchanged-prompt only when the positive captions already omit it."
        )
    if args.caption_index < 0:
        raise GenerationError("--caption-index must be non-negative")
    if not 0 <= args.seed <= 0xFFFFFFFF:
        raise GenerationError("--seed must be between 0 and 4294967295")
    if args.generation_megapixels is not None and args.generation_megapixels <= 0:
        raise GenerationError("--generation-megapixels must be positive")
    if args.limit is not None and args.limit <= 0:
        raise GenerationError("--limit must be positive")
    if args.timeout <= 0 or args.poll_interval <= 0:
        raise GenerationError("--timeout and --poll-interval must be positive")


def make_work_items(
    args: argparse.Namespace,
    media_paths: list[Path],
    caption_data: dict[str, Any] | None,
    binding: WorkflowBinding,
) -> list[WorkItem]:
    default_pixels = binding.default_width * binding.default_height
    generation_pixels = (
        round(args.generation_megapixels * 1_000_000)
        if args.generation_megapixels is not None
        else default_pixels
    )
    items = []
    for positive in media_paths:
        media = probe_media(positive, args.ffprobe)
        caption = read_caption(positive, caption_data, args.caption_index)
        prompt = make_generation_prompt(caption, args.remove_text, args.prompt_template, positive, media)
        generation_width, generation_height = fit_generation_dimensions(
            media.width,
            media.height,
            generation_pixels,
        )
        # Local H3 inference needs at least five frames. Image positives still
        # produce one-frame image negatives: normalize_output extracts exactly
        # one decoded frame and writes it to the same-stem PNG destination.
        generation_length = H3_MIN_LENGTH if media.kind == "image" else max(H3_MIN_LENGTH, media.target_frames)
        if generation_length > H3_MAX_LENGTH:
            raise GenerationError(
                f"{positive.name} needs H3 length {generation_length}, exceeding the local node maximum "
                f"of {H3_MAX_LENGTH}; split the positive into shorter clips"
            )
        output = args.negative_dir / f"{positive.stem}{'.png' if media.kind == 'image' else '.mp4'}"
        items.append(
            WorkItem(
                positive=positive,
                output=output,
                caption=caption,
                generation_prompt=prompt,
                media=media,
                generation_width=generation_width,
                generation_height=generation_height,
                generation_length=generation_length,
                seed=stable_seed(args.seed, positive.name),
            )
        )
    return items


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    workflow = load_api_workflow(args.workflow)
    binding = bind_local_h3_workflow(
        workflow,
        conditioning_node=args.conditioning_node,
        prompt_node=args.prompt_node,
        shape_node=args.shape_node,
        output_node=args.output_node,
    )
    media_paths = enumerate_media(args.positive_dir)
    if args.limit is not None:
        media_paths = media_paths[: args.limit]
    if not media_paths:
        raise GenerationError(f"No supported media found in {args.positive_dir}")
    caption_data = load_caption_data(args.positive_dir)
    items = make_work_items(args, media_paths, caption_data, binding)

    print(
        f"Found {len(items)} positive(s); local H3 prompt node={binding.prompt_node}, "
        f"shape node={binding.shape_node}, output node={binding.output_node}"
    )
    for index, item in enumerate(items, start=1):
        audio = "+audio" if item.media.has_audio else "no audio"
        print(
            f"[{index}/{len(items)}] {item.positive.name} -> {item.output.name}; "
            f"generate {item.generation_width}x{item.generation_height}x{item.generation_length}, "
            f"normalize {item.media.width}x{item.media.height}x{item.media.target_frames} ({audio}); "
            f"seed={item.seed}\n  prompt: {item.generation_prompt}"
        )
    if args.dry_run:
        return 0

    args.negative_dir.mkdir(parents=True, exist_ok=True)
    client = ComfyClient(args.comfy_url, args.timeout, args.poll_interval)
    manifest_path = args.negative_dir / ".nsync_generation_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}

    completed = skipped = 0
    for index, item in enumerate(items, start=1):
        if item.output.exists() and not args.overwrite:
            if not output_is_valid(item.output, item.media, args.ffprobe):
                raise GenerationError(
                    f"Existing output is not a valid pair for {item.positive.name}: {item.output}. "
                    "Inspect it, then pass --overwrite to replace it."
                )
            print(f"[{index}/{len(items)}] Already valid, skipping {item.output.name}")
            skipped += 1
            continue

        print(f"[{index}/{len(items)}] Queueing {item.positive.name} in local ComfyUI", flush=True)
        prompt_workflow = prepare_workflow(workflow, binding, item)
        prompt_id = client.queue(prompt_workflow)
        history = client.wait(prompt_id)
        resource = find_output_resource(history, binding.output_node)
        suffix = Path(resource["filename"]).suffix or ".bin"
        with tempfile.TemporaryDirectory(prefix="nsync-comfy-") as temp_directory:
            downloaded = Path(temp_directory) / f"generated{suffix}"
            client.download(resource, downloaded)
            normalize_output(downloaded, item.output, item.media, args.ffmpeg, args.ffprobe)

        manifest[item.positive.name] = {
            "output": item.output.name,
            "prompt": item.generation_prompt,
            "seed": item.seed,
            "comfy_prompt_id": prompt_id,
            "positive_dimensions": [item.media.width, item.media.height],
            "positive_target_frames": item.media.target_frames,
            "positive_has_audio": item.media.has_audio,
            "generation_dimensions": [item.generation_width, item.generation_height],
            "generation_length": item.generation_length,
        }
        write_manifest(manifest_path, manifest)
        completed += 1
        print(f"[{index}/{len(items)}] Wrote {item.output}", flush=True)

    print(f"Done: generated {completed}, skipped {skipped}, total {len(items)}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except GenerationError as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())
