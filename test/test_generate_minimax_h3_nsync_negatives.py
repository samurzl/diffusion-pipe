import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.generate_minimax_h3_nsync_negatives import (
    DEFAULT_H3_WORKFLOW,
    GenerationError,
    MediaInfo,
    WorkItem,
    WorkflowBinding,
    apply_workflow_model_overrides,
    bind_local_h3_workflow,
    build_parser,
    find_output_resource,
    fit_generation_dimensions,
    load_api_workflow,
    make_work_items,
    make_generation_prompt,
    normalize_output,
    prepare_workflow,
    read_caption,
)


def local_h3_workflow():
    return {
        "12": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["1", 0],
                "vae": ["2", 0],
                "prompt": "template prompt",
                "width": 1344,
                "height": 768,
                "length": 124,
            },
            "_meta": {"title": "MiniMax H3 Image to Video"},
        },
        "20": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 7},
        },
        "30": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["29", 0], "filename_prefix": "video/ComfyUI"},
        },
    }


class GenerateMiniMaxH3NSyncNegativesTest(unittest.TestCase):
    def test_ready_workflow_is_default_and_accepts_manual_model_paths(self):
        parsed = build_parser().parse_args(
            ["positive", "negative", "--remove-text", "TOKperson"]
        )
        self.assertEqual(parsed.workflow, DEFAULT_H3_WORKFLOW)
        self.assertTrue(DEFAULT_H3_WORKFLOW.is_file())

        with tempfile.TemporaryDirectory() as directory:
            comfy_root = Path(directory) / "ComfyUI"
            models = comfy_root / "models"
            model_paths = {
                "diffusion_model": models / "diffusion_models" / "quantized" / "h3.safetensors",
                "text_encoder": models / "text_encoders" / "h3_text.safetensors",
                "video_vae": models / "vae" / "h3_video.safetensors",
                "audio_vae": models / "vae" / "h3_audio.safetensors",
                "turbo_lora": models / "loras" / "turbo" / "h3_turbo.safetensors",
            }
            for model_path in model_paths.values():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.touch()
            workflow = load_api_workflow(DEFAULT_H3_WORKFLOW)

            apply_workflow_model_overrides(
                workflow,
                comfy_root=comfy_root,
                **{name: str(path) for name, path in model_paths.items()},
            )

            self.assertEqual(workflow["1"]["inputs"]["unet_name"], "quantized/h3.safetensors")
            self.assertEqual(workflow["2"]["inputs"]["clip_name"], "h3_text.safetensors")
            self.assertEqual(workflow["3"]["inputs"]["vae_name"], "h3_video.safetensors")
            self.assertEqual(workflow["4"]["inputs"]["vae_name"], "h3_audio.safetensors")
            self.assertEqual(workflow["15"]["inputs"]["lora_name"], "turbo/h3_turbo.safetensors")

    def test_absolute_model_override_requires_comfy_root(self):
        workflow = load_api_workflow(DEFAULT_H3_WORKFLOW)
        with self.assertRaisesRegex(GenerationError, "--comfy-root is required"):
            apply_workflow_model_overrides(
                workflow,
                diffusion_model="/workspace/ComfyUI/models/diffusion_models/h3.safetensors",
            )

    def test_image_positive_produces_one_frame_png_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "positive" / "portrait.jpg"
            positive.parent.mkdir()
            positive.touch()
            positive.with_suffix(".txt").write_text("TOKperson in a studio", encoding="utf-8")
            negative_dir = root / "negative"
            negative_dir.mkdir()
            media = MediaInfo(positive, "image", 1024, 768, 1, 0.0, 1, False)
            args = SimpleNamespace(
                ffprobe="ffprobe",
                caption_index=0,
                remove_text=["TOKperson"],
                prompt_template="{caption}",
                generation_megapixels=0.3,
                negative_dir=negative_dir,
                seed=42,
            )
            binding = WorkflowBinding("5", "prompt", "5", "14", 736, 416)

            with patch(
                "tools.generate_minimax_h3_nsync_negatives.probe_media",
                return_value=media,
            ):
                item = make_work_items(args, [positive], None, binding)[0]

            self.assertEqual(item.generation_length, 5)
            self.assertEqual(item.output, negative_dir / "portrait.png")

            generated = root / "generated.mp4"
            generated.touch()
            normalized = MediaInfo(item.output, "image", 1024, 768, 1, 0.0, 1, False)

            def create_temporary_output(command, _generated, _destination):
                Path(command[-1]).touch()

            with patch(
                "tools.generate_minimax_h3_nsync_negatives.probe_media",
                side_effect=[
                    MediaInfo(generated, "video", 736, 416, 5, 24.0, 5, True),
                    normalized,
                ],
            ), patch(
                "tools.generate_minimax_h3_nsync_negatives._run_ffmpeg",
                side_effect=create_temporary_output,
            ) as run_ffmpeg:
                normalize_output(generated, item.output, media, "ffmpeg", "ffprobe")

            command = run_ffmpeg.call_args.args[0]
            frame_option = command.index("-frames:v")
            self.assertEqual(command[frame_option + 1], "1")
            self.assertTrue(item.output.is_file())

    def test_bundled_api_workflow_is_ready_for_negative_generation(self):
        workflow_path = Path(__file__).resolve().parents[1] / "examples" / "minimax_h3_t2va_api.json"
        workflow = load_api_workflow(workflow_path)
        binding = bind_local_h3_workflow(workflow)

        self.assertEqual(binding.prompt_node, "5")
        self.assertEqual(binding.shape_node, "5")
        self.assertEqual(binding.output_node, "14")
        self.assertNotIn("first_frame", workflow["5"]["inputs"])
        self.assertNotIn("last_frame", workflow["5"]["inputs"])
        self.assertEqual(workflow["5"]["inputs"]["width"], 736)
        self.assertEqual(workflow["5"]["inputs"]["height"], 416)
        self.assertEqual(workflow["5"]["inputs"]["length"], 56)
        self.assertEqual(workflow["8"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(workflow["9"]["inputs"]["steps"], 6)
        self.assertEqual(workflow["15"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(workflow["15"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(workflow["16"]["class_type"], "MiniMaxH3SigmaShift")
        self.assertEqual(workflow["16"]["inputs"]["shift_video"], 6.0)
        self.assertEqual(workflow["16"]["inputs"]["shift_audio"], 3.0)
        self.assertEqual(workflow["7"]["inputs"]["model"], ["16", 0])
        self.assertEqual(workflow["9"]["inputs"]["model"], ["16", 0])
        self.assertEqual(workflow["13"]["inputs"]["fps"], 24.0)
        self.assertEqual(workflow["14"]["class_type"], "SaveVideo")
        self.assertEqual(workflow["13"]["inputs"]["audio"], ["12", 0])
        self.assertEqual(workflow["14"]["inputs"]["video"], ["13", 0])
        self.assertEqual(
            sum(node["class_type"] == "SaveVideo" for node in workflow.values()),
            1,
        )
        for node in workflow.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], workflow)

    def test_bind_and_prepare_local_h3_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = local_h3_workflow()
            binding = bind_local_h3_workflow(workflow)
            media = MediaInfo(root / "shot.mp4", "video", 1920, 1080, 60, 30.0, 48, True)
            item = WorkItem(
                positive=media.path,
                output=root / "shot.mp4",
                caption="a person walking",
                generation_prompt="a person walking",
                media=media,
                generation_width=1344,
                generation_height=768,
                generation_length=48,
                seed=123,
            )

            prepared = prepare_workflow(workflow, binding, item)

            self.assertEqual(binding.prompt_node, "12")
            self.assertEqual(binding.shape_node, "12")
            self.assertEqual(binding.output_node, "30")
            self.assertEqual(prepared["12"]["inputs"]["prompt"], "a person walking")
            self.assertEqual(prepared["12"]["inputs"]["length"], 48)
            self.assertEqual(prepared["20"]["inputs"]["noise_seed"], 123)
            self.assertEqual(prepared["30"]["inputs"]["filename_prefix"], "nsync/shot_123")
            self.assertEqual(workflow["12"]["inputs"]["prompt"], "template prompt")

    def test_hosted_minimax_node_is_rejected(self):
        workflow = local_h3_workflow()
        workflow["40"] = {
            "class_type": "MinimaxHailuo03TextToVideoNode",
            "inputs": {"model": {}},
        }

        with self.assertRaisesRegex(GenerationError, "Hosted MiniMax"):
            bind_local_h3_workflow(workflow)

    def test_connected_positive_frame_is_rejected(self):
        workflow = local_h3_workflow()
        workflow["12"]["inputs"]["first_frame"] = ["5", 0]

        with self.assertRaisesRegex(GenerationError, "first_frame connected"):
            bind_local_h3_workflow(workflow)

    def test_target_text_is_removed_from_generation_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            media = MediaInfo(path, "image", 1024, 768, 1, 0, 1, False)

            prompt = make_generation_prompt(
                "nsyncPerson, walking in NSYNC STYLE, at sunset",
                ["nsyncPerson", "nsync style"],
                "Documentary footage of {caption}",
                media.path,
                media,
            )

            self.assertEqual(prompt, "Documentary footage of walking in, at sunset")
            self.assertNotIn("nsync", prompt.lower())

    def test_caption_sources_and_output_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "shot.mp4"
            positive.write_bytes(b"")
            (root / "shot.txt").write_text("text caption\n", encoding="utf-8")
            self.assertEqual(read_caption(positive, None, 0), "text caption")

            captions = {"shot.mp4": ["first caption", "second caption"]}
            self.assertEqual(read_caption(positive, captions, 1), "second caption")

            history = {
                "outputs": {
                    "30": {
                        "videos": [
                            {"filename": "shot.mp4", "subfolder": "nsync", "type": "output"}
                        ]
                    }
                }
            }
            self.assertEqual(find_output_resource(history, "30")["filename"], "shot.mp4")

    def test_generation_dimensions_preserve_aspect_and_pixel_area(self):
        width, height = fit_generation_dimensions(1920, 1080, 1344 * 768)

        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertAlmostEqual(width / height, 16 / 9, delta=(16 / 9) * 0.03)
        self.assertAlmostEqual(width * height, 1344 * 768, delta=(1344 * 768) * 0.05)


if __name__ == "__main__":
    unittest.main()
