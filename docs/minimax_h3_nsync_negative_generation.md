# Generating MiniMax H3 NSYNC negatives with local ComfyUI

[`tools/generate_minimax_h3_nsync_negatives.py`](../tools/generate_minimax_h3_nsync_negatives.py) builds the paired generated-negative directory required by MiniMax H3 NSYNC training. ComfyUI runs the local MiniMax H3 model as the inference backend; the script only uses ComfyUI's local HTTP queue, history, and file endpoints. A ready API-format graph is included at [`examples/minimax_h3_t2va_api.json`](../examples/minimax_h3_t2va_api.json); it uses the LightX2V four-step Turbo LoRA, and the RunPod bootstrap patches its loader filenames to the model variants it finds in the existing ComfyUI.

If you are starting with an empty GPU instance, use the [fresh RunPod MiniMax H3 NSYNC + Self-Flow guide](minimax_h3_nsync_self_flow_runpod.md) for the Pod template, installation, model downloads, ComfyUI setup, negative generation, caching, and training commands in one sequence.

The script never calls the hosted MiniMax service. It rejects workflows containing ComfyUI MiniMax partner/API nodes such as `MinimaxHailuo03TextToVideoNode`.

## What the script guarantees

For every supported file directly inside the positive directory, the script:

1. reads the matching same-stem `.txt` caption or the file's list in `captions.json`;
2. removes each `--remove-text` phrase from the generation prompt;
3. derives a generation canvas with the positive's aspect ratio;
4. runs one text-conditioned local MiniMax H3 ComfyUI job;
5. downloads the single saved result;
6. writes a negative with the same filename stem; and
7. normalizes the result to the positive's dimensions, 24 fps duration/frame count, media type, and audio presence.

Image positives produce same-stem `.png` negatives. Video positives produce same-stem `.mp4` negatives. Extensions may differ because diffusion-pipe pairs NSYNC media by stem.

The generated media and training captions serve different purposes:

- The **generation prompt** should omit the target character, trigger, concept, or style. This helps the base H3 model create an off-target negative that still depicts the same general content.
- The **training caption** must remain exactly equal for the positive and its negative. The negative dataset therefore points `caption_path` at the positive directory.

The script does not copy rewritten prompts into the negative directory as captions.

## Requirements

- A locally running ComfyUI instance with local MiniMax H3 inference working.
- The bundled H3 API workflow, or another H3 text-to-video workflow exported in ComfyUI **API format**.
- LightX2V's `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` in ComfyUI's `models/loras` directory.
- `ffmpeg` and `ffprobe` available on `PATH` or supplied through `--ffmpeg` and `--ffprobe`.
- Positive media with same-stem `.txt` captions, or a standard `captions.json`.
- Different positive and negative directories.

Verify the media tools before starting:

```bash
ffmpeg -version
ffprobe -version
```

## Use or customize the local ComfyUI workflow

For the standard setup, use the bundled workflow directly:

```bash
export H3_WORKFLOW_API=/workspace/workflows/minimax_h3_t2va_api.json
```

The RunPod setup script creates that runtime copy with the correct ComfyUI loader filenames. If using the standard model filenames without the bootstrap, use `examples/minimax_h3_t2va_api.json` directly. The bundled defaults are 736×416 (approximately 0.3 MP), 56 frames on H3's `17n+5` grid (approximately 2.33 seconds at 24 fps), Euler, six sampling steps, LoRA strength 1.0, and LightX2V's required video/audio sigma shifts of 6/3. The generator replaces the canvas and frame request per positive so paired negatives still match their sources. Continue to [Prepare the positive directory](#prepare-the-positive-directory) unless you need to customize sampling.

To build a custom workflow, create and successfully run the generation graph once in ComfyUI before exporting it:

1. Load the local MiniMax H3 diffusion model, video VAE, audio VAE, and MiniMax text encoder.
2. Use `MiniMaxH3ImageToVideo` for the positive conditioning and empty AV latent.
3. Leave `first_frame` and `last_frame` disconnected. Do not use the positive media as a reference: doing so can copy the target concept/style into the negative.
4. Decode the H3 video and audio outputs and combine them into the video saved by one `SaveVideo` node. Generated audio is required when a positive video has audio.
5. Keep model filenames, sampling steps, sampler, scheduler, CFG, and local quantization settings in the workflow.
6. Export the workflow in API format. A normal UI-format workflow contains a `nodes` list and cannot be queued through ComfyUI's `/prompt` endpoint.

The script automatically finds one `MiniMaxH3ImageToVideo` node and prefers one `SaveVideo` output. Older local H3 workflows using `EmptyMiniMaxH3LatentAV` plus `CLIPTextEncode` are also supported. If a workflow has multiple matching nodes, select them with `--conditioning-node`, `--prompt-node`, `--shape-node`, or `--output-node`.

## Prepare the positive directory

Same-stem text captions look like this:

```text
target_positive_media/
├── shot_001.mp4
├── shot_001.txt
├── portrait_002.png
└── portrait_002.txt
```

For example, `shot_001.txt` might contain:

```text
TOKperson walking through a train station at sunset
```

Pass `--remove-text TOKperson`. H3 then receives `walking through a train station at sunset`, while NSYNC training continues to use the original caption containing `TOKperson` for both paired examples.

`captions.json` is also supported and follows the normal diffusion-pipe format:

```json
{
  "shot_001.mp4": [
    "TOKperson walking through a train station at sunset",
    "TOKperson crossing a station concourse"
  ]
}
```

The first list entry is used for generation by default. Use `--caption-index 1` to select the second entry. Training can still reuse the entire caption list through `caption_path`.

Positive files must have unique stems. For example, do not place both `shot_001.png` and `shot_001.mp4` in the same positive directory.

## Preview the jobs

Run a dry run before spending time on H3 inference:

```bash
python tools/generate_minimax_h3_nsync_negatives.py \
  /path/to/target_positive_media \
  /path/to/generated_negative_media \
  --workflow /path/to/minimax_h3_t2va_api.json \
  --remove-text 'TOKperson' \
  --dry-run
```

The dry run probes every positive and prints:

- the final generation prompt;
- the ComfyUI generation width, height, and requested length;
- the normalized output dimensions and 24 fps frame count;
- whether audio must be present; and
- the deterministic per-file seed.

Repeat `--remove-text` for every trigger, target name, or style phrase that may appear:

```bash
python tools/generate_minimax_h3_nsync_negatives.py \
  /path/to/target_positive_media \
  /path/to/generated_negative_media \
  --workflow /path/to/minimax_h3_t2va_api.json \
  --remove-text 'TOKperson' \
  --remove-text 'TOKstyle' \
  --prompt-template 'Natural documentary footage of {caption}' \
  --dry-run
```

`--remove-text` is required by default. Use `--allow-unchanged-prompt` only when the captions already omit the target; otherwise the generated negative may reproduce the concept that NSYNC is meant to suppress.

## Generate and resume

After reviewing the dry-run output, run the same command without `--dry-run`:

```bash
python tools/generate_minimax_h3_nsync_negatives.py \
  /path/to/target_positive_media \
  /path/to/generated_negative_media \
  --workflow /path/to/minimax_h3_t2va_api.json \
  --remove-text 'TOKperson'
```

Jobs run sequentially because local MiniMax H3 inference uses batch size 1. An interrupted run is resumable: valid existing outputs are checked and skipped. If an existing same-stem output is invalid, the script stops instead of silently replacing it. Inspect the file and pass `--overwrite` when replacing it is intentional.

The negative directory also receives `.nsync_generation_manifest.json`, which records each generation prompt, seed, ComfyUI prompt ID, source geometry, generation geometry, length, and audio requirement. Its `.json` extension means the training dataset loader ignores it as media.

Useful options:

| Option | Purpose |
|---|---|
| `--comfy-url URL` | Select the ComfyUI server; defaults to `http://127.0.0.1:8188`. |
| `--remove-text TEXT` | Remove a literal target phrase, case-insensitively; repeatable. |
| `--prompt-template TEMPLATE` | Wrap the cleaned caption. Supports `{caption}`, `{stem}`, `{width}`, `{height}`, `{frames}`, and `{seconds}`. |
| `--caption-index N` | Select a caption from each `captions.json` list. |
| `--generation-megapixels N` | Override the workflow canvas area while preserving each positive's aspect ratio. |
| `--seed N` | Set the base seed; filename-based offsets keep every job deterministic. |
| `--limit N` | Process only the first `N` positives, useful for validation. |
| `--timeout SECONDS` | Change the per-job ComfyUI timeout. |
| `--overwrite` | Intentionally replace existing outputs. |

Run `python tools/generate_minimax_h3_nsync_negatives.py --help` for every option.

## Configure the NSYNC dataset

Use [`examples/minimax_h3_nsync_self_flow_dataset.toml`](../examples/minimax_h3_nsync_self_flow_dataset.toml) as the template:

```toml
resolutions = [512]
enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7
frame_buckets = [1, 33]

[[directory]]
path = '/path/to/target_positive_media'
num_repeats = 1
nsync_role = 'positive'
nsync_pair = 'target_style'

[[directory]]
path = '/path/to/generated_negative_media'
caption_path = '/path/to/target_positive_media'
num_repeats = 1
nsync_role = 'negative'
nsync_pair = 'target_style'
```

The two directories must use the same `nsync_pair`. `caption_path` is essential: it gives every negative the exact positive training caption rather than the cleaned generation prompt.

Enable NSYNC in the training configuration as shown in [`examples/minimax_h3_nsync_self_flow.toml`](../examples/minimax_h3_nsync_self_flow.toml):

```toml
[training_methods.nsync]
enabled = true
eps = 1e-8
```

Also keep `uncond_fraction = 0`, use data-parallel world size 1, and disable `optimizer.gradient_release`. See the [MiniMax H3 training notes](minimax_h3_notes.md) for the complete NSYNC and Self-Flow constraints.

## Troubleshooting

### Hosted MiniMax/Comfy API nodes are not allowed

The workflow contains a MiniMax partner/API node. Replace it with the local MiniMax H3 loader, conditioning, sampling, and decode graph. The ComfyUI HTTP server is only the transport to your local model.

### This is a UI-format workflow

Export the graph in API format. In ComfyUI versions that hide API export, enable the developer-mode options first.

### Multiple prompt, shape, or output nodes were found

Remove unused outputs or pass the reported node ID through `--conditioning-node`, `--prompt-node`, `--shape-node`, or `--output-node`.

### The positive has audio, but the ComfyUI output does not

Decode the H3 audio latent and connect it to the video saved by `SaveVideo`. The script deliberately rejects a silent negative for an audio-positive pair because the training modalities would differ.

### Existing output is not a valid pair

The same-stem output has different dimensions, frame duration, media type, or audio presence. Inspect it and use `--overwrite` to regenerate it.

### Generation times out

Confirm that ComfyUI is still processing the job and increase `--timeout`. The default is one hour per positive.

### A caption is missing

When `captions.json` exists, every positive filename must have a non-empty list in it. Otherwise, provide a same-stem `.txt` file next to every positive.
