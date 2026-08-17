# Manual RunPod setup for MiniMax H3 LoRA training with NSYNC and Self-Flow

This guide starts with an empty RunPod Pod and ends with a single-GPU MiniMax H3 LoRA training run that has:

- NSYNC gradient surgery;
- Self-Flow with its EMA LoRA teacher;
- paired positive and generated-negative media; and
- automatic negative generation through a **local** MiniMax H3 ComfyUI workflow.

No hosted MiniMax API is used. ComfyUI and diffusion-pipe share the same model files on the Pod. Generate all negatives first, stop ComfyUI, and only then start caching and training so the two processes do not compete for GPU memory.

This walkthrough uses the repository's combined configuration in [`examples/minimax_h3_nsync_self_flow.toml`](../examples/minimax_h3_nsync_self_flow.toml). Read the [MiniMax H3 implementation and training notes](minimax_h3_notes.md) before changing its method-specific settings.

## 1. Create the Pod

The template details below were checked on **August 13, 2026**. RunPod changes its template gallery over time, so use the image tag, rather than a similarly named older template, to disambiguate it.

In RunPod, choose **Deploy > Pods** and configure:

| Setting | Recommended value |
|---|---|
| GPU count | `1` |
| GPU | RTX A6000 or RTX 6000 Ada (48 GB), or another 48+ GB NVIDIA GPU |
| Minimum practical GPU | RTX 4090 24 GB; expect more offloading and use the OOM advice below |
| Template | Official **RunPod PyTorch 2.9.1, CUDA 13.0, Ubuntu 24.04** |
| Exact container image | `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404` |
| Container disk | 30 GB or more |
| Persistent storage mounted at `/workspace` | 200 GB minimum; 300 GB is more comfortable |
| HTTP ports | `8188` for ComfyUI; optionally `6006` for TensorBoard |
| System RAM | 64 GB minimum; 96 GB or more is preferable |

The CUDA 13.0 image is intentional: the [Comfy-Org H3 model card](https://huggingface.co/Comfy-Org/MiniMax-H3) recommends CUDA 13.0 PyTorch for the INT8 ConvRot diffusion checkpoint used below. The exact current CUDA/PyTorch combinations are defined by RunPod's [official container source](https://github.com/runpod/containers/blob/main/official-templates/shared/versions.hcl).

Attach a **Network Volume** if the models, data, cache, and checkpoints must survive deleting the Pod. A normal Volume Disk also mounts at `/workspace` and survives stops/restarts, but it is deleted with the Pod. See [RunPod's storage comparison](https://docs.runpod.io/pods/storage/types).

The four base model files plus the LightX2V Turbo LoRA in this guide occupy about 52 GiB before dataset caches or outputs. The storage recommendation assumes a small LoRA dataset; add enough capacity for the positive media, an equally sized negative set, caches, checkpoints, and saved LoRAs. Do not store anything important outside `/workspace`.

If the gallery does not expose the exact image, create a private custom template using the image name above. If CUDA 13.0 is unavailable in the selected region, use `runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404`; INT8 ConvRot can be less efficient there, so the CUDA 13.0 image remains the preferred setup.

Deploy the Pod, open **Connect > Web Terminal**, and continue below.

## 2. Verify the fresh Pod

Run each command in the Pod terminal:

```bash
nvidia-smi
```

```bash
python --version
```

```bash
python -c "import torch; print('torch:', torch.__version__); print('wheel CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

```bash
nvcc --version
```

The Python check should report PyTorch `2.9.1`, wheel CUDA `13.0`, and the selected GPU. Stop here and choose the correct template if `torch.cuda.is_available()` is false.

Check the persistent mount:

```bash
df -h /workspace
```

## 3. Install system tools and clone diffusion-pipe

```bash
apt-get update
```

```bash
DEBIAN_FRONTEND=noninteractive apt-get install -y git git-lfs ffmpeg curl wget jq tmux rsync
```

```bash
git lfs install
```

Clone this repository, including its pinned ComfyUI submodule:

```bash
git clone --recurse-submodules https://github.com/samurzl/diffusion-pipe.git /workspace/diffusion-pipe
```

```bash
cd /workspace/diffusion-pipe
```

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Define the paths used in the rest of the guide. Re-run this block after opening a new terminal:

```bash
export DP_ROOT=/workspace/diffusion-pipe
export DP_VENV=/workspace/venvs/diffusion-pipe
export COMFYUI_ROOT="$DP_ROOT/submodules/ComfyUI"
export COMFYUI_PYTHON="$DP_VENV/bin/python"
export H3_MODEL_ROOT=/workspace/models/minimax-h3
export H3_DIFFUSION_MODEL="$H3_MODEL_ROOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
export H3_TEXT_ENCODER="$H3_MODEL_ROOT/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
export H3_VIDEO_VAE="$H3_MODEL_ROOT/vae/minimax_h3_video_vae_fp16.safetensors"
export H3_AUDIO_VAE="$H3_MODEL_ROOT/vae/minimax_h3_audio_vae_fp32.safetensors"
export H3_TURBO_LORA="$H3_MODEL_ROOT/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
export H3_DATA_ROOT=/workspace/data/minimax-h3
export H3_CONFIG_ROOT=/workspace/configs/minimax-h3
export H3_OUTPUT_ROOT=/workspace/output/minimax-h3-nsync-self-flow
export H3_WORKFLOW_API="$DP_ROOT/examples/minimax_h3_t2va_api.json"
```

## 4. Create the Python environment

Use a virtual environment that can see the template's CUDA-enabled PyTorch instead of replacing it with an unrelated wheel:

```bash
mkdir -p /workspace/venvs
python -m venv --system-site-packages "$DP_VENV"
source "$DP_VENV/bin/activate"
```

After step 4, every new terminal needs the path-export block from step 3 followed by:

```bash
source "$DP_VENV/bin/activate"
```

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install diffusion-pipe, the pinned ComfyUI submodule, and the Hugging Face download CLI:

```bash
cd "$DP_ROOT"
python -m pip install --no-cache-dir -r requirements.txt
```

```bash
python -m pip install --no-cache-dir -r submodules/ComfyUI/requirements.txt
```

```bash
python -m pip install --no-cache-dir huggingface_hub
```

Verify the environment:

```bash
python -m pip check
```

```bash
python -c "import importlib.metadata as m; import torch, deepspeed, comfy_kitchen; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); print('deepspeed:', deepspeed.__version__); print('comfy-kitchen:', m.version('comfy-kitchen'))"
```

```bash
ffmpeg -version | head -n 1
ffprobe -version | head -n 1
```

Flash Attention is not required for this H3 setup. Do not add optional attention packages until the baseline works.

## 5. Download the MiniMax H3 model files once

Review the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading the weights.

Create the shared model directory:

```bash
mkdir -p "$H3_MODEL_ROOT"
```

Download the pruned FL2VA diffusion checkpoint, the MiniMax Qwen3-VL text encoder, and both VAEs:

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir "$H3_MODEL_ROOT"
```

Interrupted `hf download` commands are resumable; run the same command again.

Download LightX2V's four-step v1.0 ComfyUI LoRA:

```bash
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors \
  --local-dir "$H3_MODEL_ROOT/loras"
```

Confirm the files:

```bash
find "$H3_MODEL_ROOT" -type f -name '*.safetensors' -printf '%s %p\n' | sort -n
```

```bash
du -sh "$H3_MODEL_ROOT"
```

Make the same files visible to ComfyUI without duplicating them:

```bash
mkdir -p \
  "$DP_ROOT/submodules/ComfyUI/models/diffusion_models" \
  "$DP_ROOT/submodules/ComfyUI/models/text_encoders" \
  "$DP_ROOT/submodules/ComfyUI/models/vae" \
  "$DP_ROOT/submodules/ComfyUI/models/loras"
```

```bash
ln -sfn \
  "$H3_MODEL_ROOT/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "$DP_ROOT/submodules/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
```

```bash
ln -sfn \
  "$H3_MODEL_ROOT/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
  "$DP_ROOT/submodules/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
```

```bash
ln -sfn \
  "$H3_MODEL_ROOT/vae/minimax_h3_video_vae_fp16.safetensors" \
  "$DP_ROOT/submodules/ComfyUI/models/vae/minimax_h3_video_vae_fp16.safetensors"
```

```bash
ln -sfn \
  "$H3_MODEL_ROOT/vae/minimax_h3_audio_vae_fp32.safetensors" \
  "$DP_ROOT/submodules/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors"
```

```bash
ln -sfn \
  "$H3_TURBO_LORA" \
  "$DP_ROOT/submodules/ComfyUI/models/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
```

Verify the links:

```bash
find "$DP_ROOT/submodules/ComfyUI/models" -maxdepth 2 -type l -ls
```

## 6. Prepare and upload the positive dataset

Create the dataset directories:

```bash
mkdir -p "$H3_DATA_ROOT/positive" "$H3_DATA_ROOT/negative"
```

Every positive media file needs a same-stem `.txt` caption, unless the directory uses a standard `captions.json`:

```text
/workspace/data/minimax-h3/positive/
├── shot_001.mp4
├── shot_001.txt
├── shot_002.mp4
└── shot_002.txt
```

For a character LoRA whose trigger is `TOKperson`, `shot_001.txt` might contain:

```text
TOKperson walking through a train station at sunset
```

The positive caption is the **training caption** and must keep the trigger. The negative generator will remove `TOKperson` only from the prompt it sends to H3. The negative dataset will still reuse the unmodified positive caption.

Upload media and captions using JupyterLab, RunPod's Cloud Sync, or SSH. For example, run this on your own computer using the IP and TCP port shown under the Pod's **Connect** dialog:

```bash
rsync -avP -e 'ssh -p RUNPOD_SSH_PORT' /local/path/to/positive/ root@RUNPOD_IP:/workspace/data/minimax-h3/positive/
```

Back in the Pod, inspect the upload:

```bash
find "$H3_DATA_ROOT/positive" -maxdepth 1 -type f -printf '%f\n' | sort
```

Do not put two media files with the same stem in the positive directory. For example, `shot_001.png` and `shot_001.mp4` cannot coexist because both would pair with the same negative stem.

## 7. Start the repository's local ComfyUI

Create persistent output and log directories:

```bash
mkdir -p /workspace/comfy-output /workspace/logs
```

Start ComfyUI in a detached `tmux` session:

```bash
if curl --fail --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo 'The existing ComfyUI server is already running.'
else
  tmux new-session -d -s comfy \
    "$COMFYUI_PYTHON $COMFYUI_ROOT/main.py --listen 0.0.0.0 --port 8188 --output-directory /workspace/comfy-output 2>&1 | tee /workspace/logs/comfy.log"
fi
```

This guide runs the pinned ComfyUI submodule with the diffusion-pipe virtual environment. If you use a separate ComfyUI installation or virtual environment, set `COMFYUI_ROOT` and `COMFYUI_PYTHON` to those paths before running the command.

Watch startup:

```bash
tail -f /workspace/logs/comfy.log
```

Press `Ctrl-C` to stop following the log; this does not stop ComfyUI. Check its local API:

```bash
curl --fail http://127.0.0.1:8188/system_stats | jq '.system, .devices'
```

In RunPod, open **Connect > HTTP Service [Port 8188]**.

### Use the ready API workflow

The repository includes [`examples/minimax_h3_t2va_api.json`](../examples/minimax_h3_t2va_api.json). It is already in ComfyUI API format and contains local text-only H3 conditioning, LightX2V's four-step v1.0 Turbo LoRA at strength 1.0, its required 6/3 video/audio sigma shift, Euler with six sampling steps, video and audio decoding, 24 fps muxing, and exactly one `SaveVideo` output. Its defaults are 736×416 (approximately 0.3 MP) and 56 frames (approximately 2.33 seconds, because H3 snaps a two-second request upward to its `17n+5` frame grid). It has no first-frame, last-frame, reference, or hosted MiniMax API connections.

`H3_WORKFLOW_API` points directly at the tracked workflow. The standard filenames downloaded in step 5 match its loader inputs. Confirm the graph is ready:

```bash
test -s "$H3_WORKFLOW_API" \
  && jq -e '.["5"].class_type == "MiniMaxH3ImageToVideo" and .["9"].inputs.steps == 6 and .["15"].class_type == "LoraLoaderModelOnly" and .["16"].class_type == "MiniMaxH3SigmaShift" and .["14"].class_type == "SaveVideo"' "$H3_WORKFLOW_API" \
  && echo "API workflow ready: $H3_WORKFLOW_API"
```

No manual ComfyUI export is needed. The generator changes the prompt, width, height, length, seed, and filename prefix for every source file while leaving the tested sampling and AV decode graph intact.

Mixed image/video datasets are handled per file. Image positives produce same-stem one-frame `.png` negatives; H3 generates its minimum five-frame latent internally and the utility extracts one frame. Video positives produce `.mp4` negatives normalized to the positive video's dimensions, duration, frame count, and audio presence.

You may omit `--workflow` because the generator defaults to this tracked graph. If your ComfyUI loader names differ from step 5, pass the five model files through the generator's explicit model arguments. The [manual ComfyUI workflow guide](minimax_h3_nsync_negative_generation.md#set-up-the-local-comfyui-workflow-manually) shows both the standard-filename and fully explicit forms.

## 8. Generate the NSYNC negatives automatically

First inspect the generator's options:

```bash
cd "$DP_ROOT"
python tools/generate_minimax_h3_nsync_negatives.py --help
```

Run a dry run. Replace `TOKperson` with the exact character, concept, or style phrase that must be omitted from generated negatives:

```bash
python tools/generate_minimax_h3_nsync_negatives.py \
  "$H3_DATA_ROOT/positive" \
  "$H3_DATA_ROOT/negative" \
  --workflow "$H3_WORKFLOW_API" \
  --remove-text 'TOKperson' \
  --generation-megapixels 0.3 \
  --dry-run
```

Check every printed generation prompt. The cleaned prompt should preserve the scene/content while removing the target. Repeat `--remove-text` when captions contain more than one target phrase:

```bash
python tools/generate_minimax_h3_nsync_negatives.py \
  "$H3_DATA_ROOT/positive" \
  "$H3_DATA_ROOT/negative" \
  --workflow "$H3_WORKFLOW_API" \
  --remove-text 'TOKperson' \
  --remove-text 'TOKstyle' \
  --generation-megapixels 0.3 \
  --dry-run
```

`0.3` megapixels matches the bundled workflow and is a conservative starting point for a 24 GB GPU. On a larger GPU, increase it if desired. The final negative is normalized to the positive's dimensions and duration either way.

Generate one real pair before committing to the whole dataset:

```bash
python tools/generate_minimax_h3_nsync_negatives.py \
  "$H3_DATA_ROOT/positive" \
  "$H3_DATA_ROOT/negative" \
  --workflow "$H3_WORKFLOW_API" \
  --remove-text 'TOKperson' \
  --generation-megapixels 0.3 \
  --limit 1
```

Inspect that file visually and confirm its audio when the positive has audio. Then generate the entire negative dataset:

```bash
tmux new -s h3-negatives
```

Inside that `tmux` session, run:

```bash
cd "$DP_ROOT"
python tools/generate_minimax_h3_nsync_negatives.py \
  "$H3_DATA_ROOT/positive" \
  "$H3_DATA_ROOT/negative" \
  --workflow "$H3_WORKFLOW_API" \
  --remove-text 'TOKperson' \
  --generation-megapixels 0.3 \
  2>&1 | tee /workspace/logs/h3-negative-generation.log
```

Detach without stopping it by pressing `Ctrl-B`, then `D`. Reattach later:

```bash
tmux attach -t h3-negatives
```

The generator runs H3 jobs sequentially and writes `.nsync_generation_manifest.json` in the negative directory. If the session or Pod is interrupted, start ComfyUI and run the exact same generator command again. Valid completed negatives are checked and skipped. Do not add `--overwrite` unless replacement is intentional.

For detailed media guarantees and error messages, see [Generating MiniMax H3 NSYNC negatives with local ComfyUI](minimax_h3_nsync_negative_generation.md).

## 9. Stop ComfyUI and release its GPU memory

After every negative exists, stop the local ComfyUI process before caching or training:

```bash
if tmux has-session -t comfy 2>/dev/null; then
  tmux kill-session -t comfy
fi
```

If ComfyUI was already running before this guide, stop it with that installation's normal launcher or service command. Then confirm that its API is no longer reachable; this block deliberately fails while ComfyUI is still using the GPU:

```bash
if curl --fail --max-time 2 http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo 'ComfyUI is still running. Stop it before training.' >&2
  false
else
  echo 'ComfyUI is stopped.'
fi
```

```bash
nvidia-smi
```

Do not run ComfyUI negative generation and diffusion-pipe training simultaneously on the same GPU.

## 10. Create the training and dataset configurations

Copy the repository's combined examples into persistent storage:

```bash
mkdir -p "$H3_CONFIG_ROOT" "$H3_OUTPUT_ROOT"
cp "$DP_ROOT/examples/minimax_h3_nsync_self_flow.toml" "$H3_CONFIG_ROOT/train.toml"
cp "$DP_ROOT/examples/minimax_h3_nsync_self_flow_dataset.toml" "$H3_CONFIG_ROOT/dataset.toml"
```

Fill in the training paths:

```bash
sed -i \
  -e "s|output_dir = 'path_to_output_dir'|output_dir = '$H3_OUTPUT_ROOT'|" \
  -e "s|dataset = 'examples/minimax_h3_nsync_self_flow_dataset.toml'|dataset = '$H3_CONFIG_ROOT/dataset.toml'|" \
  -e "s|/path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors|$H3_DIFFUSION_MODEL|" \
  -e "s|/path/to/minimax_h3_video_vae_fp16.safetensors|$H3_VIDEO_VAE|" \
  -e "s|/path/to/minimax_h3_audio_vae_fp32.safetensors|$H3_AUDIO_VAE|" \
  -e "s|/path/to/qwen3vl_32b_minimax_h3_int8_convrot.safetensors|$H3_TEXT_ENCODER|" \
  "$H3_CONFIG_ROOT/train.toml"
```

Fill in the paired dataset paths:

```bash
sed -i \
  -e "s|/path/to/target_positive_media|$H3_DATA_ROOT/positive|g" \
  -e "s|/path/to/generated_negative_media|$H3_DATA_ROOT/negative|g" \
  "$H3_CONFIG_ROOT/dataset.toml"
```

Check for any remaining placeholders:

```bash
if grep -nEv '^[[:space:]]*#' "$H3_CONFIG_ROOT/train.toml" "$H3_CONFIG_ROOT/dataset.toml" | grep -E "path_to_|/path/to/|your_dataset"; then
  echo 'Replace the placeholders printed above.'
else
  echo 'No path placeholders remain.'
fi
```

Review the complete files:

```bash
sed -n '1,220p' "$H3_CONFIG_ROOT/train.toml"
```

```bash
sed -n '1,180p' "$H3_CONFIG_ROOT/dataset.toml"
```

Keep these constraints intact:

- `pipeline_stages = 1` because Self-Flow's teacher branch is forward-only;
- one GPU and therefore data-parallel world size 1, as NSYNC requires;
- `micro_batch_size_per_gpu = 1` for video;
- `gradient_accumulation_steps` describes logical positive batches; NSYNC internally runs positive, negative, and anchor passes;
- `uncond_fraction` must remain at its default `0`;
- `optimizer.gradient_release` must remain disabled;
- `compile = false`; it is also disabled automatically for these dynamic methods;
- the positive and negative directories use the same `nsync_pair`; and
- the negative directory's `caption_path` points to the positive directory.

Validate the configuration before loading a model:

```bash
cd "$DP_ROOT"
python train.py --validate_only --config "$H3_CONFIG_ROOT/train.toml"
```

Do not start the expensive cache phase until this succeeds.

## 11. Cache latents and text embeddings in a separate process

H3's text encoder is large. Cache first and exit so that no residual text-encoder allocation carries into training:

```bash
tmux new -s h3-cache
```

Inside the `h3-cache` session, run:

```bash
cd "$DP_ROOT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py \
  --deepspeed \
  --config "$H3_CONFIG_ROOT/train.toml" \
  --cache_only \
  2>&1 | tee /workspace/logs/h3-cache.log
```

Detach with `Ctrl-B`, then `D`. Follow progress from another terminal:

```bash
tail -f /workspace/logs/h3-cache.log
```

The cache is stored under `cache` directories alongside the dataset media. If media, captions, or bucket settings change later, rebuild with `--regenerate_cache`.

## 12. Start NSYNC + Self-Flow LoRA training

Wait for the cache-only command to exit successfully. Then open a fresh `tmux` session:

```bash
tmux new -s h3-train
```

Inside the training session, run:

```bash
cd "$DP_ROOT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py \
  --deepspeed \
  --config "$H3_CONFIG_ROOT/train.toml" \
  --trust_cache \
  2>&1 | tee /workspace/logs/h3-training.log
```

Detach with `Ctrl-B`, then `D`. Reattach with:

```bash
tmux attach -t h3-train
```

The first logical optimizer step performs sequential positive, negative, and anchor passes. It is therefore substantially slower than an ordinary one-pass LoRA step. Saved inference LoRAs use Self-Flow's EMA adapter under the normal LoRA keys and remain directly usable in ComfyUI.

Training runs, saved LoRAs, and DeepSpeed checkpoints will appear under:

```text
/workspace/output/minimax-h3-nsync-self-flow/
```

## 13. Monitor and resume

Watch GPU utilization:

```bash
nvidia-smi dmon
```

Watch the log:

```bash
tail -f /workspace/logs/h3-training.log
```

Optionally start TensorBoard in another detached session:

```bash
tmux new-session -d -s tensorboard \
  "$DP_VENV/bin/tensorboard --logdir $H3_OUTPUT_ROOT --bind_all --port 6006"
```

Open RunPod's HTTP service for port `6006` if that port was exposed in the Pod template.

To resume the latest DeepSpeed checkpoint, stop any stale training process and run the same command with `--resume_from_checkpoint`:

```bash
cd "$DP_ROOT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py \
  --deepspeed \
  --config "$H3_CONFIG_ROOT/train.toml" \
  --trust_cache \
  --resume_from_checkpoint
```

The config passed on the command line controls a resumed run. Keep the original `train.toml` and `dataset.toml` with the outputs.

## Troubleshooting

### The Pod reports a different PyTorch or CUDA version

The selected template/image is not the one in this guide. Recreate the Pod with `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404` and verify it before installing dependencies. Do not diagnose H3 kernels until the version check in step 2 passes.

### ComfyUI does not list the H3 nodes

When using an existing ComfyUI installation, confirm that it contains the local H3 nodes:

```bash
test -f "$COMFYUI_ROOT/comfy_extras/nodes_minimax_h3.py" \
  && echo 'MiniMax H3 nodes are present' \
  || echo 'Update this ComfyUI installation using its normal update method'
```

The separate pinned submodule used by diffusion-pipe can be initialized with:

```bash
cd "$DP_ROOT"
git submodule update --init --recursive
git -C submodules/ComfyUI rev-parse HEAD
```

Restart ComfyUI and inspect `/workspace/logs/comfy.log`.

### ComfyUI does not list the model files

Check the model paths configured in step 3:

```bash
printf '%s\n' "$H3_DIFFUSION_MODEL" "$H3_TEXT_ENCODER" "$H3_VIDEO_VAE" "$H3_AUDIO_VAE"
test -f "$H3_DIFFUSION_MODEL" \
  && test -f "$H3_TEXT_ENCODER" \
  && test -f "$H3_VIDEO_VAE" \
  && test -f "$H3_AUDIO_VAE"
```

Restart ComfyUI after adding or changing model files.

### ComfyUI runs out of VRAM during negative generation

Stop ComfyUI, restart its command with `--lowvram`, and keep `--generation-megapixels 0.3` in the generator command. On 24 GB cards, offloading the 25 GiB text encoder is expected and prompt encoding can be slow. A 48 GB GPU makes this stage much smoother.

### Cache-only finishes, but training immediately runs out of memory

Make sure the cache-only process and ComfyUI have both exited, then start training in a fresh process. On a 24 GB GPU:

1. keep `blocks_to_swap = 48` and `activation_checkpointing = 'unsloth'`;
2. change `ema_dtype = 'float32'` to `ema_dtype = 'bfloat16'`; and
3. lower `image_micro_batch_size_per_gpu` to `1` if the dataset contains images.

Do not lower the video microbatch below 1.

### The generator rejects the workflow as UI format

Confirm that `H3_WORKFLOW_API` points to `examples/minimax_h3_t2va_api.json`; the bundled workflow is already in API format. If you substituted a custom graph, enable developer-mode options in ComfyUI and export **Save (API Format)**; a normal UI workflow cannot be passed directly to `--workflow`.

### The generator says the positive has audio but the output does not

The API workflow does not decode or connect H3's audio output to `CreateVideo`/`SaveVideo`. Fix the workflow, export it again in API format, and retry. The generator intentionally refuses silent negatives for audio-positive pairs.

### A training pair is missing or falls into a different bucket

Do not rename generated files. Positives and negatives pair by filename stem, must have the same media type, and must resolve to the same dimension/frame bucket. The generator normalizes these properties automatically; hand-edited negatives can break them.

### Captions or media changed after caching

Delete nothing manually. Re-run the cache-only command with `--regenerate_cache`, then restart training without trusting the old cache.

### The Pod is about to be deleted

Verify that models, data, configs, logs, caches, and outputs are all under `/workspace`. A Network Volume survives Pod deletion; a Volume Disk does not. Back up final LoRAs and important checkpoints outside RunPod as well.
