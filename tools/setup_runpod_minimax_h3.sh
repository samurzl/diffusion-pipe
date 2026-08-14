#!/usr/bin/env bash
# Idempotent RunPod bootstrap for MiniMax H3 training with diffusion-pipe.
#
# The persistent /workspace volume holds the versioned Python environment,
# repository, data, and outputs. An existing ComfyUI installation supplies the
# H3 model files directly; this script neither modifies ComfyUI nor downloads
# duplicate checkpoints.

set -Eeuo pipefail

SETUP_SCHEMA_VERSION=1
EXPECTED_TORCH_VERSION=2.9.1
PREFERRED_CUDA_VERSION=13.0
FALLBACK_CUDA_VERSION=12.8

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DP_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
COMFYUI_ROOT_ARG=""
COMFYUI_PYTHON_ARG=""
DIFFUSION_MODEL_ARG=""
TEXT_ENCODER_ARG=""
VIDEO_VAE_ARG=""
AUDIO_VAE_ARG=""
TURBO_LORA_ARG=""
SKIP_SYSTEM_PACKAGES=0
ALLOW_TEMPLATE_MISMATCH=0
ALLOW_NONPERSISTENT_WORKSPACE=0
FORCE_PYTHON_INSTALL=0
REBUILD_VENV=0

log() {
    printf '[minimax-h3-setup] %s\n' "$*"
}

warn() {
    printf '[minimax-h3-setup] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[minimax-h3-setup] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: tools/setup_runpod_minimax_h3.sh [options]

Build or reuse a persistent diffusion-pipe Python environment on a RunPod Pod,
and discover MiniMax H3 model files plus the LightX2V Turbo LoRA in an existing
ComfyUI installation.

Options:
  --comfy-root PATH              Existing ComfyUI directory containing main.py
                                 and models/. Common /workspace paths are found
                                 automatically when this is omitted.
  --comfy-python PATH            Python executable used to launch that ComfyUI.
                                 Its .venv/venv is detected when possible.
  --diffusion-model PATH         Override the discovered FL2VA checkpoint.
  --text-encoder PATH            Override the discovered H3 text encoder.
  --video-vae PATH               Override the discovered H3 video VAE.
  --audio-vae PATH               Override the discovered H3 audio VAE.
  --turbo-lora PATH              Override the discovered LightX2V MiniMax H3
                                 Turbo 4-step ComfyUI LoRA.
  --workspace PATH               Persistent mount root (default: /workspace).
  --skip-system-packages         Do not install missing apt packages.
  --force-python-install         Re-run pip even when the cached environment
                                 signature and import check are healthy.
  --rebuild-venv                 Move the matching versioned venv to a timestamped
                                 backup and create it again.
  --allow-template-mismatch      Continue with a PyTorch/CUDA version other than
                                 the documented RunPod baseline.
  --allow-nonpersistent-workspace
                                 Continue when PATH appears to share the root
                                 container filesystem. Intended for testing only.
  -h, --help                     Show this help.

The script writes /workspace/workflows/minimax_h3_t2va_api.json and
/workspace/minimax-h3-env.sh (under the selected workspace). Source the
environment file after setup and in every new shell.
EOF
}

require_option_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "$value" ]] || die "$option requires a value"
}

while (($#)); do
    case "$1" in
        --comfy-root)
            require_option_value "$1" "${2:-}"
            COMFYUI_ROOT_ARG="$2"
            shift 2
            ;;
        --comfy-python)
            require_option_value "$1" "${2:-}"
            COMFYUI_PYTHON_ARG="$2"
            shift 2
            ;;
        --diffusion-model)
            require_option_value "$1" "${2:-}"
            DIFFUSION_MODEL_ARG="$2"
            shift 2
            ;;
        --text-encoder)
            require_option_value "$1" "${2:-}"
            TEXT_ENCODER_ARG="$2"
            shift 2
            ;;
        --video-vae)
            require_option_value "$1" "${2:-}"
            VIDEO_VAE_ARG="$2"
            shift 2
            ;;
        --audio-vae)
            require_option_value "$1" "${2:-}"
            AUDIO_VAE_ARG="$2"
            shift 2
            ;;
        --turbo-lora)
            require_option_value "$1" "${2:-}"
            TURBO_LORA_ARG="$2"
            shift 2
            ;;
        --workspace)
            require_option_value "$1" "${2:-}"
            WORKSPACE_ROOT="$2"
            shift 2
            ;;
        --skip-system-packages)
            SKIP_SYSTEM_PACKAGES=1
            shift
            ;;
        --force-python-install)
            FORCE_PYTHON_INSTALL=1
            shift
            ;;
        --rebuild-venv)
            REBUILD_VENV=1
            shift
            ;;
        --allow-template-mismatch)
            ALLOW_TEMPLATE_MISMATCH=1
            shift
            ;;
        --allow-nonpersistent-workspace)
            ALLOW_NONPERSISTENT_WORKSPACE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1 (run with --help)"
            ;;
    esac
done

[[ -f "$DP_ROOT/requirements.txt" ]] || die "Could not locate diffusion-pipe requirements at $DP_ROOT"
[[ -f "$DP_ROOT/submodules/ComfyUI/requirements.txt" ]] || {
    [[ -d "$DP_ROOT/.git" ]] || die "$DP_ROOT is not a Git checkout"
}

mkdir -p "$WORKSPACE_ROOT"
WORKSPACE_ROOT="$(cd -- "$WORKSPACE_ROOT" && pwd -P)"
[[ -w "$WORKSPACE_ROOT" ]] || die "Workspace is not writable: $WORKSPACE_ROOT"

if command -v findmnt >/dev/null 2>&1; then
    workspace_source="$(findmnt -n -o SOURCE -T "$WORKSPACE_ROOT" 2>/dev/null || true)"
    root_source="$(findmnt -n -o SOURCE -T / 2>/dev/null || true)"
    if [[ -n "$workspace_source" && "$workspace_source" == "$root_source" ]]; then
        if ((ALLOW_NONPERSISTENT_WORKSPACE)); then
            warn "$WORKSPACE_ROOT appears to be on the ephemeral container filesystem"
        else
            die "$WORKSPACE_ROOT is not a separate persistent mount. Attach the Network Volume, or pass --allow-nonpersistent-workspace only for testing."
        fi
    fi
else
    warn "findmnt is unavailable; persistent workspace mounting could not be verified"
fi

install_system_packages() {
    local -a packages=()
    command -v git >/dev/null 2>&1 || packages+=(git)
    command -v git-lfs >/dev/null 2>&1 || packages+=(git-lfs)
    command -v ffmpeg >/dev/null 2>&1 || packages+=(ffmpeg)
    command -v ffprobe >/dev/null 2>&1 || packages+=(ffmpeg)
    command -v curl >/dev/null 2>&1 || packages+=(curl)
    command -v wget >/dev/null 2>&1 || packages+=(wget)
    command -v jq >/dev/null 2>&1 || packages+=(jq)
    command -v tmux >/dev/null 2>&1 || packages+=(tmux)
    command -v rsync >/dev/null 2>&1 || packages+=(rsync)

    if ((${#packages[@]} == 0)); then
        log "System tools are already installed"
        return
    fi
    if ((SKIP_SYSTEM_PACKAGES)); then
        die "Required system tools are missing and --skip-system-packages was used: ${packages[*]}"
    fi
    [[ "$(id -u)" == 0 ]] || die "Run as root so missing apt packages can be installed: ${packages[*]}"

    local -a unique_packages=()
    while IFS= read -r package; do
        unique_packages+=("$package")
    done < <(printf '%s\n' "${packages[@]}" | sort -u)
    packages=("${unique_packages[@]}")
    log "Installing missing system packages: ${packages[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
}

install_system_packages
git lfs install >/dev/null

log "Initializing the repository's pinned ComfyUI code dependency"
git -C "$DP_ROOT" submodule sync --recursive
git -C "$DP_ROOT" submodule update --init --recursive
[[ -f "$DP_ROOT/submodules/ComfyUI/requirements.txt" ]] || die "Pinned ComfyUI submodule is incomplete"

command -v python >/dev/null 2>&1 || die "python is not available in the Pod template"
runtime_info="$(python - <<'PY'
import platform
import torch

print(platform.python_version())
print(torch.__version__)
print(torch.version.cuda or "none")
print("true" if torch.cuda.is_available() else "false")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
)"

PYTHON_VERSION="$(printf '%s\n' "$runtime_info" | sed -n '1p')"
TORCH_VERSION="$(printf '%s\n' "$runtime_info" | sed -n '2p')"
CUDA_VERSION="$(printf '%s\n' "$runtime_info" | sed -n '3p')"
CUDA_AVAILABLE="$(printf '%s\n' "$runtime_info" | sed -n '4p')"
GPU_NAME="$(printf '%s\n' "$runtime_info" | sed -n '5p')"
PYTHON_VERSION="${PYTHON_VERSION:-unknown}"
TORCH_VERSION="${TORCH_VERSION:-unknown}"
CUDA_VERSION="${CUDA_VERSION:-none}"
CUDA_AVAILABLE="${CUDA_AVAILABLE:-false}"
GPU_NAME="${GPU_NAME:-none}"
TORCH_BASE_VERSION="${TORCH_VERSION%%+*}"

[[ "$CUDA_AVAILABLE" == true ]] || die "PyTorch cannot access CUDA; select the documented GPU Pod template"
if [[ "$TORCH_BASE_VERSION" != "$EXPECTED_TORCH_VERSION" ]]; then
    if ((ALLOW_TEMPLATE_MISMATCH)); then
        warn "Expected PyTorch $EXPECTED_TORCH_VERSION, found $TORCH_VERSION"
    else
        die "Expected PyTorch $EXPECTED_TORCH_VERSION, found $TORCH_VERSION. Use runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404 or pass --allow-template-mismatch."
    fi
fi
case "$CUDA_VERSION" in
    "$PREFERRED_CUDA_VERSION")
        ;;
    "$FALLBACK_CUDA_VERSION")
        warn "Using CUDA $CUDA_VERSION fallback; CUDA $PREFERRED_CUDA_VERSION is preferred for H3 INT8 ConvRot"
        ;;
    *)
        if ((ALLOW_TEMPLATE_MISMATCH)); then
            warn "Expected CUDA $PREFERRED_CUDA_VERSION (or fallback $FALLBACK_CUDA_VERSION), found $CUDA_VERSION"
        else
            die "Expected CUDA $PREFERRED_CUDA_VERSION (or fallback $FALLBACK_CUDA_VERSION), found $CUDA_VERSION. Pass --allow-template-mismatch to override."
        fi
        ;;
esac
log "Runtime: Python $PYTHON_VERSION, PyTorch $TORCH_VERSION, CUDA $CUDA_VERSION, GPU $GPU_NAME"

comfy_has_models() {
    local root="$1"
    [[ -f "$root/main.py" && -d "$root/models" ]] || return 1
    find -H "$root/models/diffusion_models" -maxdepth 3 \( -type f -o -type l \) -name 'minimax_h3_fl2va*.safetensors' -print -quit 2>/dev/null | grep -q . || return 1
    find -H "$root/models/text_encoders" -maxdepth 3 \( -type f -o -type l \) -name 'qwen3vl_32b_minimax_h3*.safetensors' -print -quit 2>/dev/null | grep -q . || return 1
    [[ -f "$root/models/vae/minimax_h3_video_vae_fp16.safetensors" ]] || return 1
    [[ -f "$root/models/vae/minimax_h3_audio_vae_fp32.safetensors" ]] || return 1
}

find_comfy_root() {
    local -a candidates=()
    local candidate main_py
    if [[ -n "$COMFYUI_ROOT_ARG" ]]; then
        [[ -d "$COMFYUI_ROOT_ARG" ]] || die "ComfyUI directory does not exist: $COMFYUI_ROOT_ARG"
        candidate="$(cd -- "$COMFYUI_ROOT_ARG" && pwd -P)"
        [[ -f "$candidate/main.py" && -d "$candidate/models" ]] || die "ComfyUI root must contain main.py and models/: $candidate"
        printf '%s\n' "$candidate"
        return 0
    fi
    [[ -n "${COMFYUI_ROOT:-}" ]] && candidates+=("$COMFYUI_ROOT")
    candidates+=(
        "$WORKSPACE_ROOT/ComfyUI"
        "$WORKSPACE_ROOT/comfyui/ComfyUI"
        "$WORKSPACE_ROOT/comfyui"
        "$WORKSPACE_ROOT/runpod-slim/ComfyUI"
    )

    for candidate in "${candidates[@]}"; do
        [[ -d "$candidate" ]] || continue
        candidate="$(cd -- "$candidate" && pwd -P)"
        if comfy_has_models "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    while IFS= read -r main_py; do
        candidate="$(dirname -- "$main_py")"
        [[ "$candidate" == "$DP_ROOT/submodules/ComfyUI" ]] && continue
        if comfy_has_models "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(find "$WORKSPACE_ROOT" -maxdepth 5 -type f -name main.py -path '*/ComfyUI/main.py' -print 2>/dev/null)

    return 1
}

if ! COMFYUI_ROOT="$(find_comfy_root)"; then
    die "Could not find an existing ComfyUI with MiniMax H3 models. Pass --comfy-root /workspace/path/to/ComfyUI."
fi
log "Using existing ComfyUI: $COMFYUI_ROOT"

find_first_model() {
    local directory="$1"
    shift
    local name path
    for name in "$@"; do
        path="$(find -H "$directory" -maxdepth 3 \( -type f -o -type l \) -name "$name" -print -quit 2>/dev/null || true)"
        if [[ -n "$path" && -f "$path" ]]; then
            readlink -f -- "$path"
            return 0
        fi
    done
    return 1
}

resolve_override() {
    local label="$1"
    local override="$2"
    if [[ -n "$override" ]]; then
        [[ -f "$override" ]] || die "$label does not exist: $override"
        readlink -f -- "$override"
        return 0
    fi
    return 1
}

H3_DIFFUSION_MODEL="$(resolve_override "Diffusion model" "$DIFFUSION_MODEL_ARG" || find_first_model \
    "$COMFYUI_ROOT/models/diffusion_models" \
    minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
    minimax_h3_fl2va_int8_convrot.safetensors \
    minimax_h3_fl2va_pruned_bf16.safetensors \
    minimax_h3_fl2va_bf16.safetensors)" || die "No supported MiniMax H3 FL2VA diffusion model was found"

H3_TEXT_ENCODER="$(resolve_override "Text encoder" "$TEXT_ENCODER_ARG" || find_first_model \
    "$COMFYUI_ROOT/models/text_encoders" \
    qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
    qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
    qwen3vl_32b_minimax_h3_bf16.safetensors)" || die "No supported MiniMax H3 text encoder was found"

H3_VIDEO_VAE="$(resolve_override "Video VAE" "$VIDEO_VAE_ARG" || find_first_model \
    "$COMFYUI_ROOT/models/vae" minimax_h3_video_vae_fp16.safetensors)" || die "MiniMax H3 video VAE was not found"

H3_AUDIO_VAE="$(resolve_override "Audio VAE" "$AUDIO_VAE_ARG" || find_first_model \
    "$COMFYUI_ROOT/models/vae" minimax_h3_audio_vae_fp32.safetensors)" || die "MiniMax H3 audio VAE was not found"

H3_TURBO_LORA="$(resolve_override "LightX2V Turbo LoRA" "$TURBO_LORA_ARG" || find_first_model \
    "$COMFYUI_ROOT/models/loras" \
    minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors)" || die "The LightX2V MiniMax H3 Turbo 4-step ComfyUI LoRA was not found under $COMFYUI_ROOT/models/loras. Download minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors from lightx2v/Minimax-h3-Turbo, or pass --turbo-lora PATH."

log "Diffusion model: $H3_DIFFUSION_MODEL"
log "Text encoder:   $H3_TEXT_ENCODER"
log "Video VAE:      $H3_VIDEO_VAE"
log "Audio VAE:      $H3_AUDIO_VAE"
log "Turbo LoRA:     $H3_TURBO_LORA"

if [[ -n "$COMFYUI_PYTHON_ARG" ]]; then
    [[ -x "$COMFYUI_PYTHON_ARG" ]] || die "ComfyUI Python is not executable: $COMFYUI_PYTHON_ARG"
    COMFYUI_PYTHON="$(cd -- "$(dirname -- "$COMFYUI_PYTHON_ARG")" && pwd -P)/$(basename -- "$COMFYUI_PYTHON_ARG")"
else
    COMFYUI_PYTHON=""
    for candidate in \
        "$COMFYUI_ROOT/.venv/bin/python" \
        "$COMFYUI_ROOT/venv/bin/python" \
        "$WORKSPACE_ROOT/venv/bin/python"; do
        if [[ -x "$candidate" ]]; then
            COMFYUI_PYTHON="$(cd -- "$(dirname -- "$candidate")" && pwd -P)/$(basename -- "$candidate")"
            break
        fi
    done
fi

runtime_id="$(python - "$PYTHON_VERSION" "$TORCH_BASE_VERSION" "$CUDA_VERSION" <<'PY'
import re
import sys

def clean(value):
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")

print(f"py{clean(sys.argv[1])}-torch{clean(sys.argv[2])}-cu{clean(sys.argv[3])}")
PY
)"
DP_VENV="$WORKSPACE_ROOT/venvs/diffusion-pipe-$runtime_id"
mkdir -p "$WORKSPACE_ROOT/venvs"

if ((REBUILD_VENV)) && [[ -e "$DP_VENV" ]]; then
    backup_path="$DP_VENV.backup.$(date -u +%Y%m%dT%H%M%SZ)"
    log "Moving the existing environment to $backup_path"
    mv -- "$DP_VENV" "$backup_path"
fi

if [[ ! -x "$DP_VENV/bin/python" ]]; then
    [[ ! -e "$DP_VENV" ]] || die "$DP_VENV exists but is not a usable venv; pass --rebuild-venv to move it aside"
    log "Creating persistent Python environment: $DP_VENV"
    python -m venv --system-site-packages "$DP_VENV"
    FORCE_PYTHON_INSTALL=1
fi

signature_payload="$({
    printf 'schema=%s\n' "$SETUP_SCHEMA_VERSION"
    printf 'runtime=%s\n' "$runtime_id"
    sha256sum "$DP_ROOT/requirements.txt"
    sha256sum "$DP_ROOT/submodules/ComfyUI/requirements.txt"
    git -C "$DP_ROOT/submodules/ComfyUI" rev-parse HEAD
} 2>/dev/null)"
desired_signature="$(printf '%s' "$signature_payload" | sha256sum | awk '{print $1}')"
signature_file="$DP_VENV/.diffusion-pipe-setup-signature"
if [[ -f "$signature_file" ]]; then
    installed_signature="$(<"$signature_file")"
else
    installed_signature=""
fi

environment_is_healthy() {
    "$DP_VENV/bin/python" - "$TORCH_BASE_VERSION" <<'PY' >/dev/null 2>&1
import av
import comfy_kitchen
import datasets
import deepspeed
import optimi
import peft
import safetensors
import sys
import torch
import transformers

assert torch.cuda.is_available()
assert torch.__version__.split("+", 1)[0] == sys.argv[1]
PY
}

if [[ "$installed_signature" != "$desired_signature" ]] || ! environment_is_healthy; then
    FORCE_PYTHON_INSTALL=1
fi

if ((FORCE_PYTHON_INSTALL)); then
    log "Installing or refreshing diffusion-pipe Python dependencies"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$DP_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$DP_VENV/bin/python" -m pip install --no-cache-dir -r "$DP_ROOT/requirements.txt"
    PIP_DISABLE_PIP_VERSION_CHECK=1 "$DP_VENV/bin/python" -m pip install --no-cache-dir -r "$DP_ROOT/submodules/ComfyUI/requirements.txt"
    "$DP_VENV/bin/python" -m pip check || warn "pip reported a dependency conflict; the required import/CUDA check will decide whether setup can continue"
    environment_is_healthy || die "The Python environment still fails its import/CUDA check; rerun with --rebuild-venv"
    printf '%s\n' "$desired_signature" > "$signature_file"
else
    log "Persistent Python environment is current; skipping pip installation"
    "$DP_VENV/bin/python" -m pip check || warn "pip reported a dependency conflict, but the cached environment passed the required import/CUDA check"
fi

if [[ -z "$COMFYUI_PYTHON" ]]; then
    COMFYUI_PYTHON="$DP_VENV/bin/python"
    warn "No dedicated ComfyUI venv was detected; COMFYUI_PYTHON will use the diffusion-pipe venv"
fi

H3_DATA_ROOT="$WORKSPACE_ROOT/data/minimax-h3"
H3_CONFIG_ROOT="$WORKSPACE_ROOT/configs/minimax-h3"
H3_OUTPUT_ROOT="$WORKSPACE_ROOT/output/minimax-h3-nsync-self-flow"
H3_WORKFLOW_API="$WORKSPACE_ROOT/workflows/minimax_h3_t2va_api.json"
mkdir -p \
    "$H3_DATA_ROOT/positive" \
    "$H3_DATA_ROOT/negative" \
    "$H3_CONFIG_ROOT" \
    "$H3_OUTPUT_ROOT" \
    "$WORKSPACE_ROOT/workflows" \
    "$WORKSPACE_ROOT/logs" \
    "$WORKSPACE_ROOT/comfy-output"

comfy_model_name() {
    local category_root="$1"
    local resolved_model="$2"
    local candidate
    while IFS= read -r candidate; do
        if [[ "$(readlink -f -- "$candidate")" == "$resolved_model" ]]; then
            printf '%s\n' "${candidate#"$category_root"/}"
            return 0
        fi
    done < <(find -H "$category_root" -maxdepth 3 \( -type f -o -type l \) -name '*.safetensors' -print 2>/dev/null)
    return 1
}

H3_COMFY_DIFFUSION_NAME="$(comfy_model_name "$COMFYUI_ROOT/models/diffusion_models" "$H3_DIFFUSION_MODEL" || basename -- "$H3_DIFFUSION_MODEL")"
H3_COMFY_TEXT_ENCODER_NAME="$(comfy_model_name "$COMFYUI_ROOT/models/text_encoders" "$H3_TEXT_ENCODER" || basename -- "$H3_TEXT_ENCODER")"
H3_COMFY_VIDEO_VAE_NAME="$(comfy_model_name "$COMFYUI_ROOT/models/vae" "$H3_VIDEO_VAE" || basename -- "$H3_VIDEO_VAE")"
H3_COMFY_AUDIO_VAE_NAME="$(comfy_model_name "$COMFYUI_ROOT/models/vae" "$H3_AUDIO_VAE" || basename -- "$H3_AUDIO_VAE")"
H3_COMFY_TURBO_LORA_NAME="$(comfy_model_name "$COMFYUI_ROOT/models/loras" "$H3_TURBO_LORA" || basename -- "$H3_TURBO_LORA")"

BUNDLED_H3_WORKFLOW="$DP_ROOT/examples/minimax_h3_t2va_api.json"
[[ -f "$BUNDLED_H3_WORKFLOW" ]] || die "Bundled MiniMax H3 API workflow is missing: $BUNDLED_H3_WORKFLOW"
workflow_tmp="$(mktemp "$WORKSPACE_ROOT/workflows/.minimax_h3_t2va_api.json.XXXXXX")"
python - \
    "$BUNDLED_H3_WORKFLOW" \
    "$workflow_tmp" \
    "$H3_COMFY_DIFFUSION_NAME" \
    "$H3_COMFY_TEXT_ENCODER_NAME" \
    "$H3_COMFY_VIDEO_VAE_NAME" \
    "$H3_COMFY_AUDIO_VAE_NAME" \
    "$H3_COMFY_TURBO_LORA_NAME" <<'PY'
import json
import sys

source, destination, diffusion, text_encoder, video_vae, audio_vae, turbo_lora = sys.argv[1:]
with open(source, encoding="utf-8") as file:
    workflow = json.load(file)
workflow["1"]["inputs"]["unet_name"] = diffusion
workflow["2"]["inputs"]["clip_name"] = text_encoder
workflow["3"]["inputs"]["vae_name"] = video_vae
workflow["4"]["inputs"]["vae_name"] = audio_vae
workflow["15"]["inputs"]["lora_name"] = turbo_lora
with open(destination, "w", encoding="utf-8") as file:
    json.dump(workflow, file, indent=2)
    file.write("\n")
PY
mv -f -- "$workflow_tmp" "$H3_WORKFLOW_API"
log "Ready-to-queue ComfyUI API workflow: $H3_WORKFLOW_API"

ENV_FILE="$WORKSPACE_ROOT/minimax-h3-env.sh"
ENV_FILE_TMP="$(mktemp "$WORKSPACE_ROOT/.minimax-h3-env.sh.XXXXXX")"
{
    printf '# Generated by %q. Re-run setup instead of editing this file.\n' "$0"
    printf 'export DP_ROOT=%q\n' "$DP_ROOT"
    printf 'export DP_VENV=%q\n' "$DP_VENV"
    printf 'export COMFYUI_ROOT=%q\n' "$COMFYUI_ROOT"
    printf 'export COMFYUI_PYTHON=%q\n' "$COMFYUI_PYTHON"
    printf 'export COMFYUI_URL=%q\n' 'http://127.0.0.1:8188'
    printf 'export H3_MODEL_ROOT=%q\n' "$COMFYUI_ROOT/models"
    printf 'export H3_DIFFUSION_MODEL=%q\n' "$H3_DIFFUSION_MODEL"
    printf 'export H3_TEXT_ENCODER=%q\n' "$H3_TEXT_ENCODER"
    printf 'export H3_VIDEO_VAE=%q\n' "$H3_VIDEO_VAE"
    printf 'export H3_AUDIO_VAE=%q\n' "$H3_AUDIO_VAE"
    printf 'export H3_TURBO_LORA=%q\n' "$H3_TURBO_LORA"
    printf 'export H3_DATA_ROOT=%q\n' "$H3_DATA_ROOT"
    printf 'export H3_CONFIG_ROOT=%q\n' "$H3_CONFIG_ROOT"
    printf 'export H3_OUTPUT_ROOT=%q\n' "$H3_OUTPUT_ROOT"
    printf 'export H3_WORKFLOW_API=%q\n' "$H3_WORKFLOW_API"
    printf 'if [[ ${VIRTUAL_ENV:-} != %q ]]; then source %q; fi\n' "$DP_VENV" "$DP_VENV/bin/activate"
} > "$ENV_FILE_TMP"
chmod 0644 "$ENV_FILE_TMP"
mv -f -- "$ENV_FILE_TMP" "$ENV_FILE"

log "Setup complete"
printf '\nSource the generated environment in this shell:\n\n'
printf '  source %q\n\n' "$ENV_FILE"
printf 'On each new Pod attached to the same Network Volume, run:\n\n'
printf '  bash %q --comfy-root %q\n' "$DP_ROOT/tools/setup_runpod_minimax_h3.sh" "$COMFYUI_ROOT"
printf '  source %q\n\n' "$ENV_FILE"
printf 'The next run will reuse %s and skip pip when its signature is current.\n' "$DP_VENV"
