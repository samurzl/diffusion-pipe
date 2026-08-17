"""Lightweight configuration validation for the training entry point.

This module intentionally only imports the standard library (with ``toml`` as a
Python 3.10 fallback). It is used before importing torch, DeepSpeed, Hugging Face,
or ComfyUI so configuration errors cannot waste a model-loading/cache-building
cycle.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import tarfile
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; the project already depends on toml.
    import toml as tomllib


SUPPORTED_MODEL_TYPES = {
    'anima',
    'auraflow',
    'chroma',
    'cosmos',
    'cosmos_predict2',
    'ernie_image',
    'flux',
    'flux2',
    'hidream',
    'hunyuan-video',
    'hunyuan_image',
    'hunyuan_video_15',
    'ideogram4',
    'krea2',
    'ltx-video',
    'ltx2',
    'lumina_2',
    'minimax_h3',
    'omnigen2',
    'qwen_image',
    'sd3',
    'sdxl',
    'wan',
    'z_image',
}

DTYPE_NAMES = {
    'float32',
    'float16',
    'bfloat16',
    'float8',
    'float8_e4m3fn',
    'float8_e5m2',
}

TRAINING_DTYPE_NAMES = {'bfloat16', 'float16', 'float32'}

TOP_LEVEL_KEYS = {
    'activation_checkpointing',
    'adapter',
    'blocks_to_swap',
    'caching_batch_size',
    'checkpoint_every_n_epochs',
    'checkpoint_every_n_minutes',
    'compile',
    'dataset',
    'disable_block_swap_for_eval',
    'epochs',
    'eval_before_first_step',
    'eval_datasets',
    'eval_every_n_epochs',
    'eval_every_n_examples',
    'eval_every_n_steps',
    'eval_gradient_accumulation_steps',
    'eval_image_micro_batch_size_per_gpu',
    'eval_micro_batch_size_per_gpu',
    'force_constant_lr',
    'gradient_accumulation_steps',
    'gradient_clipping',
    'huber_delta',
    'image_micro_batch_size_per_gpu',
    'logging_steps',
    'lr_scheduler',
    'map_num_proc',
    'max_steps',
    'micro_batch_size_per_gpu',
    'model',
    'monitoring',
    'optimizer',
    'output_dir',
    'partition_method',
    'partition_split',
    'pipeline_stages',
    'reentrant_activation_checkpointing',
    'regenerate_cache',
    'resume_from_checkpoint',
    'save_dtype',
    'save_every_n_epochs',
    'save_every_n_examples',
    'save_every_n_steps',
    'smooth_l1_beta',
    'steps_per_print',
    'training_methods',
    'trust_cache',
    'uncond_fraction',
    'video_clip_mode',
    'warmup_steps',
    'x_axis_examples',
}

MODEL_KEYS = {
    'audio_vae',
    'bypass_guidance_embedding',
    'byt5_path',
    'cache_text_embeddings',
    'cfg',
    'checkpoint_path',
    'ckpt_path',
    'clip_path',
    'cross_attn_lr',
    'debiased_estimation_loss',
    'diffusers_path',
    'diffusion_model',
    'diffusion_model_dtype',
    'dtype',
    'first_frame_conditioning_p',
    'flux_shift',
    'guidance',
    'i2v_visual_cond_timestep',
    'image_shift',
    'llama3_4bit',
    'llama3_path',
    'llm_adapter_lr',
    'llm_adapter_path',
    'llm_path',
    'lumina_shift',
    'max_llama3_sequence_length',
    'max_sequence_length',
    'max_t',
    'merge_adapters',
    'min_snr_gamma',
    'min_t',
    'mlp_lr',
    'mod_lr',
    'mode',
    'multiscale_loss_weight',
    'self_attn_lr',
    'shift',
    'sigmoid_scale',
    'single_file_path',
    't5_path',
    'text_encoder',
    'text_encoder_1_lr',
    'text_encoder_2_lr',
    'text_encoder_fp8',
    'text_encoder_nf4',
    'text_encoder_path',
    'text_encoders',
    'timestep_sample_method',
    'transformer_dtype',
    'transformer_path',
    'type',
    'unet_lr',
    'v_pred',
    'vae',
    'vae_path',
}

ADAPTER_KEYS = {
    'alpha',
    'decompose_factor',
    'dropout',
    'dtype',
    'exclude_modules',
    'init_from_existing',
    'rank',
    'rank_dropout',
    'type',
}

DATASET_KEYS = {
    'ar_buckets',
    'cache_shuffle_delimiter',
    'cache_shuffle_num',
    'caption_path',
    'caption_prefix',
    'control_path',
    'default_mask_file',
    'directory',
    'enable_ar_bucket',
    'frame_buckets',
    'mask_path',
    'max_ar',
    'min_ar',
    'nsync_anchor_pairs',
    'nsync_pair',
    'nsync_role',
    'num_ar_buckets',
    'num_repeats',
    'online_captions',
    'path',
    'resolutions',
    'shuffle_metadata',
    'shuffle_tags',
    'size_buckets',
    'skip_empty_caption',
    'subsample_ratio',
    'unbucketed',
}

DATASET_ROOT_KEYS = DATASET_KEYS - {
    'caption_path',
    'control_path',
    'default_mask_file',
    'mask_path',
    'nsync_anchor_pairs',
    'nsync_pair',
    'nsync_role',
    'path',
}

DATASET_DIRECTORY_KEYS = DATASET_KEYS - {'directory', 'subsample_ratio', 'unbucketed'}

BUCKET_DATASET_KEYS = {
    'ar_buckets',
    'enable_ar_bucket',
    'frame_buckets',
    'max_ar',
    'min_ar',
    'num_ar_buckets',
    'size_buckets',
}

BLOCK_SWAP_MODEL_TYPES = {
    'anima',
    'auraflow',
    'chroma',
    'cosmos_predict2',
    'ernie_image',
    'flux',
    'flux2',
    'hidream',
    'hunyuan-video',
    'hunyuan_image',
    'hunyuan_video_15',
    'ideogram4',
    'krea2',
    'ltx2',
    'minimax_h3',
    'qwen_image',
    'wan',
    'z_image',
}

BLOCK_SWAP_LIMITS = {
    'ltx2': 46,
    'minimax_h3': 48,
}

MODEL_SUBMODULES = {
    'chroma': ('flow', 'src'),
    'cosmos': ('Cosmos', 'cosmos1'),
    'hidream': ('HiDream', 'hi_diffusers'),
    'hunyuan-video': ('HunyuanVideo', 'hyvideo'),
    'hunyuan_image': ('HunyuanImage-2.1', 'hyimage'),
    'ltx-video': ('LTX_Video', 'ltx_video'),
    'lumina_2': ('Lumina_2', 'models'),
    'omnigen2': ('OmniGen2', 'omnigen2'),
}

IGNORED_DATASET_SUFFIXES = {'.txt', '.npz', '.json', '.parquet', '.bak', '.db'}

LOCAL_MODEL_FILE_SUFFIXES = {
    '.bin',
    '.ckpt',
    '.gguf',
    '.pt',
    '.pth',
    '.safetensors',
}

MODEL_REQUIRED_KEYS = {
    'auraflow': ('vae_path', 'text_encoder_path', 'transformer_path', 'max_sequence_length'),
    'chroma': ('diffusers_path', 'transformer_path'),
    'cosmos': ('vae_path', 'text_encoder_path', 'transformer_path'),
    'ernie_image': ('diffusion_model', 'vae', 'text_encoders'),
    'flux': ('diffusers_path',),
    'flux2': ('diffusion_model', 'vae', 'text_encoders'),
    'hidream': ('diffusers_path', 'llama3_path'),
    'hunyuan_image': ('vae_path', 'text_encoder_path', 'byt5_path', 'transformer_path'),
    'hunyuan_video_15': ('diffusion_model', 'vae', 'text_encoders'),
    'ideogram4': ('diffusion_model', 'vae', 'text_encoders'),
    'krea2': ('diffusion_model', 'vae', 'text_encoders'),
    'ltx-video': ('diffusers_path', 'single_file_path'),
    'ltx2': ('diffusion_model', 'text_encoder'),
    'lumina_2': ('vae_path', 'llm_path', 'transformer_path'),
    'minimax_h3': ('diffusion_model', 'vae', 'audio_vae', 'text_encoders'),
    'omnigen2': ('diffusers_path',),
    'sd3': ('diffusers_path',),
    'sdxl': ('checkpoint_path',),
    'wan': ('ckpt_path',),
    'z_image': ('diffusion_model', 'vae', 'text_encoders'),
}

SAVE_INTERVAL_KEYS = (
    'save_every_n_epochs',
    'save_every_n_steps',
    'save_every_n_examples',
)

EVAL_INTERVAL_KEYS = (
    'eval_every_n_epochs',
    'eval_every_n_steps',
    'eval_every_n_examples',
)


class ConfigValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        noun = 'error' if len(errors) == 1 else 'errors'
        details = '\n'.join(f'  - {error}' for error in errors)
        super().__init__(f'Configuration validation found {len(errors)} {noun}:\n{details}')


def _validate_known_keys(
    value: dict[str, Any],
    allowed: set[str],
    description: str,
    errors: list[str],
) -> None:
    for key in sorted(value.keys() - allowed):
        errors.append(f'{description} contains unknown key {key!r} (possible typo)')


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def validate_preflight_dependencies(
    config: dict[str, Any],
    *,
    repository_root: str | os.PathLike[str],
    include_training_dependencies: bool,
    cache_only: bool = False,
) -> None:
    """Check dependency presence without importing heavyweight training packages."""
    errors: list[str] = []
    repository_root = Path(repository_root)

    # Every model imports models.base, which imports the pinned ComfyUI checkout.
    required_submodules = [('ComfyUI', 'comfy')]
    model_config = config.get('model', {})
    model_type = model_config.get('type') if isinstance(model_config, dict) else None
    if model_type in MODEL_SUBMODULES:
        required_submodules.append(MODEL_SUBMODULES[model_type])
    # apply_patches currently imports HunyuanVideo for every training process.
    if include_training_dependencies:
        required_submodules.append(('HunyuanVideo', 'hyvideo'))
    for directory, import_root in required_submodules:
        marker = repository_root / 'submodules' / directory / import_root
        if not marker.exists():
            errors.append(
                f'required git submodule is not initialized: submodules/{directory} '
                '(run "git submodule update --init --recursive")'
            )

    if not cache_only:
        optimizer_config = config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', '') if isinstance(optimizer_config, dict) else ''
        optimizer_module = {
            'adamw8bit': 'bitsandbytes',
            'adamw8bitkahan': 'bitsandbytes',
            'adamw_optimi': 'optimi',
            'stableadamw': 'optimi',
            'offload': 'torchao.prototype.low_bit_optim',
            'automagic': 'optimum.quanto',
            # GenericOptim is implemented locally and has no extra package.
            'genericoptim': None,
        }.get(str(optimizer_type).lower())
        known_without_extra_dependency = {'adamw', 'genericoptim', 'sgd', ''}
        if (
            optimizer_module is None
            and str(optimizer_type).lower() not in known_without_extra_dependency
        ):
            optimizer_module = 'pytorch_optimizer'
        if optimizer_module and not _module_available(optimizer_module):
            errors.append(
                f'optimizer {optimizer_type!r} requires Python module {optimizer_module!r}, '
                'but it is not installed'
            )

    monitoring = config.get('monitoring', {})
    if (
        isinstance(monitoring, dict)
        and monitoring.get('enable_wandb', False)
        and include_training_dependencies
        and not cache_only
        and not _module_available('wandb')
    ):
        errors.append('monitoring.enable_wandb=true requires Python module "wandb"')

    if include_training_dependencies:
        required_modules = [
            'accelerate',
            'av',
            'comfy_aimdo',
            'datasets',
            'deepspeed',
            'diffusers',
            'einops',
            'imageio',
            'multiprocess',
            'numpy',
            'peft',
            'PIL',
            'safetensors',
            'tensorboard',
            'torch',
            'torchaudio',
            'torchvision',
            'tqdm',
            'transformers',
        ]
        if model_type == 'hunyuan-video':
            required_modules.append('loguru')
        missing = [module for module in required_modules if not _module_available(module)]
        if missing:
            errors.append(f'missing required Python modules: {", ".join(missing)}')

    if errors:
        raise ConfigValidationError(errors)


def _validate_safetensors_header(path: Path, key: str, errors: list[str]) -> None:
    """Validate the tiny JSON header without mapping or loading tensor data."""
    try:
        file_size = path.stat().st_size
        with path.open('rb') as handle:
            length_bytes = handle.read(8)
            if len(length_bytes) != 8:
                raise ValueError('file is shorter than the 8-byte header length')
            header_length = struct.unpack('<Q', length_bytes)[0]
            if header_length == 0 or header_length > file_size - 8:
                raise ValueError(
                    f'header declares {header_length} bytes but file size is {file_size}'
                )
            header = json.loads(handle.read(header_length))
            if not isinstance(header, dict):
                raise ValueError('header is not a JSON object')
            tensor_entries = {
                name: metadata
                for name, metadata in header.items()
                if name != '__metadata__'
            }
            if not tensor_entries:
                raise ValueError('header contains no tensors')
            data_size = file_size - 8 - header_length
            for name, metadata in tensor_entries.items():
                if not isinstance(metadata, dict):
                    raise ValueError(f'tensor {name!r} metadata is not an object')
                dtype = metadata.get('dtype')
                shape = metadata.get('shape')
                offsets = metadata.get('data_offsets')
                if not isinstance(dtype, str) or not dtype:
                    raise ValueError(f'tensor {name!r} has no dtype')
                if (
                    not isinstance(shape, list)
                    or not all(
                        isinstance(dimension, int)
                        and not isinstance(dimension, bool)
                        and dimension >= 0
                        for dimension in shape
                    )
                ):
                    raise ValueError(f'tensor {name!r} has an invalid shape')
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or not all(
                        isinstance(offset, int) and not isinstance(offset, bool)
                        for offset in offsets
                    )
                    or not 0 <= offsets[0] <= offsets[1] <= data_size
                ):
                    raise ValueError(f'tensor {name!r} has invalid data offsets')
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f'{key} is not a readable safetensors file: {path} ({exc})')


def _validate_output_dir(value: Any, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        return
    output_path = Path(value).expanduser()
    if output_path.exists() and not output_path.is_dir():
        errors.append(f'output_dir exists but is not a directory: {output_path}')
        return
    candidate = output_path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.is_dir():
        errors.append(f'output_dir has no existing parent directory: {output_path}')
    elif not os.access(candidate, os.W_OK | os.X_OK):
        errors.append(f'output_dir cannot be created or written under: {candidate}')


def _validate_checkpoint_run(path: Path, description: str, errors: list[str]) -> None:
    latest = path / 'latest'
    if not latest.is_file():
        errors.append(f'{description} is not a DeepSpeed checkpoint run (missing {latest})')
        return
    try:
        tag = latest.read_text().strip()
    except OSError as exc:
        errors.append(f'could not read checkpoint marker {latest}: {exc}')
        return
    if not tag:
        errors.append(f'checkpoint marker is empty: {latest}')
        return
    tag_path = path / tag
    if not tag_path.is_dir():
        errors.append(f'checkpoint marker {latest} points to missing directory: {tag_path}')
    elif not any(tag_path.glob('*model_states.pt')):
        errors.append(f'checkpoint directory contains no DeepSpeed model state files: {tag_path}')


def _load_toml(path: Path, description: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f'{description} does not exist or is not a file: {path}')
        return None
    try:
        value = tomllib.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f'Could not read {description} {path}: {exc}')
        return None
    if not isinstance(value, dict):
        errors.append(f'{description} must contain a TOML table: {path}')
        return None
    return value


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_batch_size(value: Any, key: str, errors: list[str]) -> None:
    if _positive_int(value):
        return
    if not isinstance(value, list) or not value:
        errors.append(f'{key} must be a positive integer or a non-empty list of [resolution, batch_size] pairs')
        return
    seen_resolutions = set()
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            errors.append(f'{key}[{index}] must be [resolution, batch_size]')
            continue
        resolution, batch_size = pair
        if not _positive_number(resolution):
            errors.append(f'{key}[{index}][0] must be a positive resolution')
        if not _positive_int(batch_size):
            errors.append(f'{key}[{index}][1] must be a positive integer batch size')
        if resolution in seen_resolutions:
            errors.append(f'{key} contains duplicate resolution {resolution!r}')
        seen_resolutions.add(resolution)


def _batch_size_values(value: Any) -> list[Any]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, list):
        return [pair[1] for pair in value if isinstance(pair, list) and len(pair) == 2]
    return []


def _validate_dtype(
    container: dict[str, Any],
    key: str,
    prefix: str,
    errors: list[str],
    *,
    allowed: set[str] = DTYPE_NAMES,
) -> None:
    if key not in container:
        return
    value = container[key]
    if value not in allowed:
        errors.append(f'{prefix}{key} must be one of {sorted(allowed)}, got {value!r}')


def _validate_explicit_local_path(value: Any, key: str, errors: list[str]) -> None:
    """Validate paths that are unambiguously local without rejecting Hub model IDs."""
    if not isinstance(value, str) or not value:
        errors.append(f'{key} must be a non-empty string')
        return
    expanded = Path(value).expanduser()
    explicitly_local = (
        expanded.exists()
        or expanded.is_absolute()
        or value.startswith(('./', '../', '~/'))
        or expanded.suffix.lower() in LOCAL_MODEL_FILE_SUFFIXES
    )
    if not explicitly_local:
        return
    if not expanded.exists():
        errors.append(f'{key} points to a local path that does not exist: {value}')
        return
    required_access = os.R_OK | os.X_OK if expanded.is_dir() else os.R_OK
    if not os.access(expanded, required_access):
        errors.append(f'{key} points to a local path that is not readable: {value}')
        return
    if expanded.is_dir():
        try:
            next(expanded.iterdir())
        except StopIteration:
            errors.append(f'{key} points to an empty directory: {value}')
        except OSError as exc:
            errors.append(f'{key} directory could not be read: {value} ({exc})')
    elif expanded.is_file() and expanded.stat().st_size == 0:
        errors.append(f'{key} points to an empty file: {value}')
    elif expanded.is_file() and expanded.suffix.lower() == '.safetensors':
        _validate_safetensors_header(expanded, key, errors)


def _validate_model_paths(model_config: dict[str, Any], errors: list[str]) -> None:
    path_keys = {
        'audio_vae',
        'byt5_path',
        'checkpoint_path',
        'ckpt_path',
        'clip_path',
        'diffusers_path',
        'diffusion_model',
        'llama3_path',
        'llm_adapter_path',
        'llm_path',
        'single_file_path',
        't5_path',
        'text_encoder',
        'text_encoder_path',
        'transformer_path',
        'vae',
        'vae_path',
    }
    for key in path_keys & model_config.keys():
        _validate_explicit_local_path(model_config[key], f'model.{key}', errors)

    diffusers_path = model_config.get('diffusers_path')
    if isinstance(diffusers_path, str):
        expanded = Path(diffusers_path).expanduser()
        explicitly_local = (
            expanded.exists()
            or expanded.is_absolute()
            or diffusers_path.startswith(('./', '../', '~/'))
        )
        if explicitly_local and expanded.is_dir() and not (expanded / 'model_index.json').is_file():
            errors.append(f'model.diffusers_path is missing model_index.json: {diffusers_path}')

    text_encoders = model_config.get('text_encoders')
    if text_encoders is not None:
        if not isinstance(text_encoders, list) or not text_encoders:
            errors.append('model.text_encoders must be a non-empty list')
        else:
            for index, encoder in enumerate(text_encoders):
                prefix = f'model.text_encoders[{index}]'
                if not isinstance(encoder, dict):
                    errors.append(f'{prefix} must be a table')
                    continue
                _validate_known_keys(encoder, {'path', 'paths', 'type'}, prefix, errors)
                if 'type' not in encoder or not isinstance(encoder['type'], str):
                    errors.append(f'{prefix}.type must be a string')
                paths = encoder.get('paths', encoder.get('path'))
                if paths is None:
                    errors.append(f'{prefix} must define path or paths')
                    continue
                if isinstance(paths, str):
                    paths = [paths]
                if not isinstance(paths, list) or not paths:
                    errors.append(f'{prefix}.paths must be a non-empty list of paths')
                    continue
                for path_index, path in enumerate(paths):
                    _validate_explicit_local_path(path, f'{prefix}.paths[{path_index}]', errors)

    for key in ('merge_adapters',):
        values = model_config.get(key, [])
        if not isinstance(values, list):
            errors.append(f'model.{key} must be a list')
            continue
        for index, value in enumerate(values):
            _validate_explicit_local_path(value, f'model.{key}[{index}]', errors)


def _validate_main_config(
    config: dict[str, Any],
    *,
    world_size: int | None,
    resume_from_checkpoint: bool | str | None,
    errors: list[str],
) -> None:
    _validate_known_keys(config, TOP_LEVEL_KEYS, 'training config', errors)

    output_dir = config.get('output_dir')
    if not isinstance(output_dir, str) or not output_dir:
        errors.append('output_dir must be a non-empty string')
    _validate_output_dir(output_dir, errors)

    if not _positive_int(config.get('epochs')):
        errors.append('epochs must be a positive integer')

    if 'max_steps' in config and not _positive_int(config['max_steps']):
        errors.append('max_steps must be a positive integer')

    configured_save_intervals = [key for key in SAVE_INTERVAL_KEYS if key in config]
    if not configured_save_intervals:
        errors.append(f'one of {", ".join(SAVE_INTERVAL_KEYS)} must be configured')
    for key in configured_save_intervals:
        if not _positive_int(config[key]):
            errors.append(f'{key} must be a positive integer')

    for key in EVAL_INTERVAL_KEYS:
        if key in config and config[key] is not None and not _positive_int(config[key]):
            errors.append(f'{key} must be a positive integer or omitted')

    for key in (
        'gradient_accumulation_steps',
        'eval_gradient_accumulation_steps',
        'caching_batch_size',
        'logging_steps',
        'map_num_proc',
        'steps_per_print',
        'warmup_steps',
    ):
        if key not in config:
            continue
        value = config[key]
        if key == 'warmup_steps':
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else:
            valid = _positive_int(value)
        if not valid:
            errors.append(f'{key} must be {"a non-negative" if key == "warmup_steps" else "a positive"} integer')

    for key in ('checkpoint_every_n_epochs', 'checkpoint_every_n_minutes'):
        if key in config and not _positive_number(config[key]):
            errors.append(f'{key} must be a positive number')

    for key in ('gradient_clipping',):
        if key in config:
            value = config[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                errors.append(f'{key} must be a non-negative number')

    for key in ('force_constant_lr', 'huber_delta'):
        if key in config and not _positive_number(config[key]):
            errors.append(f'{key} must be a positive number')
    if 'smooth_l1_beta' in config:
        value = config['smooth_l1_beta']
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            errors.append('smooth_l1_beta must be a non-negative number')
    if 'huber_delta' in config and 'smooth_l1_beta' in config:
        errors.append('configure only one of huber_delta and smooth_l1_beta')

    uncond_fraction = config.get('uncond_fraction', 0.0)
    if (
        not isinstance(uncond_fraction, (int, float))
        or isinstance(uncond_fraction, bool)
        or not 0 <= uncond_fraction <= 1
    ):
        errors.append('uncond_fraction must be a number in [0, 1]')

    video_clip_mode = config.get('video_clip_mode', 'single_beginning')
    if video_clip_mode not in ('single_beginning', 'single_middle'):
        errors.append(
            'video_clip_mode must be "single_beginning" or "single_middle", '
            f'got {video_clip_mode!r}'
        )

    for key in (
        'micro_batch_size_per_gpu',
        'eval_micro_batch_size_per_gpu',
        'image_micro_batch_size_per_gpu',
        'eval_image_micro_batch_size_per_gpu',
    ):
        if key in config:
            _validate_batch_size(config[key], key, errors)

    pipeline_stages = config.get('pipeline_stages', 1)
    if not _positive_int(pipeline_stages):
        errors.append('pipeline_stages must be a positive integer')
    elif world_size is not None:
        if pipeline_stages > world_size:
            errors.append(f'pipeline_stages ({pipeline_stages}) cannot exceed world size ({world_size})')
        elif world_size % pipeline_stages != 0:
            errors.append(f'world size ({world_size}) must be divisible by pipeline_stages ({pipeline_stages})')

    partition_method = config.get('partition_method', 'parameters')
    if not isinstance(partition_method, str) or not partition_method:
        errors.append('partition_method must be a non-empty string')
    elif not (
        partition_method in ('manual', 'parameters', 'uniform')
        or (partition_method.startswith('type:') and len(partition_method) > len('type:'))
    ):
        errors.append(
            'partition_method must be manual, parameters, uniform, or type:<layer-name>'
        )
    partition_split = config.get('partition_split')
    if partition_method == 'manual':
        if not isinstance(partition_split, list):
            errors.append('partition_split must be a list when partition_method="manual"')
        elif _positive_int(pipeline_stages) and len(partition_split) != pipeline_stages - 1:
            errors.append(f'partition_split must contain pipeline_stages - 1 ({pipeline_stages - 1}) entries')
        elif not all(_positive_int(value) for value in partition_split):
            errors.append('every partition_split entry must be a positive integer layer index')
        elif partition_split != sorted(set(partition_split)):
            errors.append('partition_split entries must be unique and strictly increasing')
    elif partition_split is not None:
        errors.append('partition_split is only used when partition_method="manual"')

    activation_checkpointing = config.get('activation_checkpointing', False)
    if activation_checkpointing not in (False, True, 'unsloth'):
        errors.append('activation_checkpointing must be true, false, or "unsloth"')

    scheduler = config.get('lr_scheduler', 'constant')
    if scheduler not in ('constant', 'linear', 'cosine'):
        errors.append(f'lr_scheduler must be constant, linear, or cosine; got {scheduler!r}')

    for key in (
        'compile',
        'disable_block_swap_for_eval',
        'eval_before_first_step',
        'reentrant_activation_checkpointing',
        'regenerate_cache',
        'trust_cache',
        'x_axis_examples',
    ):
        if key in config and not isinstance(config[key], bool):
            errors.append(f'{key} must be true or false')

    _validate_dtype(config, 'save_dtype', '', errors)

    model_config = config.get('model')
    if not isinstance(model_config, dict):
        errors.append('model must be a TOML table')
        model_config = {}
    else:
        _validate_known_keys(model_config, MODEL_KEYS, 'model', errors)
    model_type = model_config.get('type')
    if model_type not in SUPPORTED_MODEL_TYPES:
        errors.append(f'model.type must be one of {sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}')
    for key in MODEL_REQUIRED_KEYS.get(model_type, ()):
        if key not in model_config:
            errors.append(f'model.{key} is required for model.type={model_type!r}')
    if model_type in ('cosmos_predict2', 'anima'):
        for key in ('vae_path', 'transformer_path'):
            if key not in model_config:
                errors.append(f'model.{key} is required for model.type={model_type!r}')
        if 't5_path' not in model_config and 'llm_path' not in model_config:
            errors.append(f'model.type={model_type!r} requires model.t5_path or model.llm_path')
    if model_type == 'hunyuan-video':
        comfy_paths = ('transformer_path', 'vae_path', 'llm_path', 'clip_path')
        if 'ckpt_path' not in model_config and not all(key in model_config for key in comfy_paths):
            errors.append(
                'model.type="hunyuan-video" requires ckpt_path or all of transformer_path, '
                'vae_path, llm_path, and clip_path'
            )
    if model_type == 'qwen_image':
        separate_paths = ('text_encoder_path', 'vae_path', 'transformer_path')
        if 'diffusers_path' not in model_config and not all(key in model_config for key in separate_paths):
            errors.append('model.type="qwen_image" requires diffusers_path or all of text_encoder_path, vae_path, and transformer_path')
    if 'dtype' not in model_config:
        errors.append('model.dtype is required')
    _validate_dtype(
        model_config,
        'dtype',
        'model.',
        errors,
        allowed=TRAINING_DTYPE_NAMES,
    )
    _validate_dtype(model_config, 'transformer_dtype', 'model.', errors)
    _validate_dtype(model_config, 'diffusion_model_dtype', 'model.', errors)
    _validate_model_paths(model_config, errors)
    timestep_sample_method = model_config.get('timestep_sample_method', 'logit_normal')
    if model_type != 'sdxl' and timestep_sample_method not in ('logit_normal', 'uniform'):
        errors.append(
            'model.timestep_sample_method must be "logit_normal" or "uniform", '
            f'got {timestep_sample_method!r}'
        )
    for key in ('guidance', 'shift', 'sigmoid_scale', 'image_shift'):
        if key in model_config and not _positive_number(model_config[key]):
            errors.append(f'model.{key} must be a positive number')
    for key in (
        'bypass_guidance_embedding',
        'cache_text_embeddings',
        'debiased_estimation_loss',
        'flux_shift',
        'llama3_4bit',
        'lumina_shift',
        'text_encoder_fp8',
        'text_encoder_nf4',
        'v_pred',
    ):
        if key in model_config and not isinstance(model_config[key], bool):
            errors.append(f'model.{key} must be true or false')
    for key in ('max_llama3_sequence_length', 'max_sequence_length'):
        if key in model_config and not _positive_int(model_config[key]):
            errors.append(f'model.{key} must be a positive integer')
    for key in (
        'cross_attn_lr',
        'llm_adapter_lr',
        'mlp_lr',
        'mod_lr',
        'self_attn_lr',
        'text_encoder_1_lr',
        'text_encoder_2_lr',
        'unet_lr',
    ):
        if key in model_config:
            value = model_config[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                errors.append(f'model.{key} must be a non-negative number')
    if 'min_snr_gamma' in model_config and not _positive_number(model_config['min_snr_gamma']):
        errors.append('model.min_snr_gamma must be a positive number')
    if 'multiscale_loss_weight' in model_config:
        value = model_config['multiscale_loss_weight']
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            errors.append('model.multiscale_loss_weight must be a non-negative number')
    if 'first_frame_conditioning_p' in model_config:
        value = model_config['first_frame_conditioning_p']
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            errors.append('model.first_frame_conditioning_p must be a number in [0, 1]')
    min_t = model_config.get('min_t', 0.0)
    max_t = model_config.get('max_t', 1.0)
    if model_type == 'wan' and (
        not isinstance(min_t, (int, float))
        or isinstance(min_t, bool)
        or not isinstance(max_t, (int, float))
        or isinstance(max_t, bool)
        or not 0 <= min_t < max_t <= 1
    ):
        errors.append('model.min_t and model.max_t must satisfy 0 <= min_t < max_t <= 1')
    if model_type == 'minimax_h3':
        mode = model_config.get('mode', 't2v')
        if not isinstance(mode, str) or mode.lower() not in ('t2v', 'i2v'):
            errors.append('model.mode must be t2v or i2v for MiniMax H3')
        visual_timestep = model_config.get('i2v_visual_cond_timestep', 0.999)
        if not isinstance(visual_timestep, (int, float)) or isinstance(visual_timestep, bool) or not 0 <= visual_timestep <= 1:
            errors.append('model.i2v_visual_cond_timestep must be in [0, 1]')
        cfg = model_config.get('cfg', 1.0)
        if not _positive_number(cfg):
            errors.append('model.cfg must be a positive number')
        elif cfg > 1 and pipeline_stages != 1:
            errors.append('MiniMax H3 model.cfg > 1 requires pipeline_stages=1')

    adapter_config = config.get('adapter')
    if adapter_config is not None:
        if not isinstance(adapter_config, dict):
            errors.append('adapter must be a TOML table')
        else:
            _validate_known_keys(adapter_config, ADAPTER_KEYS, 'adapter', errors)
            adapter_type = adapter_config.get('type')
            if adapter_type not in ('lora', 'lokr'):
                errors.append(f'adapter.type must be lora or lokr, got {adapter_type!r}')
            if not _positive_int(adapter_config.get('rank')):
                errors.append('adapter.rank must be a positive integer')
            if 'alpha' in adapter_config:
                errors.append('adapter.alpha is not supported; it is always set equal to adapter.rank')
            _validate_dtype(
                adapter_config,
                'dtype',
                'adapter.',
                errors,
                allowed=TRAINING_DTYPE_NAMES,
            )
            if 'exclude_modules' in adapter_config:
                exclude_modules = adapter_config['exclude_modules']
                valid_exclusions = (
                    isinstance(exclude_modules, str)
                    and bool(exclude_modules)
                ) or (
                    isinstance(exclude_modules, list)
                    and all(isinstance(value, str) for value in exclude_modules)
                )
                if not valid_exclusions:
                    errors.append('adapter.exclude_modules must be a string or a list of strings')
            if adapter_type == 'lora' and 'dropout' in adapter_config:
                dropout = adapter_config['dropout']
                if (
                    not isinstance(dropout, (int, float))
                    or isinstance(dropout, bool)
                    or not 0 <= dropout <= 1
                ):
                    errors.append('adapter.dropout must be a number in [0, 1]')
            if adapter_type == 'lokr':
                if 'decompose_factor' in adapter_config:
                    factor = adapter_config['decompose_factor']
                    if not isinstance(factor, int) or isinstance(factor, bool) or factor == 0 or factor < -1:
                        errors.append('adapter.decompose_factor must be -1 or a positive integer')
                if 'rank_dropout' in adapter_config:
                    rank_dropout = adapter_config['rank_dropout']
                    if (
                        not isinstance(rank_dropout, (int, float))
                        or isinstance(rank_dropout, bool)
                        or not 0 <= rank_dropout <= 1
                    ):
                        errors.append('adapter.rank_dropout must be a number in [0, 1]')
            if 'init_from_existing' in adapter_config:
                _validate_explicit_local_path(adapter_config['init_from_existing'], 'adapter.init_from_existing', errors)
                adapter_path = adapter_config['init_from_existing']
                if isinstance(adapter_path, str):
                    expanded = Path(adapter_path).expanduser()
                    explicitly_local = (
                        expanded.exists()
                        or expanded.is_absolute()
                        or adapter_path.startswith(('./', '../', '~/'))
                    )
                    if explicitly_local and expanded.exists():
                        if not expanded.is_dir():
                            errors.append('adapter.init_from_existing must be an adapter directory')
                        else:
                            weights = list(expanded.glob('*.safetensors'))
                            if len(weights) != 1:
                                errors.append(
                                    'adapter.init_from_existing must contain exactly one .safetensors file; '
                                    f'found {len(weights)} in {expanded}'
                                )
                            else:
                                _validate_safetensors_header(
                                    weights[0],
                                    'adapter.init_from_existing weights',
                                    errors,
                                )

    optimizer_config = config.get('optimizer')
    if not isinstance(optimizer_config, dict):
        errors.append('optimizer must be a TOML table')
        optimizer_config = {}
    if not isinstance(optimizer_config.get('type'), str) or not optimizer_config.get('type'):
        errors.append('optimizer.type must be a non-empty string')
    if 'gradient_release' in optimizer_config and not isinstance(optimizer_config['gradient_release'], bool):
        errors.append('optimizer.gradient_release must be true or false')
    if 'lr' in optimizer_config and not _positive_number(optimizer_config['lr']):
        errors.append('optimizer.lr must be a positive number')
    if 'betas' in optimizer_config:
        betas = optimizer_config['betas']
        beta2_upper_inclusive = str(optimizer_config.get('type', '')).lower() == 'genericoptim'
        valid_beta2 = (
            isinstance(betas, list)
            and len(betas) == 2
            and isinstance(betas[1], (int, float))
            and not isinstance(betas[1], bool)
        )
        if valid_beta2:
            valid_beta2 = (
                0 <= betas[1] <= 1
                if beta2_upper_inclusive
                else 0 <= betas[1] < 1
            )
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or not isinstance(betas[0], (int, float))
            or isinstance(betas[0], bool)
            or not 0 <= betas[0] < 1
            or not valid_beta2
        ):
            range_description = '[0, 1]' if beta2_upper_inclusive else '[0, 1)'
            errors.append(f'optimizer.betas must contain two numbers in the range {range_description}')
    if 'beta2_half_life' in optimizer_config:
        if not _positive_number(optimizer_config['beta2_half_life']):
            errors.append('optimizer.beta2_half_life must be a positive number')
        if not isinstance(optimizer_config.get('betas'), list) or len(optimizer_config['betas']) != 2:
            errors.append('optimizer.beta2_half_life requires optimizer.betas with two entries')

    blocks_to_swap = config.get('blocks_to_swap', 0)
    if not isinstance(blocks_to_swap, int) or isinstance(blocks_to_swap, bool) or blocks_to_swap < 0:
        errors.append('blocks_to_swap must be a non-negative integer')
    elif blocks_to_swap:
        if pipeline_stages != 1:
            errors.append('blocks_to_swap requires pipeline_stages=1')
        if not isinstance(adapter_config, dict) or adapter_config.get('type') != 'lora':
            errors.append('blocks_to_swap requires a LoRA adapter')
        if model_type not in BLOCK_SWAP_MODEL_TYPES:
            errors.append(f'blocks_to_swap is not implemented for model.type={model_type!r}')
        elif model_type in BLOCK_SWAP_LIMITS and blocks_to_swap > BLOCK_SWAP_LIMITS[model_type]:
            errors.append(
                f'blocks_to_swap cannot exceed {BLOCK_SWAP_LIMITS[model_type]} '
                f'for model.type={model_type!r}'
            )

    training_methods = config.get('training_methods', {})
    if not isinstance(training_methods, dict):
        errors.append('training_methods must be a TOML table')
        training_methods = {}
    else:
        _validate_known_keys(training_methods, {'nsync', 'self_flow'}, 'training_methods', errors)
    nsync = training_methods.get('nsync', {})
    self_flow = training_methods.get('self_flow', {})
    if not isinstance(nsync, dict):
        errors.append('training_methods.nsync must be a TOML table')
        nsync = {}
    else:
        _validate_known_keys(nsync, {'enabled', 'eps'}, 'training_methods.nsync', errors)
    if not isinstance(self_flow, dict):
        errors.append('training_methods.self_flow must be a TOML table')
        self_flow = {}
    else:
        _validate_known_keys(
            self_flow,
            {
                'audio_loss_weight',
                'audio_mask_ratio',
                'ema_decay',
                'ema_dtype',
                'enabled',
                'gamma',
                'high_noise_fraction',
                'high_noise_range',
                'image_mask_ratio',
                'projection_dim',
                'student_layer',
                'student_layer_ratio',
                'teacher_layer',
                'teacher_layer_ratio',
                'video_loss_weight',
                'video_mask_ratio',
            },
            'training_methods.self_flow',
            errors,
        )
    nsync_enabled = nsync.get('enabled', False)
    self_flow_enabled = self_flow.get('enabled', False)
    if not isinstance(nsync_enabled, bool):
        errors.append('training_methods.nsync.enabled must be true or false')
    if not isinstance(self_flow_enabled, bool):
        errors.append('training_methods.self_flow.enabled must be true or false')
    if nsync_enabled or self_flow_enabled:
        if model_type != 'minimax_h3':
            errors.append('NSYNC and Self-Flow require model.type="minimax_h3"')
        if not isinstance(adapter_config, dict) or adapter_config.get('type') != 'lora':
            errors.append('NSYNC and Self-Flow require a LoRA adapter')
    if nsync_enabled and isinstance(optimizer_config, dict) and optimizer_config.get('gradient_release', False):
        errors.append('NSYNC is incompatible with optimizer.gradient_release')
    if nsync_enabled and config.get('uncond_fraction', 0.0) != 0:
        errors.append('NSYNC requires uncond_fraction=0')
    if self_flow_enabled and pipeline_stages != 1:
        errors.append('Self-Flow requires pipeline_stages=1')
    if nsync_enabled and world_size is not None and pipeline_stages != world_size:
        errors.append('NSYNC requires pipeline_stages to equal the world size (data parallel world size must be 1)')
    if (
        isinstance(optimizer_config, dict)
        and optimizer_config.get('gradient_release', False)
        and world_size is not None
        and pipeline_stages != world_size
    ):
        errors.append(
            'optimizer.gradient_release requires pipeline_stages to equal the world size '
            '(data parallel world size must be 1)'
        )

    if nsync_enabled:
        eps = nsync.get('eps', 1e-8)
        if not _positive_number(eps):
            errors.append('training_methods.nsync.eps must be a positive number')

    if self_flow_enabled:
        gamma = self_flow.get('gamma', 0.8)
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma < 0:
            errors.append('training_methods.self_flow.gamma must be a non-negative number')
        ema_decay = self_flow.get('ema_decay', 0.9999)
        if (
            not isinstance(ema_decay, (int, float))
            or isinstance(ema_decay, bool)
            or not 0 <= ema_decay < 1
        ):
            errors.append('training_methods.self_flow.ema_decay must be a number in [0, 1)')
        if self_flow.get('ema_dtype', 'float32') not in ('float32', 'bfloat16'):
            errors.append('training_methods.self_flow.ema_dtype must be float32 or bfloat16')
        for key in ('image_mask_ratio', 'video_mask_ratio', 'audio_mask_ratio'):
            value = self_flow.get(key, {'image_mask_ratio': 0.25, 'video_mask_ratio': 0.1, 'audio_mask_ratio': 0.5}[key])
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= value <= 0.5
            ):
                errors.append(f'training_methods.self_flow.{key} must be a number in [0, 0.5]')
        high_noise_fraction = self_flow.get('high_noise_fraction', 0.0)
        if (
            not isinstance(high_noise_fraction, (int, float))
            or isinstance(high_noise_fraction, bool)
            or not 0 <= high_noise_fraction <= 1
        ):
            errors.append('training_methods.self_flow.high_noise_fraction must be a number in [0, 1]')
        high_noise_range = self_flow.get('high_noise_range', [0.95, 1.0])
        if (
            not isinstance(high_noise_range, list)
            or len(high_noise_range) != 2
            or not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in high_noise_range)
            or not 0 <= high_noise_range[0] < high_noise_range[1] <= 1
        ):
            errors.append(
                'training_methods.self_flow.high_noise_range must be [low, high] with '
                '0 <= low < high <= 1'
            )
        if not _positive_int(self_flow.get('projection_dim', 1024)):
            errors.append('training_methods.self_flow.projection_dim must be a positive integer')
        student_layer = self_flow.get('student_layer')
        teacher_layer = self_flow.get('teacher_layer')
        for key, value in (('student_layer', student_layer), ('teacher_layer', teacher_layer)):
            if value is not None and not _positive_int(value):
                errors.append(f'training_methods.self_flow.{key} must be a positive integer')
        if _positive_int(student_layer) and _positive_int(teacher_layer) and student_layer >= teacher_layer:
            errors.append('training_methods.self_flow.student_layer must be less than teacher_layer')
        student_ratio = self_flow.get('student_layer_ratio', 0.3)
        teacher_ratio = self_flow.get('teacher_layer_ratio', 0.7)
        for key, value in (('student_layer_ratio', student_ratio), ('teacher_layer_ratio', teacher_ratio)):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 < value <= 1
            ):
                errors.append(f'training_methods.self_flow.{key} must be a number in (0, 1]')
        if (
            student_layer is None
            and teacher_layer is None
            and isinstance(student_ratio, (int, float))
            and isinstance(teacher_ratio, (int, float))
            and student_ratio >= teacher_ratio
        ):
            errors.append('training_methods.self_flow.student_layer_ratio must be less than teacher_layer_ratio')
        for key in ('video_loss_weight', 'audio_loss_weight'):
            value = self_flow.get(key, 1.0)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                errors.append(f'training_methods.self_flow.{key} must be a non-negative number')

    monitoring = config.get('monitoring', {})
    if not isinstance(monitoring, dict):
        errors.append('monitoring must be a TOML table')
    else:
        _validate_known_keys(
            monitoring,
            {'enable_wandb', 'wandb_api_key', 'wandb_run_name', 'wandb_tracker_name'},
            'monitoring',
            errors,
        )
        if 'enable_wandb' in monitoring and not isinstance(monitoring['enable_wandb'], bool):
            errors.append('monitoring.enable_wandb must be true or false')
        if monitoring.get('enable_wandb', False):
            for key in ('wandb_api_key', 'wandb_tracker_name', 'wandb_run_name'):
                if not isinstance(monitoring.get(key), str) or not monitoring[key]:
                    errors.append(f'monitoring.{key} must be set when W&B is enabled')

    if world_size is not None and _positive_int(pipeline_stages) and world_size % pipeline_stages == 0:
        data_parallel_size = world_size // pipeline_stages
        gradient_accumulation = config.get('gradient_accumulation_steps', 1)
        micro_batch = config.get('micro_batch_size_per_gpu', 1)
        if _positive_int(gradient_accumulation):
            if _positive_int(micro_batch):
                default_micro_batch = micro_batch
            elif (
                isinstance(micro_batch, list)
                and micro_batch
                and isinstance(micro_batch[0], list)
                and len(micro_batch[0]) == 2
                and _positive_int(micro_batch[0][1])
            ):
                default_micro_batch = micro_batch[0][1]
            else:
                default_micro_batch = None
            if default_micro_batch is not None:
                global_batch_size = default_micro_batch * gradient_accumulation * data_parallel_size
                for key in ('save_every_n_examples', 'eval_every_n_examples'):
                    value = config.get(key)
                    if _positive_int(value) and value < global_batch_size:
                        errors.append(
                            f'{key} ({value}) is smaller than the global batch size ({global_batch_size}); '
                            'it would be converted to a zero step interval'
                        )

    effective_resume = resume_from_checkpoint
    if effective_resume is None:
        effective_resume = config.get('resume_from_checkpoint', False)
    if not isinstance(effective_resume, (bool, str)):
        errors.append('resume_from_checkpoint must be true, false, or a run directory name')
    if effective_resume and isinstance(output_dir, str) and output_dir:
        output_path = Path(output_dir).expanduser()
        if effective_resume is True:
            run_directories = (
                sorted(path for path in output_path.iterdir() if path.is_dir())
                if output_path.is_dir()
                else []
            )
            if output_path.is_dir() and not run_directories:
                errors.append(f'cannot resume: output_dir contains no run directories: {output_dir}')
            elif not output_path.is_dir():
                errors.append(f'cannot resume: output_dir does not exist: {output_dir}')
            else:
                _validate_checkpoint_run(run_directories[-1], 'most recent run directory', errors)
        elif isinstance(effective_resume, str):
            resume_path = output_path / effective_resume
            if not resume_path.is_dir():
                errors.append(f'resume checkpoint directory does not exist: {resume_path}')
            else:
                _validate_checkpoint_run(resume_path, 'resume checkpoint directory', errors)


def _validate_resolution(value: Any, key: str, errors: list[str]) -> None:
    if _positive_number(value):
        return
    if isinstance(value, list) and len(value) == 2 and all(_positive_number(x) for x in value):
        return
    errors.append(f'{key} must be a positive number or a [width, height] pair of positive numbers')


def _inspect_dataset_media(
    directory: Path,
    description: str,
    errors: list[str],
) -> list[str]:
    """Return the media names the real dataset enumerator will attempt to load."""
    media_names: list[str] = []
    try:
        files = sorted(directory.iterdir())
    except OSError as exc:
        errors.append(f'{description} could not be read: {exc}')
        return media_names

    for path in files:
        if not path.is_file() or path.suffix in IGNORED_DATASET_SUFFIXES:
            continue
        if path.suffix.lower() != '.tar':
            media_names.append(path.name)
            continue
        try:
            with tarfile.open(path) as archive:
                members = [
                    member.name
                    for member in archive.getmembers()
                    if member.isfile() and Path(member.name).suffix not in IGNORED_DATASET_SUFFIXES
                ]
        except (OSError, tarfile.TarError) as exc:
            errors.append(f'{description} contains unreadable tar archive {path}: {exc}')
            continue
        if not members:
            errors.append(f'{description} tar archive contains no media files: {path}')
        media_names.extend(members)
    return media_names


def _load_caption_map(path: Path, description: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f'{description} is not valid JSON: {path} ({exc})')
        return None
    if not isinstance(value, dict):
        errors.append(f'{description} must contain a JSON object: {path}')
        return None
    return value


def _effective_dataset_value(
    directory: dict[str, Any],
    dataset: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    return directory.get(key, dataset.get(key, default))


def _validate_dataset_config(
    dataset_config: dict[str, Any],
    description: str,
    *,
    inspect_media: bool,
    skip_dataset_validation: bool,
    nsync_expected: bool,
    model_type: str | None,
    physical_batch_sizes: dict[str, Any],
    errors: list[str],
) -> None:
    _validate_known_keys(dataset_config, DATASET_ROOT_KEYS, description, errors)
    directories = dataset_config.get('directory')
    if not isinstance(directories, list) or not directories:
        errors.append(f'{description}.directory must contain at least one [[directory]] table')
        return

    subsample_ratio = dataset_config.get('subsample_ratio')
    if subsample_ratio is not None and (
        not isinstance(subsample_ratio, (int, float))
        or isinstance(subsample_ratio, bool)
        or not 0 < subsample_ratio <= 1
    ):
        errors.append(f'{description}.subsample_ratio must be a number in (0, 1]')

    unbucketed = dataset_config.get('unbucketed', False)
    if not isinstance(unbucketed, bool):
        errors.append(f'{description}.unbucketed must be true or false')
        unbucketed = False
    if unbucketed:
        if model_type != 'minimax_h3':
            errors.append(f'{description}.unbucketed is currently supported only for MiniMax H3')
        if nsync_expected:
            errors.append(f'{description}.unbucketed is not compatible with NSYNC')
        for key, value in physical_batch_sizes.items():
            batch_sizes = _batch_size_values(value)
            if not batch_sizes or any(batch_size != 1 for batch_size in batch_sizes):
                errors.append(f'{description}.unbucketed requires {key}=1 for every configured resolution')

    for key in ('cache_shuffle_num',):
        if key in dataset_config:
            value = dataset_config[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f'{description}.{key} must be a non-negative integer')
    for key in ('cache_shuffle_delimiter', 'caption_prefix'):
        if key in dataset_config and not isinstance(dataset_config[key], str):
            errors.append(f'{description}.{key} must be a string')

    nsync_roles: list[str | None] = []
    nsync_pairs: dict[str, dict[str, int]] = {}
    nsync_anchor_pairs: dict[str, list[str]] = {}
    nsync_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for index, directory in enumerate(directories):
        prefix = f'{description}.directory[{index}]'
        if not isinstance(directory, dict):
            errors.append(f'{prefix} must be a table')
            continue
        _validate_known_keys(directory, DATASET_DIRECTORY_KEYS, prefix, errors)

        media_path_value = directory.get('path')
        media_names: list[str] = []
        if not isinstance(media_path_value, str) or not media_path_value:
            errors.append(f'{prefix}.path must be a non-empty string')
        else:
            media_path = Path(media_path_value).expanduser()
            if not media_path.is_dir():
                errors.append(f'{prefix}.path is not a directory: {media_path_value}')
            elif not os.access(media_path, os.R_OK | os.X_OK):
                errors.append(f'{prefix}.path is not readable: {media_path_value}')
            else:
                if inspect_media:
                    media_names = _inspect_dataset_media(media_path, f'{prefix}.path', errors)
                    if not media_names:
                        errors.append(f'{prefix}.path contains no images, videos, or tar archives: {media_path_value}')
                cache_path = media_path / 'cache'
                if cache_path.exists() and not cache_path.is_dir():
                    errors.append(f'{prefix}.path has a cache path that is not a directory: {cache_path}')
                elif not os.access(
                    cache_path if cache_path.exists() else media_path,
                    os.W_OK | os.X_OK,
                ):
                    errors.append(
                        f'{prefix}.path cannot create or update its dataset cache under: {media_path}'
                    )

        for key in ('caption_path', 'mask_path', 'control_path'):
            if key not in directory:
                continue
            value = directory[key]
            expanded = Path(value).expanduser() if isinstance(value, str) else None
            if expanded is None or not expanded.is_dir():
                errors.append(f'{prefix}.{key} is not a directory: {value!r}')
            elif not os.access(expanded, os.R_OK | os.X_OK):
                errors.append(f'{prefix}.{key} is not readable: {value!r}')
        if 'default_mask_file' in directory:
            value = directory['default_mask_file']
            expanded = Path(value).expanduser() if isinstance(value, str) else None
            if expanded is None or not expanded.is_file():
                errors.append(f'{prefix}.default_mask_file is not a file: {value!r}')
            elif not os.access(expanded, os.R_OK):
                errors.append(f'{prefix}.default_mask_file is not readable: {value!r}')

        for key in ('enable_ar_bucket', 'online_captions', 'shuffle_metadata', 'shuffle_tags', 'skip_empty_caption'):
            value = _effective_dataset_value(directory, dataset_config, key)
            if value is not None and not isinstance(value, bool):
                errors.append(f'{prefix}.{key} must be true or false')
        cache_shuffle_num = _effective_dataset_value(directory, dataset_config, 'cache_shuffle_num', 0)
        if not isinstance(cache_shuffle_num, int) or isinstance(cache_shuffle_num, bool) or cache_shuffle_num < 0:
            errors.append(f'{prefix}.cache_shuffle_num must be a non-negative integer')
        for key, default in (('cache_shuffle_delimiter', ', '), ('caption_prefix', '')):
            value = _effective_dataset_value(directory, dataset_config, key, default)
            if not isinstance(value, str):
                errors.append(f'{prefix}.{key} must be a string')

        repeats = directory.get('num_repeats', dataset_config.get('num_repeats', 1))
        if not _positive_number(repeats):
            errors.append(f'{prefix}.num_repeats must be positive')

        size_buckets = directory.get('size_buckets', dataset_config.get('size_buckets'))
        resolutions = directory.get('resolutions', dataset_config.get('resolutions'))
        if unbucketed:
            configured_bucket_keys = sorted(
                key for key in BUCKET_DATASET_KEYS if key in directory or key in dataset_config
            )
            if configured_bucket_keys:
                errors.append(
                    f'{prefix} must omit bucket settings when unbucketed=true: '
                    f'{", ".join(configured_bucket_keys)}'
                )
            if not isinstance(resolutions, list) or len(resolutions) != 1:
                errors.append(f'{prefix}.unbucketed requires exactly one target resolution')
            else:
                _validate_resolution(resolutions[0], f'{prefix}.resolutions[0]', errors)
        else:
            if size_buckets is not None:
                if not isinstance(size_buckets, list) or not size_buckets:
                    errors.append(f'{prefix}.size_buckets must be a non-empty list')
                else:
                    for bucket_index, bucket in enumerate(size_buckets):
                        if not isinstance(bucket, list) or len(bucket) != 3 or not all(_positive_int(x) for x in bucket):
                            errors.append(f'{prefix}.size_buckets[{bucket_index}] must be [width, height, frames] using positive integers')
            elif not isinstance(resolutions, list) or not resolutions:
                errors.append(f'{prefix} needs non-empty resolutions or size_buckets')
            else:
                if len(resolutions) > 3 and not skip_dataset_validation:
                    errors.append(
                        f'{prefix} has {len(resolutions)} resolutions; this duplicates the dataset at every resolution. '
                        'Use --i_know_what_i_am_doing if intentional'
                    )
                for resolution_index, resolution in enumerate(resolutions):
                    _validate_resolution(resolution, f'{prefix}.resolutions[{resolution_index}]', errors)

            enable_ar_bucket = directory.get('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
            ar_buckets = directory.get('ar_buckets', dataset_config.get('ar_buckets'))
            if enable_ar_bucket and size_buckets is None and ar_buckets is None:
                for key in ('min_ar', 'max_ar', 'num_ar_buckets'):
                    value = directory.get(key, dataset_config.get(key))
                    valid = _positive_int(value) if key == 'num_ar_buckets' else _positive_number(value)
                    if not valid:
                        errors.append(f'{prefix}.{key} must be positive when aspect-ratio bucketing is enabled')
                min_ar = directory.get('min_ar', dataset_config.get('min_ar'))
                max_ar = directory.get('max_ar', dataset_config.get('max_ar'))
                if _positive_number(min_ar) and _positive_number(max_ar) and min_ar > max_ar:
                    errors.append(f'{prefix}.min_ar cannot exceed max_ar')
            elif ar_buckets is not None:
                if not isinstance(ar_buckets, list) or not ar_buckets:
                    errors.append(f'{prefix}.ar_buckets must be a non-empty list')
                else:
                    for bucket_index, bucket in enumerate(ar_buckets):
                        _validate_resolution(bucket, f'{prefix}.ar_buckets[{bucket_index}]', errors)

            frame_buckets = directory.get('frame_buckets', dataset_config.get('frame_buckets', [1]))
            if not isinstance(frame_buckets, list) or not frame_buckets or not all(_positive_int(x) for x in frame_buckets):
                errors.append(f'{prefix}.frame_buckets must be a non-empty list of positive integers')

        caption_dir_value = directory.get('caption_path', media_path_value)
        caption_counts: dict[str, int] = {}
        if isinstance(caption_dir_value, str) and Path(caption_dir_value).expanduser().is_dir():
            caption_dir = Path(caption_dir_value).expanduser()
            caption_json_path = caption_dir / 'captions.json'
            online_captions = _effective_dataset_value(
                directory,
                dataset_config,
                'online_captions',
                False,
            )
            caption_map = (
                _load_caption_map(caption_json_path, f'{prefix}.captions.json', errors)
                if inspect_media
                else None
            )
            if online_captions and (
                not caption_json_path.is_file()
                or (inspect_media and caption_map is None)
            ):
                errors.append(f'{prefix} enables online_captions but captions.json was not found or valid in {caption_dir_value}')

            captioned_media = 0
            for media_name in media_names:
                lookup_name = media_name if '/' in media_name else Path(media_name).name
                captions = caption_map.get(lookup_name) if caption_map is not None else None
                if captions is not None:
                    if not isinstance(captions, list) or not captions or not all(isinstance(item, str) for item in captions):
                        errors.append(
                            f'{prefix}.captions.json entry {lookup_name!r} must be a non-empty list of strings'
                        )
                        continue
                    count = len(captions)
                else:
                    text_path = (caption_dir / Path(media_name).name).with_suffix('.txt')
                    count = 1 if text_path.is_file() else 0
                caption_counts[Path(media_name).stem] = count
                captioned_media += count > 0

            skip_empty_caption = _effective_dataset_value(directory, dataset_config, 'skip_empty_caption', True)
            if inspect_media and skip_empty_caption is True and media_names and captioned_media == 0:
                errors.append(
                    f'{prefix} would be empty because skip_empty_caption=true and none of its '
                    'media files has a caption'
                )

        control_path_value = directory.get('control_path')
        if (
            inspect_media
            and isinstance(control_path_value, str)
            and Path(control_path_value).expanduser().is_dir()
        ):
            try:
                control_stems = {
                    path.stem for path in Path(control_path_value).expanduser().iterdir() if path.is_file()
                }
            except OSError as exc:
                errors.append(f'{prefix}.control_path could not be read: {exc}')
                control_stems = set()
            missing_control = sorted(
                {Path(media_name).stem for media_name in media_names} - control_stems
            )
            if missing_control:
                preview = ', '.join(missing_control[:5])
                errors.append(
                    f'{prefix}.control_path is missing control files for {len(missing_control)} media '
                    f'items; first missing stems: {preview}'
                )

        role = directory.get('nsync_role')
        pair = directory.get('nsync_pair', 'default')
        nsync_roles.append(role)
        if role is not None:
            if role not in ('positive', 'negative'):
                errors.append(f'{prefix}.nsync_role must be positive or negative, got {role!r}')
            if not isinstance(pair, str) or not pair:
                errors.append(f'{prefix}.nsync_pair must be a non-empty string')
            else:
                counts = nsync_pairs.setdefault(pair, {'positive': 0, 'negative': 0})
                if role in counts:
                    counts[role] += 1
                    summary = {
                        'caption_counts': caption_counts,
                        'media_stems': [Path(name).stem for name in media_names],
                        'num_repeats': repeats,
                        'shape_config': tuple(
                            repr(_effective_dataset_value(directory, dataset_config, key))
                            for key in (
                                'ar_buckets',
                                'enable_ar_bucket',
                                'frame_buckets',
                                'max_ar',
                                'min_ar',
                                'num_ar_buckets',
                                'resolutions',
                                'size_buckets',
                            )
                        ),
                        'caption_config': tuple(
                            repr(_effective_dataset_value(directory, dataset_config, key, default))
                            for key, default in (
                                ('cache_shuffle_num', 0),
                                ('cache_shuffle_delimiter', ', '),
                                ('caption_prefix', ''),
                                ('shuffle_tags', False),
                            )
                        ),
                    }
                    nsync_summaries.setdefault(pair, {})[role] = summary

        anchor_pairs = directory.get('nsync_anchor_pairs')
        if anchor_pairs is not None:
            if role != 'positive':
                errors.append(f'{prefix}.nsync_anchor_pairs may only be set on a positive directory')
            if not isinstance(anchor_pairs, list) or not anchor_pairs:
                errors.append(f'{prefix}.nsync_anchor_pairs must be a non-empty list of group names')
            elif not all(isinstance(anchor_pair, str) and anchor_pair for anchor_pair in anchor_pairs):
                errors.append(f'{prefix}.nsync_anchor_pairs must contain only non-empty strings')
            elif len(set(anchor_pairs)) != len(anchor_pairs):
                errors.append(f'{prefix}.nsync_anchor_pairs must not contain duplicate group names')
            elif role == 'positive' and isinstance(pair, str) and pair:
                nsync_anchor_pairs[pair] = anchor_pairs

    has_nsync_roles = any(role is not None for role in nsync_roles)
    if has_nsync_roles and any(role is None for role in nsync_roles):
        errors.append(f'{description}: every directory must define nsync_role when any directory does')
    if has_nsync_roles and not nsync_expected:
        errors.append(f'{description} defines nsync_role but NSYNC is not enabled in the training config')
    if nsync_expected and not has_nsync_roles:
        errors.append(f'{description} has no paired nsync_role directories but NSYNC is enabled')
    for pair, counts in nsync_pairs.items():
        if counts != {'positive': 1, 'negative': 1}:
            errors.append(
                f'{description}: NSYNC pair {pair!r} needs exactly one positive and one negative directory; '
                f'found {counts["positive"]} positive and {counts["negative"]} negative'
            )
            continue
        positive = nsync_summaries[pair]['positive']
        negative = nsync_summaries[pair]['negative']
        if negative['num_repeats'] != 1:
            errors.append(
                f'{description}: NSYNC negative directory for pair {pair!r} must use num_repeats=1 '
                'to avoid duplicate pairing keys'
            )
        if positive['shape_config'] != negative['shape_config']:
            errors.append(f'{description}: NSYNC pair {pair!r} has mismatched bucket configuration')
        if positive['caption_config'] != negative['caption_config']:
            errors.append(f'{description}: NSYNC pair {pair!r} has mismatched caption shuffle configuration')
        positive_stems = set(positive['media_stems'])
        negative_stems = set(negative['media_stems'])
        missing_stems = sorted(positive_stems - negative_stems)
        if missing_stems:
            preview = ', '.join(missing_stems[:5])
            errors.append(
                f'{description}: NSYNC pair {pair!r} is missing {len(missing_stems)} negative media '
                f'files; first missing stems: {preview}'
            )
        seen_negative_stems: set[str] = set()
        duplicate_negative_stems: set[str] = set()
        for stem in negative['media_stems']:
            if stem in seen_negative_stems:
                duplicate_negative_stems.add(stem)
            seen_negative_stems.add(stem)
        duplicate_negative_stems = sorted(duplicate_negative_stems)
        if duplicate_negative_stems:
            errors.append(
                f'{description}: NSYNC negative pair {pair!r} contains duplicate media stems: '
                f'{duplicate_negative_stems[:5]}'
            )
        for stem in sorted(positive_stems & negative_stems):
            positive_count = positive['caption_counts'].get(stem, 0)
            negative_count = negative['caption_counts'].get(stem, 0)
            if positive_count != negative_count:
                errors.append(
                    f'{description}: NSYNC pair {pair!r} has different caption counts for {stem!r}: '
                    f'{positive_count} positive and {negative_count} negative'
                )

    for pair, anchor_pairs in nsync_anchor_pairs.items():
        target_summary = nsync_summaries.get(pair, {}).get('positive')
        for anchor_pair in anchor_pairs:
            if anchor_pair not in nsync_pairs:
                errors.append(
                    f'{description}: NSYNC pair {pair!r} references unknown anchor pair {anchor_pair!r}'
                )
                continue
            anchor_summary = nsync_summaries.get(anchor_pair, {}).get('positive')
            if target_summary is not None and anchor_summary is not None:
                if target_summary['shape_config'] != anchor_summary['shape_config']:
                    errors.append(
                        f'{description}: NSYNC pair {pair!r} and anchor pair {anchor_pair!r} '
                        'have mismatched bucket configuration'
                    )

    if model_type == 'cosmos' and not skip_dataset_validation:
        # Cosmos only supports its fixed, explicit size buckets. The model performs
        # the same check much later, after loading its VAE and text encoder.
        for index, directory in enumerate(directories):
            if not isinstance(directory, dict):
                continue
            if any(key in directory or key in dataset_config for key in ('min_ar', 'max_ar', 'num_ar_buckets', 'resolutions')):
                errors.append(
                    f'{description}.directory[{index}] must use Cosmos-supported size_buckets; '
                    'use --i_know_what_i_am_doing to bypass model-specific dataset validation'
                )


def load_and_validate_config(
    config_path: str | os.PathLike[str],
    *,
    skip_dataset_validation: bool = False,
    inspect_dataset_media: bool = True,
    world_size: int | None = None,
    resume_from_checkpoint: bool | str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the main/dataset TOML files and report validation errors by cost."""
    errors: list[str] = []
    path = Path(config_path).expanduser()
    config = _load_toml(path, 'training config', errors)
    if config is None:
        raise ConfigValidationError(errors)

    _validate_main_config(
        config,
        world_size=world_size,
        resume_from_checkpoint=resume_from_checkpoint,
        errors=errors,
    )
    # Do not touch dataset files when the main configuration is already known
    # to be invalid. Dataset trees can contain hundreds of thousands of files.
    if errors:
        raise ConfigValidationError(errors)

    dataset_configs: dict[str, dict[str, Any]] = {}
    dataset_path_value = config.get('dataset')
    dataset_entries: list[tuple[str, Any]] = [('dataset', dataset_path_value)]

    eval_datasets = config.get('eval_datasets', [])
    if not isinstance(eval_datasets, list):
        errors.append('eval_datasets must be a list')
        eval_datasets = []
    seen_eval_names = set()
    for index, entry in enumerate(eval_datasets):
        if isinstance(entry, str):
            generated_name = f'eval{index}'
            if generated_name in seen_eval_names:
                errors.append(f'eval_datasets contains duplicate name {generated_name!r}')
            else:
                seen_eval_names.add(generated_name)
            dataset_entries.append((f'eval_datasets[{index}]', entry))
        elif isinstance(entry, dict):
            _validate_known_keys(entry, {'config', 'name'}, f'eval_datasets[{index}]', errors)
            name = entry.get('name')
            eval_path = entry.get('config')
            if not isinstance(name, str) or not name:
                errors.append(f'eval_datasets[{index}].name must be a non-empty string')
            elif name in seen_eval_names:
                errors.append(f'eval_datasets contains duplicate name {name!r}')
            else:
                seen_eval_names.add(name)
            dataset_entries.append((f'eval_datasets[{index}].config', eval_path))
        else:
            errors.append(f'eval_datasets[{index}] must be a path string or a table with name and config')

    training_methods = config.get('training_methods', {})
    nsync_expected = (
        isinstance(training_methods, dict)
        and isinstance(training_methods.get('nsync', {}), dict)
        and training_methods.get('nsync', {}).get('enabled', False) is True
    )
    model_config = config.get('model', {})
    model_type = model_config.get('type') if isinstance(model_config, dict) else None

    train_micro_batch = config.get('micro_batch_size_per_gpu', 1)
    train_image_micro_batch = config.get('image_micro_batch_size_per_gpu', train_micro_batch)
    eval_micro_batch = config.get('eval_micro_batch_size_per_gpu', train_micro_batch)
    eval_image_micro_batch = config.get('eval_image_micro_batch_size_per_gpu', eval_micro_batch)

    loaded_dataset_entries: list[tuple[str, dict[str, Any]]] = []
    for description, dataset_path_value in dataset_entries:
        if not isinstance(dataset_path_value, str) or not dataset_path_value:
            errors.append(f'{description} must be a non-empty TOML file path')
            continue
        dataset_path = Path(dataset_path_value).expanduser()
        dataset_config = _load_toml(dataset_path, description, errors)
        if dataset_config is None:
            continue
        dataset_configs[str(dataset_path)] = dataset_config
        loaded_dataset_entries.append((description, dataset_config))

    # Parse every dataset TOML before validating any of them. A syntax error in
    # a later eval dataset should not be hidden behind a scan of the train set.
    if errors:
        raise ConfigValidationError(errors)

    for description, dataset_config in loaded_dataset_entries:
        if description == 'dataset':
            physical_batch_sizes = {
                'micro_batch_size_per_gpu': train_micro_batch,
                'image_micro_batch_size_per_gpu': train_image_micro_batch,
            }
        else:
            physical_batch_sizes = {
                'eval_micro_batch_size_per_gpu': eval_micro_batch,
                'eval_image_micro_batch_size_per_gpu': eval_image_micro_batch,
            }
        _validate_dataset_config(
            dataset_config,
            description,
            inspect_media=False,
            skip_dataset_validation=skip_dataset_validation,
            nsync_expected=nsync_expected,
            model_type=model_type,
            physical_batch_sizes=physical_batch_sizes,
            errors=errors,
        )

    if errors:
        raise ConfigValidationError(errors)
    if inspect_dataset_media:
        for description, dataset_config in loaded_dataset_entries:
            if description == 'dataset':
                physical_batch_sizes = {
                    'micro_batch_size_per_gpu': train_micro_batch,
                    'image_micro_batch_size_per_gpu': train_image_micro_batch,
                }
            else:
                physical_batch_sizes = {
                    'eval_micro_batch_size_per_gpu': eval_micro_batch,
                    'eval_image_micro_batch_size_per_gpu': eval_image_micro_batch,
                }
            _validate_dataset_config(
                dataset_config,
                description,
                inspect_media=True,
                skip_dataset_validation=skip_dataset_validation,
                nsync_expected=nsync_expected,
                model_type=model_type,
                physical_batch_sizes=physical_batch_sizes,
                errors=errors,
            )

    if errors:
        raise ConfigValidationError(errors)
    return config, dataset_configs
