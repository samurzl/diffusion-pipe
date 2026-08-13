"""Lightweight configuration validation for the training entry point.

This module intentionally only imports the standard library (with ``toml`` as a
Python 3.10 fallback). It is used before importing torch, DeepSpeed, Hugging Face,
or ComfyUI so configuration errors cannot waste a model-loading/cache-building
cycle.
"""

from __future__ import annotations

import os
from pathlib import Path
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


def _positive_integral_number(value: Any) -> bool:
    return _positive_int(value) or (
        isinstance(value, float) and value > 0 and value.is_integer()
    )


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


def _validate_dtype(container: dict[str, Any], key: str, prefix: str, errors: list[str]) -> None:
    if key not in container:
        return
    value = container[key]
    if value not in DTYPE_NAMES:
        errors.append(f'{prefix}{key} must be one of {sorted(DTYPE_NAMES)}, got {value!r}')


def _validate_explicit_local_path(value: Any, key: str, errors: list[str]) -> None:
    """Validate paths that are unambiguously local without rejecting Hub model IDs."""
    if not isinstance(value, str) or not value:
        errors.append(f'{key} must be a non-empty string')
        return
    expanded = Path(value).expanduser()
    explicitly_local = expanded.is_absolute() or value.startswith(('./', '../', '~/'))
    if explicitly_local and not expanded.exists():
        errors.append(f'{key} points to a local path that does not exist: {value}')


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
    output_dir = config.get('output_dir')
    if not isinstance(output_dir, str) or not output_dir:
        errors.append('output_dir must be a non-empty string')

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
    partition_split = config.get('partition_split')
    if partition_method == 'manual':
        if not isinstance(partition_split, list):
            errors.append('partition_split must be a list when partition_method="manual"')
        elif _positive_int(pipeline_stages) and len(partition_split) != pipeline_stages - 1:
            errors.append(f'partition_split must contain pipeline_stages - 1 ({pipeline_stages - 1}) entries')
        elif not all(_positive_int(value) for value in partition_split):
            errors.append('every partition_split entry must be a positive integer layer index')

    activation_checkpointing = config.get('activation_checkpointing', False)
    if activation_checkpointing not in (False, True, 'unsloth'):
        errors.append('activation_checkpointing must be true, false, or "unsloth"')

    scheduler = config.get('lr_scheduler', 'constant')
    if scheduler not in ('constant', 'linear', 'cosine'):
        errors.append(f'lr_scheduler must be constant, linear, or cosine; got {scheduler!r}')

    for key in ('compile', 'trust_cache', 'regenerate_cache', 'eval_before_first_step'):
        if key in config and not isinstance(config[key], bool):
            errors.append(f'{key} must be true or false')

    model_config = config.get('model')
    if not isinstance(model_config, dict):
        errors.append('model must be a TOML table')
        model_config = {}
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
    _validate_dtype(model_config, 'dtype', 'model.', errors)
    _validate_dtype(model_config, 'transformer_dtype', 'model.', errors)
    _validate_dtype(model_config, 'diffusion_model_dtype', 'model.', errors)
    _validate_model_paths(model_config, errors)
    if model_type == 'auraflow' and not _positive_int(model_config.get('max_sequence_length')):
        errors.append('model.max_sequence_length must be a positive integer')
    if model_type == 'minimax_h3':
        mode = model_config.get('mode', 't2v')
        if not isinstance(mode, str) or mode.lower() not in ('t2v', 'i2v'):
            errors.append('model.mode must be t2v or i2v for MiniMax H3')
        visual_timestep = model_config.get('i2v_visual_cond_timestep', 0.999)
        if not isinstance(visual_timestep, (int, float)) or isinstance(visual_timestep, bool) or not 0 <= visual_timestep <= 1:
            errors.append('model.i2v_visual_cond_timestep must be in [0, 1]')

    adapter_config = config.get('adapter')
    if adapter_config is not None:
        if not isinstance(adapter_config, dict):
            errors.append('adapter must be a TOML table')
        else:
            adapter_type = adapter_config.get('type')
            if adapter_type not in ('lora', 'lokr'):
                errors.append(f'adapter.type must be lora or lokr, got {adapter_type!r}')
            if not _positive_integral_number(adapter_config.get('rank')):
                errors.append('adapter.rank must be a positive integer value')
            if 'alpha' in adapter_config:
                errors.append('adapter.alpha is not supported; it is always set equal to adapter.rank')
            _validate_dtype(adapter_config, 'dtype', 'adapter.', errors)
            if 'init_from_existing' in adapter_config:
                _validate_explicit_local_path(adapter_config['init_from_existing'], 'adapter.init_from_existing', errors)

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
        if not isinstance(betas, list) or len(betas) != 2 or not all(isinstance(x, (int, float)) and 0 <= x < 1 for x in betas):
            errors.append('optimizer.betas must contain two numbers in the range [0, 1)')

    blocks_to_swap = config.get('blocks_to_swap', 0)
    if not isinstance(blocks_to_swap, int) or isinstance(blocks_to_swap, bool) or blocks_to_swap < 0:
        errors.append('blocks_to_swap must be a non-negative integer')
    elif blocks_to_swap:
        if pipeline_stages != 1:
            errors.append('blocks_to_swap requires pipeline_stages=1')
        if adapter_config is None:
            errors.append('blocks_to_swap requires adapter training')

    training_methods = config.get('training_methods', {})
    if not isinstance(training_methods, dict):
        errors.append('training_methods must be a TOML table')
        training_methods = {}
    nsync = training_methods.get('nsync', {})
    self_flow = training_methods.get('self_flow', {})
    if not isinstance(nsync, dict):
        errors.append('training_methods.nsync must be a TOML table')
        nsync = {}
    if not isinstance(self_flow, dict):
        errors.append('training_methods.self_flow must be a TOML table')
        self_flow = {}
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

    monitoring = config.get('monitoring', {})
    if not isinstance(monitoring, dict):
        errors.append('monitoring must be a TOML table')
    elif monitoring.get('enable_wandb', False):
        for key in ('wandb_api_key', 'wandb_tracker_name', 'wandb_run_name'):
            if not isinstance(monitoring.get(key), str) or not monitoring[key]:
                errors.append(f'monitoring.{key} must be set when W&B is enabled')

    effective_resume = resume_from_checkpoint
    if effective_resume is None:
        effective_resume = config.get('resume_from_checkpoint', False)
    if effective_resume and isinstance(output_dir, str) and output_dir:
        output_path = Path(output_dir).expanduser()
        if effective_resume is True:
            if output_path.is_dir() and not any(path.is_dir() for path in output_path.iterdir()):
                errors.append(f'cannot resume: output_dir contains no run directories: {output_dir}')
            elif not output_path.is_dir():
                errors.append(f'cannot resume: output_dir does not exist: {output_dir}')
        elif isinstance(effective_resume, str):
            resume_path = output_path / effective_resume
            if not resume_path.is_dir():
                errors.append(f'resume checkpoint directory does not exist: {resume_path}')


def _validate_resolution(value: Any, key: str, errors: list[str]) -> None:
    if _positive_number(value):
        return
    if isinstance(value, list) and len(value) == 2 and all(_positive_number(x) for x in value):
        return
    errors.append(f'{key} must be a positive number or a [width, height] pair of positive numbers')


def _validate_dataset_config(
    dataset_config: dict[str, Any],
    description: str,
    *,
    skip_dataset_validation: bool,
    nsync_expected: bool,
    model_type: str | None,
    errors: list[str],
) -> None:
    directories = dataset_config.get('directory')
    if not isinstance(directories, list) or not directories:
        errors.append(f'{description}.directory must contain at least one [[directory]] table')
        return

    nsync_roles: list[str | None] = []
    nsync_pairs: dict[str, dict[str, int]] = {}
    for index, directory in enumerate(directories):
        prefix = f'{description}.directory[{index}]'
        if not isinstance(directory, dict):
            errors.append(f'{prefix} must be a table')
            continue

        media_path_value = directory.get('path')
        if not isinstance(media_path_value, str) or not media_path_value:
            errors.append(f'{prefix}.path must be a non-empty string')
        else:
            media_path = Path(media_path_value).expanduser()
            if not media_path.is_dir():
                errors.append(f'{prefix}.path is not a directory: {media_path_value}')
            else:
                ignored_suffixes = {'.txt', '.npz', '.json', '.parquet', '.bak', '.db'}
                try:
                    contains_media = any(
                        path.is_file() and path.suffix not in ignored_suffixes
                        for path in media_path.iterdir()
                    )
                except OSError as exc:
                    errors.append(f'{prefix}.path could not be read: {exc}')
                else:
                    if not contains_media:
                        errors.append(f'{prefix}.path contains no images, videos, or tar archives: {media_path_value}')

        for key in ('caption_path', 'mask_path', 'control_path'):
            if key not in directory:
                continue
            value = directory[key]
            if not isinstance(value, str) or not Path(value).expanduser().is_dir():
                errors.append(f'{prefix}.{key} is not a directory: {value!r}')
        if 'default_mask_file' in directory:
            value = directory['default_mask_file']
            if not isinstance(value, str) or not Path(value).expanduser().is_file():
                errors.append(f'{prefix}.default_mask_file is not a file: {value!r}')

        repeats = directory.get('num_repeats', dataset_config.get('num_repeats', 1))
        if not _positive_number(repeats):
            errors.append(f'{prefix}.num_repeats must be positive')

        size_buckets = directory.get('size_buckets', dataset_config.get('size_buckets'))
        resolutions = directory.get('resolutions', dataset_config.get('resolutions'))
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

        if directory.get('online_captions', dataset_config.get('online_captions', False)):
            caption_dir = directory.get('caption_path', media_path_value)
            if isinstance(caption_dir, str) and not (Path(caption_dir).expanduser() / 'captions.json').is_file():
                errors.append(f'{prefix} enables online_captions but captions.json was not found in {caption_dir}')

        role = directory.get('nsync_role')
        nsync_roles.append(role)
        if role is not None:
            if role not in ('positive', 'negative'):
                errors.append(f'{prefix}.nsync_role must be positive or negative, got {role!r}')
            pair = directory.get('nsync_pair', 'default')
            if not isinstance(pair, str) or not pair:
                errors.append(f'{prefix}.nsync_pair must be a non-empty string')
            else:
                counts = nsync_pairs.setdefault(pair, {'positive': 0, 'negative': 0})
                if role in counts:
                    counts[role] += 1

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
    world_size: int | None = None,
    resume_from_checkpoint: bool | str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the main/dataset TOML files and report all cheap validation errors."""
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
            dataset_entries.append((f'eval_datasets[{index}]', entry))
        elif isinstance(entry, dict):
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

    for description, dataset_path_value in dataset_entries:
        if not isinstance(dataset_path_value, str) or not dataset_path_value:
            errors.append(f'{description} must be a non-empty TOML file path')
            continue
        dataset_path = Path(dataset_path_value).expanduser()
        dataset_config = _load_toml(dataset_path, description, errors)
        if dataset_config is None:
            continue
        dataset_configs[str(dataset_path)] = dataset_config
        _validate_dataset_config(
            dataset_config,
            description,
            skip_dataset_validation=skip_dataset_validation,
            nsync_expected=nsync_expected,
            model_type=model_type,
            errors=errors,
        )

    if errors:
        raise ConfigValidationError(errors)
    return config, dataset_configs
