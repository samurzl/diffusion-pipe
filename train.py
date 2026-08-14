import argparse
import os
import json
from pathlib import Path

from utils.config_validation import (
    ConfigValidationError,
    load_and_validate_config,
    validate_preflight_dependencies,
)


def validate_runtime_environment(args, torch_module):
    """Fail before dataset inspection if the launcher or CUDA topology is unusable."""
    errors = []
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    if not 0 <= rank < world_size:
        errors.append(f'RANK ({rank}) must be in [0, WORLD_SIZE={world_size})')

    environment_local_rank = int(os.getenv('LOCAL_RANK', '-1'))
    local_rank = args.local_rank if args.local_rank >= 0 else environment_local_rank
    if local_rank < 0 and world_size == 1:
        local_rank = 0
    if local_rank < 0:
        errors.append('local rank is missing; launch training with DeepSpeed or set LOCAL_RANK')

    if not torch_module.cuda.is_available():
        errors.append('CUDA is not available to PyTorch')
    else:
        device_count = torch_module.cuda.device_count()
        if local_rank >= device_count:
            errors.append(
                f'LOCAL_RANK ({local_rank}) has no CUDA device; PyTorch sees {device_count} device(s)'
            )
        if 'LOCAL_WORLD_SIZE' in os.environ:
            local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
            if local_world_size > device_count:
                errors.append(
                    f'LOCAL_WORLD_SIZE ({local_world_size}) exceeds the {device_count} CUDA device(s) '
                    'visible to PyTorch'
                )

    if errors:
        raise ConfigValidationError(errors)
    args.local_rank = local_rank


# Run a cheap, dependency-light preflight before importing torch, DeepSpeed,
# Hugging Face, or ComfyUI. Model and dataset initialization can take many
# minutes, so discovering a TOML typo after those imports is needlessly costly.
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--reset_dataloader', action='store_true', help='Start dataloader from scratch when resuming from checkpoint, i.e. only load the optimizer states.')
parser.add_argument('--reset_optimizer', action='store_true')
parser.add_argument('--reset_optimizer_params', action='store_true')
parser.add_argument('--regenerate_cache', action='store_true', default=None, help='Force regenerate cache.')
mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
mode_group.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
mode_group.add_argument('--test_sample', action='store_true', help='Generate and write an image to example.png and then quit.')
parser.add_argument('--trust_cache', action='store_true', help='Load from metadata cache files if they exist, without checking if any fingerprints have changed. Can make loading much faster for large datasets.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser.add_argument('--validate_only', action='store_true', help='Validate config, dependencies, and dataset paths without importing training dependencies, initializing CUDA, or loading models.')

# These are the arguments currently added by deepspeed.add_config_arguments().
# Defining them here lets argparse reject CLI typos before importing DeepSpeed.
parser.add_argument('--deepspeed', action='store_true')
parser.add_argument('--deepspeed_config', default=None)
parser.add_argument('--deepscale', action='store_true')
parser.add_argument('--deepscale_config', default=None)
args = parser.parse_args()

_preflight_config = None
_preflight_dataset_configs = {}
try:
    environment_errors = []
    world_size = None
    environment_values = {'WORLD_SIZE': 1, 'RANK': 0}
    for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'):
        if name not in os.environ:
            continue
        try:
            value = int(os.environ[name])
        except ValueError:
            environment_errors.append(f'environment variable {name} must be an integer, got {os.environ[name]!r}')
            continue
        if name in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE') and value <= 0:
            environment_errors.append(f'environment variable {name} must be positive, got {value}')
        elif name in ('RANK', 'LOCAL_RANK') and value < 0:
            environment_errors.append(f'environment variable {name} must be non-negative, got {value}')
        environment_values[name] = value
        if name == 'WORLD_SIZE':
            world_size = value
    if not environment_errors:
        if environment_values['RANK'] >= environment_values['WORLD_SIZE']:
            environment_errors.append(
                f'environment variable RANK ({environment_values["RANK"]}) must be smaller than '
                f'WORLD_SIZE ({environment_values["WORLD_SIZE"]})'
            )
        if (
            'LOCAL_RANK' in environment_values
            and 'LOCAL_WORLD_SIZE' in environment_values
            and environment_values['LOCAL_RANK'] >= environment_values['LOCAL_WORLD_SIZE']
        ):
            environment_errors.append(
                f'environment variable LOCAL_RANK ({environment_values["LOCAL_RANK"]}) must be '
                f'smaller than LOCAL_WORLD_SIZE ({environment_values["LOCAL_WORLD_SIZE"]})'
            )
    if not 1 <= args.master_port <= 65535:
        environment_errors.append(f'--master_port must be in [1, 65535], got {args.master_port}')
    if environment_errors:
        raise ConfigValidationError(environment_errors)

    validation_world_size = (
        world_size
        if world_size is not None
        else (None if args.validate_only else 1)
    )

    _preflight_config, _preflight_dataset_configs = load_and_validate_config(
        args.config,
        skip_dataset_validation=args.i_know_what_i_am_doing,
        inspect_dataset_media=False,
        world_size=validation_world_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    cli_errors = []
    if args.dump_dataset and _preflight_config.get('model', {}).get('type') != 'flux':
        cli_errors.append('--dump_dataset is currently implemented only for model.type="flux"')
    if args.dump_dataset:
        dump_path = args.dump_dataset.expanduser()
        if dump_path.exists() and not dump_path.is_dir():
            cli_errors.append(f'--dump_dataset exists but is not a directory: {dump_path}')
        else:
            dump_parent = dump_path
            while not dump_parent.exists() and dump_parent != dump_parent.parent:
                dump_parent = dump_parent.parent
            if not dump_parent.is_dir() or not os.access(dump_parent, os.W_OK | os.X_OK):
                cli_errors.append(f'--dump_dataset cannot be created or written under: {dump_parent}')
    if args.test_sample and not os.access(Path.cwd(), os.W_OK | os.X_OK):
        cli_errors.append(f'--test_sample cannot write example.png in the current directory: {Path.cwd()}')
    if args.deepspeed_config is not None or args.deepscale_config is not None:
        cli_errors.append(
            '--deepspeed_config/--deepscale_config cannot be combined with this script because '
            'train.py builds and passes its DeepSpeed configuration internally'
        )
    if (
        (args.reset_dataloader or args.reset_optimizer or args.reset_optimizer_params)
        and args.resume_from_checkpoint is None
        and not _preflight_config.get('resume_from_checkpoint', False)
    ):
        cli_errors.append(
            '--reset_dataloader, --reset_optimizer, and --reset_optimizer_params require resuming from a checkpoint'
        )
    if cli_errors:
        raise ConfigValidationError(cli_errors)
    validate_preflight_dependencies(
        _preflight_config,
        repository_root=Path(__file__).resolve().parent,
        include_training_dependencies=not args.validate_only,
        cache_only=args.cache_only,
    )
except ConfigValidationError as exc:
    raise SystemExit(f'\n{exc}') from None

if args.validate_only:
    try:
        _preflight_config, _preflight_dataset_configs = load_and_validate_config(
            args.config,
            skip_dataset_validation=args.i_know_what_i_am_doing,
            world_size=validation_world_size,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except ConfigValidationError as exc:
        raise SystemExit(f'\n{exc}') from None
    print(f'Configuration is valid: {args.config}')
    raise SystemExit(0)


import torch

try:
    validate_runtime_environment(args, torch)
    # File enumeration, caption/control matching, and tar inspection are the
    # final preflight stage because they can be slow for very large datasets.
    _preflight_config, _preflight_dataset_configs = load_and_validate_config(
        args.config,
        skip_dataset_validation=args.i_know_what_i_am_doing,
        world_size=validation_world_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
except ConfigValidationError as exc:
    raise SystemExit(f'\n{exc}') from None


from datetime import datetime, timezone
import shutil
import glob
import time
import random
import inspect
import importlib
from collections import defaultdict

import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule

# needed for broadcasting Queue in dataset.py
mp.current_process().authkey = b'afsaskgfdjh4'

wandb_enable = False

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.state = defaultdict(dict)
        self.param_groups = []

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config or 'save_every_n_examples' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP[transformer_dtype]
    if diffusion_model_dtype := model_config.get('diffusion_model_dtype', None):
        model_config['diffusion_model_dtype'] = DTYPE_MAP[diffusion_model_dtype]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if 'alpha' in adapter_config:
            raise NotImplementedError(
                'This script forces alpha=rank to make the saved adapter format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
            )
        adapter_config['alpha'] = adapter_config['rank']
        adapter_config.setdefault('dtype', model_dtype_str)
        adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]

        # per-adapter defaults
        if adapter_config['type'] == 'lora':
            adapter_config.setdefault('dropout', 0.0)
        elif adapter_config['type'] == 'lokr':
            adapter_config.setdefault('decompose_factor', -1)
            adapter_config.setdefault('rank_dropout', 0.0)
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_every_n_examples', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)
    config.setdefault('x_axis_examples', False)

    training_methods = config.setdefault('training_methods', {})
    nsync_config = training_methods.setdefault('nsync', {})
    self_flow_config = training_methods.setdefault('self_flow', {})
    nsync_config.setdefault('enabled', False)
    self_flow_config.setdefault('enabled', False)
    if nsync_config['enabled'] or self_flow_config['enabled']:
        if model_config['type'] != 'minimax_h3':
            raise ValueError('NSYNC and Self-Flow are currently implemented specifically for model.type=minimax_h3')
        if config.get('adapter', {}).get('type') != 'lora':
            raise ValueError('NSYNC and Self-Flow currently require LoRA adapter training')
        if config['compile']:
            print('Disabling torch.compile because NSYNC/Self-Flow use backward hooks and runtime LoRA adapter selection')
            config['compile'] = False
    if nsync_config['enabled'] and config['optimizer'].get('gradient_release', False):
        raise ValueError('NSYNC gradient surgery is incompatible with optimizer.gradient_release')
    if nsync_config['enabled'] and config.get('uncond_fraction', 0.0) != 0:
        raise ValueError('NSYNC requires uncond_fraction=0 so paired positive and negative prompts stay identical')
    if self_flow_config['enabled'] and config['pipeline_stages'] != 1:
        raise ValueError('Self-Flow currently requires pipeline_stages=1 because its EMA teacher branch is forward-only')


def validate_optimizer_availability(config):
    """Catch optimizer typos/missing optional packages before loading any model."""
    optim_type = config['optimizer']['type']
    optim_type_lower = optim_type.lower()
    module_and_attribute = {
        'adamw8bit': ('bitsandbytes', 'optim.AdamW8bit'),
        'adamw_optimi': ('optimi', 'AdamW'),
        'stableadamw': ('optimi', 'StableAdamW'),
        'offload': ('torchao.prototype.low_bit_optim', 'CPUOffloadOptimizer'),
        'automagic': ('optimizers.automagic', 'Automagic'),
        'genericoptim': ('optimizers.generic_optim', 'GenericOptim'),
        'adamw8bitkahan': ('optimizers.adamw_8bit', 'AdamW8bitKahan'),
    }
    if optim_type_lower == 'adamw':
        value = torch.optim.AdamW
    elif optim_type_lower == 'sgd':
        value = torch.optim.SGD
    elif optim_type_lower in module_and_attribute:
        module_name, attribute_path = module_and_attribute[optim_type_lower]
        try:
            value = importlib.import_module(module_name)
            for attribute in attribute_path.split('.'):
                value = getattr(value, attribute)
        except (ImportError, AttributeError) as exc:
            raise ConfigValidationError([
                f'optimizer {optim_type!r} is unavailable; check optimizer.type and install its '
                f'optional dependency ({exc})'
            ]) from None
    else:
        module_name, attribute_path = 'pytorch_optimizer', optim_type
        try:
            value = importlib.import_module(module_name)
            for attribute in attribute_path.split('.'):
                value = getattr(value, attribute)
        except (ImportError, AttributeError) as exc:
            raise ConfigValidationError([
                f'optimizer {optim_type!r} is unavailable; check optimizer.type and install its '
                f'optional dependency ({exc})'
            ]) from None

    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        ignored_keys = {'beta2_half_life', 'gradient_release', 'type'}
        configured_keys = set(config['optimizer']) - ignored_keys
        accepted_keys = set(signature.parameters) - {'params'}
        unknown_keys = sorted(configured_keys - accepted_keys)
        if unknown_keys:
            raise ConfigValidationError([
                f'optimizer {optim_type!r} does not accept configured parameter(s): '
                f'{", ".join(unknown_keys)}'
            ])


MODEL_PIPELINES = {
    'anima': ('models.cosmos_predict2', 'CosmosPredict2Pipeline'),
    'auraflow': ('models.auraflow', 'AuraFlowPipeline'),
    'chroma': ('models.chroma', 'ChromaPipeline'),
    'cosmos': ('models.cosmos', 'CosmosPipeline'),
    'cosmos_predict2': ('models.cosmos_predict2', 'CosmosPredict2Pipeline'),
    'ernie_image': ('models.ernie_image', 'ErnieImagePipeline'),
    'flux': ('models.flux', 'FluxPipeline'),
    'flux2': ('models.flux2', 'Flux2Pipeline'),
    'hidream': ('models.hidream', 'HiDreamPipeline'),
    'hunyuan-video': ('models.hunyuan_video', 'HunyuanVideoPipeline'),
    'hunyuan_image': ('models.hunyuan_image', 'HunyuanImagePipeline'),
    'hunyuan_video_15': ('models.hunyuan_video_15', 'HunyuanVideo15Pipeline'),
    'ideogram4': ('models.ideogram4', 'Ideogram4Pipeline'),
    'krea2': ('models.krea2', 'Krea2Pipeline'),
    'ltx-video': ('models.ltx_video', 'LTXVideoPipeline'),
    'ltx2': ('models.ltx2', 'LTX2Pipeline'),
    'lumina_2': ('models.lumina_2', 'Lumina2Pipeline'),
    'minimax_h3': ('models.minimax_h3', 'MinimaxH3Pipeline'),
    'omnigen2': ('models.omnigen2', 'OmniGen2Pipeline'),
    'qwen_image': ('models.qwen_image', 'QwenImagePipeline'),
    'sd3': ('models.sd3', 'SD3Pipeline'),
    'sdxl': ('models.sdxl', 'SDXLPipeline'),
    'wan': ('models.wan.wan', 'WanPipeline'),
    'z_image': ('models.z_image', 'ZImagePipeline'),
}


def resolve_model_pipeline(model_type):
    """Import the selected implementation before distributed initialization."""
    module_name, class_name = MODEL_PIPELINES[model_type]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_most_recent_run_dir(output_dir):
    run_directories = [path for path in glob.glob(os.path.join(output_dir, '*')) if os.path.isdir(path)]
    return sorted(run_directories)[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss, 'step': step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
            if wandb_enable:
                wandb.log({f'{name}/loss': avg_loss, 'step': step})

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        if wandb_enable:
            wandb.log({'eval/eval_time_sec': duration, 'step': step})
        pbar.close()


def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    empty_cuda_cache()
    model.prepare_block_swap_training()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()


if __name__ == '__main__':
    # Inline TOML tables are not pickleable, which messes up the multiprocessing
    # dataset code. The JSON round trip also gives this process a private copy of
    # the already-preflighted config.
    config = json.loads(json.dumps(_preflight_config))

    set_config_defaults(config)
    if not args.cache_only:
        try:
            validate_optimizer_availability(config)
        except ConfigValidationError as exc:
            raise SystemExit(f'\n{exc}') from None

    # With multiple GPUs / large batch sizes, the dataloader can trigger "too many open files" errors unless we do this.
    torch.multiprocessing.set_sharing_strategy('file_system')
    deepspeed.utils.set_log_level_from_string('info')
    apply_patches()
    model_pipeline_class = resolve_model_pipeline(config['model']['type'])

    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Now initialize deepspeed
    deepspeed.init_distributed()

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(local_rank)

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    model = model_pipeline_class(config)

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    dataset_config = json.loads(json.dumps(
        _preflight_dataset_configs[str(Path(config['dataset']).expanduser())]
    ))

    micro_batch_size_per_gpu = config.get('micro_batch_size_per_gpu', 1)
    if isinstance(micro_batch_size_per_gpu, int):
        micro_batch_size_per_gpu = {None: micro_batch_size_per_gpu}
    elif isinstance(micro_batch_size_per_gpu, list):
        micro_batch_size_per_gpu = {x[0]: x[1] for x in micro_batch_size_per_gpu}

    eval_micro_batch_size_per_gpu = config.get('eval_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(eval_micro_batch_size_per_gpu, int):
        eval_micro_batch_size_per_gpu = {None: eval_micro_batch_size_per_gpu}
    elif isinstance(eval_micro_batch_size_per_gpu, list):
        eval_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_micro_batch_size_per_gpu}

    image_micro_batch_size_per_gpu = config.get('image_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(image_micro_batch_size_per_gpu, int):
        image_micro_batch_size_per_gpu = {None: image_micro_batch_size_per_gpu}
    elif isinstance(image_micro_batch_size_per_gpu, list):
        image_micro_batch_size_per_gpu = {x[0]: x[1] for x in image_micro_batch_size_per_gpu}

    eval_image_micro_batch_size_per_gpu = config.get('eval_image_micro_batch_size_per_gpu', eval_micro_batch_size_per_gpu)
    if isinstance(eval_image_micro_batch_size_per_gpu, int):
        eval_image_micro_batch_size_per_gpu = {None: eval_image_micro_batch_size_per_gpu}
    elif isinstance(eval_image_micro_batch_size_per_gpu, list):
        eval_image_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_image_micro_batch_size_per_gpu}

    default_micro_batch_size_per_gpu = list(micro_batch_size_per_gpu.values())[0]

    gradient_release = config['optimizer'].get('gradient_release', False)
    logical_gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    nsync_enabled = config['training_methods']['nsync']['enabled']
    # NSYNC executes positive, negative, and anchor as sequential microbatches,
    # followed by one projected optimizer update.
    engine_gradient_accumulation_steps = logical_gradient_accumulation_steps * (3 if nsync_enabled else 1)
    ds_config = {
        'train_micro_batch_size_per_gpu': default_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': engine_gradient_accumulation_steps,
        # Can't do gradient clipping with gradient release, since there are no grads at the end of the step anymore.
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }
    caching_batch_size = config.get('caching_batch_size', 1)
    trust_cache = args.trust_cache or config.get('trust_cache', False)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, trust_cache=trust_cache, caching_batch_size=caching_batch_size, keep_models_loaded=args.test_sample)

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    if nsync_enabled and not train_data.nsync_enabled:
        raise ValueError(
            'NSYNC is enabled, but the training dataset has no paired nsync_role directories. '
            'See examples/minimax_h3_nsync_self_flow_dataset.toml.'
        )
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        eval_dataset_config = json.loads(json.dumps(
            _preflight_dataset_configs[str(Path(config_path).expanduser())]
        ))
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])

    # For testing

    # import imageio
    # from pathlib import Path
    # import torch.nn.functional as F
    # dataset_manager.cache(unload_models=False)
    # output_dir = Path('/home/anon/tmp')
    # train_data.post_init(
    #     0,
    #     1,
    #     1,
    #     1,
    # )
    # vae = model.vae
    # vae.model.to('cuda')
    # count = 1
    # for item in train_data:
    #     latents = item['latents'].to('cuda')
    #     h, w = latents.shape[-2:]
    #     mask = item['mask'].to('cuda')
    #     caption = item['caption'][0]
    #     mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
    #     mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
    #     mask = mask.unsqueeze(2)  # make mask same number of dims as target
    #     latents = latents * mask.to(latents.device)
    #     video = vae.model.decode(latents, vae.scale).float().clamp_(-1, 1).squeeze(0)
    #     video = torch.permute(video, (1, 2, 3, 0))
    #     video = (video + 1) / 2
    #     video = (video * 255).type(torch.uint8).cpu()
    #     imageio.v3.imwrite(output_dir / f'{count}.mp4', video, fps=16)
    #     with open(output_dir / f'{count}.txt', 'w') as f:
    #         f.write(caption)
    #     if count >= 10:
    #         break
    #     count += 1
    # quit()

    if args.dump_dataset:
        # only works for flux
        import torchvision
        dataset_manager.cache(unload_models=False)
        if is_main_process():
            with torch.no_grad():
                os.makedirs(args.dump_dataset, exist_ok=True)
                vae = model.vae.to('cuda')
                train_data.post_init(
                    0,
                    1,
                    1,
                    1,
                    1,
                )
                for i, item in enumerate(train_data):
                    latents = item['latents']
                    latents = latents / vae.config.scaling_factor
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        latents = latents + vae.config.shift_factor
                    img = vae.decode(latents.to(vae.device, vae.dtype)).sample.to(torch.float32)
                    img = img.squeeze(0)
                    img = ((img + 1) / 2).clamp(0, 1)
                    pil_img = torchvision.transforms.functional.to_pil_image(img)
                    pil_img.save(args.dump_dataset / f'{i}.png')
                    if i >= 100:
                        break
        dist.barrier()
        quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    # Free up as much RAM as we can.
    del dataset_manager
    if config['model'].get('cache_text_embeddings', True):
        # Only ComfyUI-based models, and only if we are caching text embeddings (which most models require).
        model.free_vae_and_te()

    if args.test_sample:
        model.prepare_sample_test('a golden retriever running through a grassy field', cfg=1)

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        if init_from_existing := adapter_config.get('init_from_existing', None):
            model.load_adapter_weights(init_from_existing)
            if hasattr(model, 'sync_self_flow_teacher'):
                model.sync_self_flow_teacher()
    else:
        is_adapter = False

    # Determine run_dir on rank 0 and broadcast it
    run_dir_container = [None]
    if is_main_process():
        if resume_from_checkpoint is True:
            run_dir_container[0] = get_most_recent_run_dir(config['output_dir'])
        elif isinstance(resume_from_checkpoint, str):
            run_dir_container[0] = os.path.join(config['output_dir'], resume_from_checkpoint)
        else:
            run_dir_container[0] = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))

    torch.distributed.broadcast_object_list(run_dir_container, src=0, group=dist.get_world_group())
    run_dir = run_dir_container[0]

    os.makedirs(run_dir, exist_ok=True)
    if not resume_from_checkpoint and is_main_process():
        shutil.copy(args.config, run_dir)
        shutil.copy(config['dataset'], run_dir)
        for eval_dataset in config['eval_datasets']:
            eval_config_path = eval_dataset if isinstance(eval_dataset, str) else eval_dataset['config']
            shutil.copy(eval_config_path, run_dir)
    dist.barrier()

    # WandB logging
    wandb_enable = config.get('monitoring', {}).get('enable_wandb', False)
    if wandb_enable and is_main_process():
        import wandb
        wandb_api_key     = config['monitoring']['wandb_api_key']
        wandb_tracker     = config['monitoring']['wandb_tracker_name']
        wandb_run_name    = config['monitoring']['wandb_run_name']
        logging_dir       = run_dir
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_tracker,
            name=wandb_run_name,
            config=config,
            dir=logging_dir
        )

    # Block swapping
    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        # Don't automatically move to GPU, we'll do that ourselves.
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            # TODO: block swapping doesn't work with Deepspeed non-reentrant checkpoint, but PyTorch native one is fine. Some
            # weights end up on CPU where they shouldn't. Why? Are we giving anything up by not using the Deepspeed implementation?
            #checkpoint_func = deepspeed.checkpointing.non_reentrant_checkpoint
            from functools import partial
            checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=config['reentrant_activation_checkpointing'])
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1)
    partition_method=config.get('partition_method', 'parameters')
    partition_split = config.get('partition_split',[len(layers) / num_stages])
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split,
        loss_fn=model.get_loss_fn(),
        dynamic_shape=True,
        **additional_pipeline_module_kwargs
    )
    model.pipeline_model = pipeline_model
    parameters_to_train = [
        p for p in pipeline_model.parameters()
        if p.requires_grad and not getattr(p, 'is_ema_teacher', False)
    ]

    if config['compile']:
        pipeline_model.compile(dynamic=True)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        config=ds_config,
    )
    # Newer Deepspeed versions fail when pipeline_stages>1 because of a check on this field which defaults to False. But, pipeline
    # parallelism has always relied on "Torch-style" backward(), so I think this is an oversight by Deepspeed devs and it's safe
    # to force this to True to get it to work.
    model_engine._support_torch_style_backward = True
    global_batch_size = model_engine.train_micro_batch_size_per_gpu() * logical_gradient_accumulation_steps * model_engine.grid.get_data_parallel_world_size()
    print(f'Global batch size = {global_batch_size}')

    if args.test_sample:
        import torchvision
        img = model.sample(w=512, h=512)
        img = img.squeeze(0).movedim(-1, 0)
        print(img.shape, img.min().item(), img.max().item())
        torchvision.utils.save_image(img, 'example.png')
        quit()

    if save_every_n_examples := config.pop('save_every_n_examples', None):
        config['save_every_n_steps'] = save_every_n_examples // global_batch_size
        print(f"Computed save_every_n_steps = {config['save_every_n_steps']}")
    if eval_every_n_examples := config.pop('eval_every_n_examples', None):
        config['eval_every_n_steps'] = eval_every_n_examples // global_batch_size
        print(f"Computed eval_every_n_steps = {config['eval_every_n_steps']}")

    def get_optimizer(model_parameters):
        if len(model_parameters) == 0:
            return DummyOptimizer()

        optim_config = config['optimizer']
        optim_type = optim_config['type']
        optim_type_lower = optim_type.lower()

        if beta2_half_life := optim_config.pop('beta2_half_life', None):
            betas = optim_config['betas']
            assert len(betas) == 2
            betas[1] = 0.5 ** (global_batch_size / beta2_half_life)
            print(f'Computed beta2 = {betas[1]}')
            optim_config['betas'] = betas

        args = []
        kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}

        if optim_type_lower == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            # klass = deepspeed.ops.adam.FusedAdam
            klass = torch.optim.AdamW
        elif optim_type_lower == 'adamw8bit':
            import bitsandbytes
            klass = bitsandbytes.optim.AdamW8bit
        elif optim_type_lower == 'adamw_optimi':
            import optimi
            klass = optimi.AdamW
        elif optim_type_lower == 'stableadamw':
            import optimi
            klass = optimi.StableAdamW
        elif optim_type_lower == 'sgd':
            klass = torch.optim.SGD
        elif optim_type_lower == 'adamw8bitkahan':
            from optimizers import adamw_8bit
            klass = adamw_8bit.AdamW8bitKahan
        elif optim_type_lower == 'offload':
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            klass = CPUOffloadOptimizer
            args.append(torch.optim.AdamW)
            kwargs['fused'] = True
        elif optim_type_lower == 'automagic':
            from optimizers import automagic
            klass = automagic.Automagic
        elif optim_type_lower == 'genericoptim':
            from optimizers import generic_optim
            klass = generic_optim.GenericOptim
        else:
            import pytorch_optimizer
            klass = getattr(pytorch_optimizer, optim_type)

        if optim_config.get('gradient_release', False):
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}", ranks=[0])
            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                assert self.mpu.get_data_parallel_world_size() == 1, 'When using gradient release, data parallel world size must be 1. Make sure pipeline_stages = num_gpus.'
                return
            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)
            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1/gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1/gas)

            optimizer_dict = {}
            for pg in model.get_param_groups(model_parameters):
                param_kwargs = kwargs.copy()
                if isinstance(pg, dict):
                    # param group
                    for p in pg['params']:
                        param_kwargs['lr'] = pg['lr']
                        optimizer_dict[p] = klass([p], **param_kwargs)
                else:
                    # param
                    optimizer_dict[pg] = klass([pg], **param_kwargs)

            def optimizer_hook(p):
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))
        elif optim_type_lower == 'genericoptim':
            kwargs['compile'] = config['compile']
            kwargs['mpu'] = pipeline_model.mpu()
            new_param_groups = []
            param_groups = model.get_param_groups(model_parameters)
            for pg in param_groups:
                params = pg.pop('params')
                params_2d = []
                params_other = []
                for p in params:
                    if p.ndim == 2:
                        params_2d.append(p)
                    else:
                        params_other.append(p)
                pg_2d = pg.copy()
                pg_2d['params'] = params_2d
                if kwargs.get('second_moment_type', None) == 'sn':
                    pg_2d['subset_size'] = 'heuristics'
                for key in ('rank', 'proj_type', 'update_proj_gap'):
                    if key in kwargs:
                        pg_2d[key] = kwargs.pop(key)
                new_param_groups.append(pg_2d)
                pg_other = pg
                pg_other['params'] = params_other
                new_param_groups.append(pg_other)
            param_groups = new_param_groups
        else:
            param_groups = model.get_param_groups(model_parameters)

        # split weight decay and no weight decay params
        new_param_groups = []
        for pg in param_groups:
            params_no_wd = []
            params_wd = []
            params = pg.pop('params')
            for p in params:
                if p.ndim == 1 or p.original_name.startswith('llm_adapter.embed'):
                    params_no_wd.append(p)
                else:
                    params_wd.append(p)
            pg_no_wd = pg.copy()
            pg['params'] = params_wd
            pg_no_wd['params'] = params_no_wd
            pg_no_wd['weight_decay'] = 0
            if optim_type_lower == 'genericoptim':
                # If we aren't using weight decay, don't use Muon either (handles LLM adapter embed properly)
                pg_no_wd['muon'] = False
                pg_no_wd['adamuon'] = False
                pg_no_wd['normuon'] = False
            if len(params_wd) > 0:
                new_param_groups.append(pg)
            if len(params_no_wd) > 0:
                new_param_groups.append(pg_no_wd)
        param_groups = new_param_groups

        return klass(param_groups, *args, **kwargs)

    model_engine._configure_optimizer(get_optimizer, parameters_to_train)
    optimizer = model_engine.optimizer

    model.model_engine = model_engine
    if nsync_enabled and model_engine.grid.get_data_parallel_world_size() != 1:
        raise ValueError(
            'NSYNC currently requires data parallel world size 1. Pipeline parallelism is supported; '
            'set pipeline_stages equal to the number of GPUs.'
        )
    if hasattr(model, 'wrap_model_engine'):
        model.wrap_model_engine(model_engine)
    if model_engine.is_pipe_parallel:
         grid = model_engine.grid
         model_engine.first_last_stage_group = dist.new_group(ranks=[grid.pp_group[0], grid.pp_group[-1]])

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        micro_batch_size_per_gpu,
        logical_gradient_accumulation_steps,
        image_micro_batch_size_per_gpu,
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            eval_micro_batch_size_per_gpu,
            config['eval_gradient_accumulation_steps'],
            eval_image_micro_batch_size_per_gpu,
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine, model_engine.gradient_accumulation_steps(), model)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()

    scheduler_type = config.get('lr_scheduler', 'constant')
    if scheduler_type == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=config['epochs'] * steps_per_epoch)
    elif scheduler_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] * steps_per_epoch, eta_min=1e-6)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler: {scheduler_type}')
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    step = 1
    examples = global_batch_size
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        param_groups = optimizer.param_groups.copy()
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config and not args.reset_optimizer and not args.reset_optimizer_params,
            load_optimizer_states=not args.reset_optimizer,
        )
        if args.reset_optimizer_params:
            optimizer.param_groups = param_groups
        dist.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        if args.reset_dataloader:
            train_dataloader.epoch = client_state['custom_loader']['epoch']
        else:
            train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        if 'examples' in client_state:
            examples = client_state['examples'] + global_batch_size
        else:
            examples = step * global_batch_size
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model, model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

    # TODO: this is state we need to save and resume when resuming from checkpoint. It only affects logging.
    epoch_loss = 0
    num_steps = 0
    empty_cuda_cache()
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        loss = model_engine.train_batch(iterator).item()
        epoch_loss += loss
        num_steps += 1
        train_dataloader.sync_epoch()

        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step, examples)
        finished_epoch = True if new_epoch != epoch else False

        x_axis = examples if config['x_axis_examples'] else step

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss, x_axis)
            if hasattr(optimizer, '_grad_norm'):
                tb_writer.add_scalar(f'train/grad_norm', optimizer._grad_norm, x_axis)
            if hasattr(model, 'last_nsync_stats'):
                for name, value in model.last_nsync_stats.items():
                    tb_writer.add_scalar(f'train/nsync_{name}', value, x_axis)
            if wandb_enable:
                log_values = {'train/loss': loss, 'step': x_axis}
                if hasattr(optimizer, '_grad_norm'):
                    log_values['train/grad_norm'] = optimizer._grad_norm
                if hasattr(model, 'last_nsync_stats'):
                    for name, value in model.last_nsync_stats.items():
                        log_values[f'train/nsync_{name}'] = value
                wandb.log(log_values)
            if optimizer.__class__.__name__ == 'Prodigy':
                prodigy_d = get_prodigy_d(optimizer)
                tb_writer.add_scalar(f'train/prodigy_d', prodigy_d, x_axis)
            if optimizer.__class__.__name__ in ('Automagic', 'GenericOptim'):
                lrs, avg_lr = _get_automagic_lrs(optimizer)
                if avg_lr > 0:
                    tb_writer.add_histogram(f'train/automagic_lrs', lrs, x_axis)
                    tb_writer.add_scalar(f'train/automagic_avg_lr', avg_lr, x_axis)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model, model_engine, eval_dataloaders, tb_writer, x_axis, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

        if finished_epoch:
            if is_main_process():
                tb_writer.add_scalar(f'train/epoch_loss', epoch_loss/num_steps, epoch)
                if wandb_enable:
                    wandb.log({'train/epoch_loss': epoch_loss/num_steps, 'epoch': epoch})
            epoch_loss = 0
            num_steps = 0
            if new_epoch is None:
                final_model_name = f'epoch{epoch}'
                break
            epoch = new_epoch

        checkpointed, saved = saver.process_step(step, examples)
        if 'max_steps' in config and step >= config['max_steps']:
            final_model_name = f'step{step}'
            break
        step += 1
        examples += global_batch_size

    # Save final training state checkpoint and model, unless we just saved them.
    if not checkpointed:
        saver.save_checkpoint(step, examples)
    if not saved:
        saver.save_model(final_model_name)

    if is_main_process():
        print('TRAINING COMPLETE!')
