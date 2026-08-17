import os
import sys
import types
import math
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
import comfy_kitchen as ck
from peft.tuners.tuners_utils import BaseTunerLayer

from models.base import ComfyPipeline, make_contiguous, PreprocessMediaFile, ModelWrapper
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, one_at_a_time, round_down_to_multiple
from utils.offloading import ModelOffloader
from utils.nsync import NSYNCGradientController
from utils.self_flow import mask_to_runs, representation_cosine_loss, sample_bernoulli_mask
import comfy.latent_formats
import comfy.model_management
import comfy.ldm.minimax.model
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.minimax.model import (
    PackedLayout, time_shift_sigma, patchify_video, pack_audio,
    rope_rotation_table, unpatchify_video, MLP,
)

FRAMERATE = 24  # fixed for this model


def _mod_scale_shift(h, shift, scale, segments):
    dtype = h.dtype
    pieces = []
    # segments: [(start, stop, mod_row)] covering h contiguously.
    for a, b, row in segments:
        piece = h[:, a:b] * (1.0 + scale[:, row, None].to(dtype)) + shift[:, row, None].to(dtype)
        pieces.append(piece)
    return torch.cat(pieces, dim=1)

def _mod_gate(x, gate, other, segments):
    dtype = x.dtype
    pieces = []
    # other is the fresh attn/mlp output: accumulate the gated residual into the stream in place, one fused kernel per segment
    for a, b, row in segments:
        piece = x[:, a:b] + other[:, a:b] * gate[:, row, None].to(dtype)
        pieces.append(piece)
    return torch.cat(pieces, dim=1)

# patch these to remove in-place operations which break backward pass
comfy.ldm.minimax.model._mod_scale_shift = _mod_scale_shift
comfy.ldm.minimax.model._mod_gate = _mod_gate


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = operations.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = operations.Linear(inner, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x, rope_freqs=None, attention_mask=None, transformer_options={}):
        b, s = x.shape[:2]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(b, s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # fused per-head RMSNorm + partial split-half rope
            q = q.view(b, s, self.heads, self.head_dim)
            k = k.view(b, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            # this is seemingly the only way to force eager
            q, k = ck.backends.eager.rope.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            q = self.q_norm(q.view(b, s, self.heads, self.head_dim))
            k = self.k_norm(k.view(b, s, self.heads, self.head_dim))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = optimized_attention(q, k, v, self.heads, mask=attention_mask, skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out)

# Patch to force eager rms_rope_split_half so backward works. Even using offical methods to set 'eager' in comfy kitchen doesn't work.
comfy.ldm.minimax.model.Attention = Attention


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = operations.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device)

    def forward(self, t_emb):
        # [B, M, t_dim] -> expand tensors of [B, M*modalities, hidden]
        x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0], x.shape[1] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)

# batch dimension
comfy.ldm.minimax.model.AdalnProj = AdalnProj


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 apply_silu=True, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)

    def forward(self, x, t_emb, mod_segments, rope_freqs, attention_mask=None, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, attention_mask=attention_mask, transformer_options=transformer_options), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

# add attention_mask
comfy.ldm.minimax.model.DiTBlock = DiTBlock


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True, adaln_dtype=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)
        # output heads are the checkpoint's fp32 island; norm/adaln are stored at model dtype
        self.video_out = operations.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = operations.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)

    def forward(self, x, t_emb, video_segments, audio_segments):
        # Each segment is (start, stop, timestep_row). Self-Flow can therefore
        # use a different final AdaLN timestep for arbitrary target-token runs.
        def normalize_segments(segments):
            if torch.is_tensor(segments):
                segments = segments.tolist()
            if len(segments) == 3 and not isinstance(segments[0], (list, tuple)):
                segments = [segments]
            return segments

        video_segments = normalize_segments(video_segments)
        audio_segments = normalize_segments(audio_segments)

        shift, scale = self.adaln_proj(t_emb)
        normalized = self.norm(x)

        def project_segments(segments, head):
            pieces = []
            for start, stop, row in segments:
                hidden = (
                    normalized[:, start:stop]
                    * (1.0 + scale[:, row, None])
                    + shift[:, row, None]
                ).to(torch.float32)
                pieces.append(head(hidden))
            return torch.cat(pieces, dim=1)

        return (
            project_segments(video_segments, self.video_out),
            project_segments(audio_segments, self.audio_out),
        )

# batch-enabled
comfy.ldm.minimax.model.FinalLayer = FinalLayer


def unpack_audio(rows, ch=2):
    b, s, C = rows.shape
    t = s // ch
    return rows.reshape(b, ch, t, C).permute(0, 3, 1, 2)

# batch-enabled
comfy.ldm.minimax.model.unpack_audio = unpack_audio


class PreprocessMediaFileMinimax(PreprocessMediaFile):
    def __init__(self, config):
        super().__init__(config, support_video=True, support_audio=True, framerate=FRAMERATE, audio_sample_rate=32000, round_height=32, round_width=32)

    # No offsets. VAE simply encodes each chunk of 17 frames into 5 latent frames, and then
    # slices off the last 3 latent frames from the full latent. 1 frame is special case:
    # 1 video frame -> 1 latent frame.
    def align_frames(self, frames):
        return max(round_down_to_multiple(frames, 17), 1)


class MinimaxH3Pipeline(ComfyPipeline):
    name = 'minimax_h3'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['DiTBlock', 'RefinerBlock']
    keep_in_high_precision = ['time_embedder', 'audio_patch_proj', 'condition_proj', 'final_layer', 'rope.inv_freq', 'token_refiner', 'video_patch_proj', 'adaln_t_table']
    spatial_compression = 16
    channels = 24
    is_video_vae = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = self.model_config.get('mode', 't2v').lower()
        if self.mode not in ('t2v', 'i2v'):
            raise ValueError(f'MiniMax H3 model.mode must be t2v or i2v, got {self.mode!r}')
        self.i2v = self.mode == 'i2v'
        if self.i2v:
            # Keep I2V caches separate from older T2V caches. I2V adds both a
            # separately encoded first-frame latent and Qwen vision tokens.
            # Version the first supported schema so incomplete experimental
            # I2V caches cannot be mistaken for working conditioning caches.
            self.name = 'minimax_h3_i2v_v1'
        self.i2v_visual_cond_timestep = float(
            self.model_config.get('i2v_visual_cond_timestep', 0.999)
        )
        if not 0 <= self.i2v_visual_cond_timestep <= 1:
            raise ValueError(
                'MiniMax H3 model.i2v_visual_cond_timestep must be in [0, 1], '
                f'got {self.i2v_visual_cond_timestep}'
            )
        self.is_video_vae = True
        self.latent_format = comfy.latent_formats.MiniMaxH3Video()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.framerate = FRAMERATE

        training_methods = self.config.get('training_methods', {})
        self.nsync_config = training_methods.get('nsync', {})
        self.self_flow_config = training_methods.get('self_flow', {})
        self.nsync_enabled = self.nsync_config.get('enabled', False)
        self.self_flow_enabled = self.self_flow_config.get('enabled', False)
        self.nsync_controller = None
        if self.nsync_enabled:
            self.nsync_controller = NSYNCGradientController(
                eps=self.nsync_config.get('eps', 1e-8),
                # DeepSpeed averages over the three role microbatches. Restore
                # the logical positive-batch gradient scale after surgery.
                gradient_scale=3.0,
            )

        self.self_flow_gamma = float(self.self_flow_config.get('gamma', 0.8))
        self.self_flow_ema_decay = float(self.self_flow_config.get('ema_decay', 0.9999))
        if self.self_flow_gamma < 0:
            raise ValueError(f'Self-Flow gamma must be non-negative, got {self.self_flow_gamma}')
        if not 0 <= self.self_flow_ema_decay < 1:
            raise ValueError(f'Self-Flow ema_decay must be in [0, 1), got {self.self_flow_ema_decay}')
        ema_dtype_name = self.self_flow_config.get('ema_dtype', 'float32')
        ema_dtypes = {'float32': torch.float32, 'bfloat16': torch.bfloat16}
        if ema_dtype_name not in ema_dtypes:
            raise ValueError(f'Self-Flow ema_dtype must be one of {sorted(ema_dtypes)}, got {ema_dtype_name!r}')
        self.self_flow_ema_dtype = ema_dtypes[ema_dtype_name]
        self.self_flow_image_mask_ratio = float(self.self_flow_config.get('image_mask_ratio', 0.25))
        self.self_flow_video_mask_ratio = float(self.self_flow_config.get('video_mask_ratio', 0.1))
        self.self_flow_audio_mask_ratio = float(self.self_flow_config.get('audio_mask_ratio', 0.5))
        self.self_flow_high_noise_fraction = float(self.self_flow_config.get('high_noise_fraction', 0.0))
        if not 0 <= self.self_flow_high_noise_fraction <= 1:
            raise ValueError(
                f'Self-Flow high_noise_fraction must be in [0, 1], got {self.self_flow_high_noise_fraction}'
            )
        high_noise_range = self.self_flow_config.get('high_noise_range', [0.95, 1.0])
        if len(high_noise_range) != 2 or not 0 <= high_noise_range[0] < high_noise_range[1] <= 1:
            raise ValueError(f'Self-Flow high_noise_range must satisfy 0 <= low < high <= 1, got {high_noise_range}')
        for name, ratio in (
            ('image_mask_ratio', self.self_flow_image_mask_ratio),
            ('video_mask_ratio', self.self_flow_video_mask_ratio),
            ('audio_mask_ratio', self.self_flow_audio_mask_ratio),
        ):
            if not 0 <= ratio <= 0.5:
                raise ValueError(f'Self-Flow {name} must be in [0, 0.5], got {ratio}')

        self.cfg = self.model_config.get('cfg', 1.0)
        if self.cfg > 1:
            # Because uncond branch is no_grad() but tensors passed between pipeline parallel layers must require grad.
            assert self.config['pipeline_stages'] == 1, 'CFG training requires pipeline_stages=1'

        # combined video and (optional) audio VAE in one object
        def load_fn():
            sd = comfy.utils.load_torch_file(self.model_config['vae'])
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()

            def vae_encode_crop_pixels(self, pixels):
                if not self.crop_input:
                    return pixels

                downscale_ratio = self.spacial_compression_encode()

                dims = pixels.shape[-3:-1]
                for d in range(len(dims)):
                    x = (dims[d] // downscale_ratio) * downscale_ratio
                    x_offset = (dims[d] % downscale_ratio) // 2
                    if x != dims[d]:
                        pixels = pixels.narrow(d + 1, x_offset, x)
                return pixels

            # patch this to handle 5D video tensor (original code expects 4D even for video)
            vae.vae_encode_crop_pixels = types.MethodType(vae_encode_crop_pixels, vae)
            vae_list = [vae]

            # audio VAE
            sd = comfy.utils.load_torch_file(self.model_config['audio_vae'])
            vae = comfy.sd.VAE(sd=sd)
            vae.throw_exception_if_invalid()
            vae_list.append(vae)

            return vae_list

        self.vae = ModelWrapper(load_fn)

    def load_diffusion_model(self):
        # Model is so big, it's easy to OOM while loading with multiple GPUs.
        with one_at_a_time():
            rank = int(os.environ['LOCAL_RANK'])
            print(f'Loading model on rank {rank}')
            super().load_diffusion_model()

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        if (self.self_flow_enabled or self.nsync_enabled) and adapter_config['type'] != 'lora':
            raise ValueError('MiniMax H3 NSYNC and Self-Flow currently require adapter.type=lora')

        if self.self_flow_enabled:
            self._setup_self_flow_teacher_adapter()

        if self.nsync_enabled:
            student_parameters = [
                parameter
                for name, parameter in self.diffusion_model.named_parameters()
                if parameter.requires_grad and '.default.' in name and 'lora_' in name
            ]
            self.nsync_controller.register_parameters(student_parameters)

    def _iter_lora_layers(self):
        for module in self.diffusion_model.modules():
            if isinstance(module, BaseTunerLayer):
                yield module

    @staticmethod
    def _set_layer_adapter(module, adapter_name):
        # PEFT's public set_adapter also toggles requires_grad. That is unsafe
        # between a student forward and its later pipeline-scheduled backward,
        # so select the already-created adapter without changing leaf state.
        module._active_adapter = [adapter_name]

    def set_self_flow_adapter(self, adapter_name):
        if not self.self_flow_enabled:
            return
        for module in self._iter_lora_layers():
            self._set_layer_adapter(module, adapter_name)

    @staticmethod
    @torch.no_grad()
    def _copy_adapter_weights(module, source_name, target_name, decay=None, link_for_save=False):
        for container_name in ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B', 'lora_magnitude_vector'):
            container = getattr(module, container_name, None)
            if container is None or source_name not in container or target_name not in container:
                continue
            source = container[source_name]
            target = container[target_name]
            source_parameters = list(source.parameters()) if isinstance(source, nn.Module) else [source]
            target_parameters = list(target.parameters()) if isinstance(target, nn.Module) else [target]
            if len(source_parameters) != len(target_parameters):
                raise RuntimeError(f'PEFT adapter parameter mismatch in {container_name}')
            for source_parameter, target_parameter in zip(source_parameters, target_parameters):
                if link_for_save:
                    # Self-Flow evaluates with EMA weights. The saver uses this
                    # link to emit the teacher value under the ordinary student
                    # LoRA key, without including a second adapter in the file.
                    source_parameter.ema_save_source = target_parameter
                if decay is None:
                    target_parameter.copy_(source_parameter)
                else:
                    target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)

    def _setup_self_flow_teacher_adapter(self):
        self.self_flow_student_adapter = 'default'
        self.self_flow_teacher_adapter = 'self_flow_teacher'
        try:
            self.lora_model.add_adapter(
                self.self_flow_teacher_adapter,
                self.peft_config,
                autocast_adapter_dtype=False,
            )
        except TypeError:
            # Compatibility with older PEFT releases.
            self.lora_model.add_adapter(self.self_flow_teacher_adapter, self.peft_config)

        for module in self._iter_lora_layers():
            for container_name in (
                'lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B', 'lora_magnitude_vector'
            ):
                container = getattr(module, container_name, None)
                if container is None or self.self_flow_teacher_adapter not in container:
                    continue
                target = container[self.self_flow_teacher_adapter]
                target_parameters = list(target.parameters()) if isinstance(target, nn.Module) else [target]
                for target_parameter in target_parameters:
                    target_parameter.data = target_parameter.data.to(self.self_flow_ema_dtype)
            self._copy_adapter_weights(
                module,
                self.self_flow_student_adapter,
                self.self_flow_teacher_adapter,
                link_for_save=True,
            )
            if hasattr(module, 'lora_dropout') and self.self_flow_teacher_adapter in module.lora_dropout:
                module.lora_dropout[self.self_flow_teacher_adapter] = nn.Identity()
            self._set_layer_adapter(module, self.self_flow_student_adapter)

        # Frozen parameters are omitted by the repository's DeepSpeed checkpoint
        # flow. Keep EMA leaves checkpoint-visible but explicitly exclude them
        # from the optimizer and final LoRA artifact.
        for name, parameter in self.diffusion_model.named_parameters():
            if f'.{self.self_flow_teacher_adapter}.' in name:
                parameter.requires_grad_(True)
                parameter.is_ema_teacher = True
                parameter.skip_adapter_save = True
                parameter.original_name = name
            elif '.default.' in name and 'lora_' in name:
                parameter.requires_grad_(True)

    @torch.no_grad()
    def sync_self_flow_teacher(self):
        if not self.self_flow_enabled:
            return
        for module in self._iter_lora_layers():
            self._copy_adapter_weights(module, self.self_flow_student_adapter, self.self_flow_teacher_adapter)
        self.set_self_flow_adapter(self.self_flow_student_adapter)

    @torch.no_grad()
    def update_self_flow_teacher(self):
        if not self.self_flow_enabled:
            return
        for module in self._iter_lora_layers():
            self._copy_adapter_weights(
                module,
                self.self_flow_student_adapter,
                self.self_flow_teacher_adapter,
                decay=self.self_flow_ema_decay,
            )
        self.set_self_flow_adapter(self.self_flow_student_adapter)

    def wrap_model_engine(self, model_engine):
        if not (self.nsync_enabled or self.self_flow_enabled):
            return
        original_take_model_step = model_engine._take_model_step

        def take_model_step(engine, *args, **kwargs):
            if self.nsync_enabled:
                self.last_nsync_stats = self.nsync_controller.apply_gradient_surgery()
            skipped_steps_before = getattr(engine, 'skipped_steps', 0)
            result = original_take_model_step(*args, **kwargs)
            if (
                self.self_flow_enabled
                and getattr(engine, 'skipped_steps', 0) == skipped_steps_before
            ):
                self.update_self_flow_teacher()
            return result

        model_engine._take_model_step = types.MethodType(take_model_step, model_engine)

    # Override to exclude adaln, since full and pruned model have different sizes. Makes LoRA compatible with both.
    def get_target_modules(self, target_model):
        target_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if 'adaln' in full_submodule_name:
                    continue
                if isinstance(submodule, nn.Linear):
                    target_modules.add(full_submodule_name)
        return list(target_modules)

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFileMinimax(self.config)

    def get_call_text_encoder_fn(self, text_encoder):
        if not self.i2v:
            return super().get_call_text_encoder_fn(text_encoder)

        te_idx = None
        for index, candidate in enumerate(self.text_encoders):
            if text_encoder == candidate:
                te_idx = index
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], media: list):
            if not len(captions) == len(is_video) == len(media):
                raise ValueError('MiniMax H3 I2V text caching received mismatched captions and media')

            text_embeds, attention_masks, token_tags = [], [], []
            uncond_text_embeds, uncond_attention_masks, uncond_token_tags = [], [], []

            def encode(caption, images):
                tokens = text_encoder.tokenize(caption, images=images)
                encoded = text_encoder.encode_from_tokens_scheduled(tokens)
                embeds = encoded[0][0][0].to(self.dtype)
                extra = encoded[0][1]
                tags = extra.get('minimax_token_tags')
                if tags is None:
                    tags = torch.ones(embeds.shape[0], dtype=torch.int64, device=embeds.device)
                else:
                    tags = tags.to(device=embeds.device, dtype=torch.int64).reshape(-1)
                    if tags.shape[0] != embeds.shape[0]:
                        raise RuntimeError(
                            'MiniMax H3 text token tags did not match the expanded multimodal sequence: '
                            f'{tags.shape[0]} tags for {embeds.shape[0]} embeddings'
                        )
                attention_mask = torch.ones(
                    embeds.shape[0], dtype=torch.int64, device=embeds.device
                )
                return embeds, attention_mask, tags

            for caption, video, first_frame in zip(captions, is_video, media):
                images = [first_frame] if video else []
                if video and first_frame is None:
                    raise ValueError('MiniMax H3 I2V video is missing its first-frame text conditioning')

                embeds, attention_mask, tags = encode(caption, images)
                text_embeds.append(embeds)
                attention_masks.append(attention_mask)
                token_tags.append(tags)

                # CFG and classifier-free caption dropout must retain the same
                # first-frame vision presentation. A single global empty-prompt
                # embedding would silently remove those image tokens.
                embeds, attention_mask, tags = encode('', images)
                uncond_text_embeds.append(embeds)
                uncond_attention_masks.append(attention_mask)
                uncond_token_tags.append(tags)

            # Lists intentionally preserve variable sequence lengths. The
            # dataset cache unbatches them into one tensor per example.
            return {
                f'text_embeds_{te_idx}': text_embeds,
                f'attention_mask_{te_idx}': attention_masks,
                f'minimax_token_tags_{te_idx}': token_tags,
                f'uncond_text_embeds_{te_idx}': uncond_text_embeds,
                f'uncond_attention_mask_{te_idx}': uncond_attention_masks,
                f'uncond_minimax_token_tags_{te_idx}': uncond_token_tags,
            }

        return fn

    def vae_encode(self, img: torch.Tensor, audio: list):
        video_vae, audio_vae = self.vae._model
        # move channel dim to end
        # works for both images (b c h w) and video (b c f h w)
        img = img.movedim(1, -1)
        latents = video_vae.encode(img)
        if self.latent_format is not None:
            # some older models do this in prepare_inputs() so it can be None
            latents = self.latent_format.process_in(latents)

        audio_latents = []
        for a in audio:
            if a is not None:
                assert a.shape[0] == 1
                a_latents = audio_vae.encode(a.movedim(1, -1))
            else:
                a_latents = None
            audio_latents.append(a_latents)

        return latents, audio_latents

    def get_call_vae_fn(self, vae):
        def fn(images: torch.Tensor, audio: list):
            if images.shape[2] > 1:
                # check videos for missing audio and warn
                missing_audio = sum(1 if a is None else 0 for a in audio)
                if missing_audio > 0:
                    print(
                        f'WARNING: this batch had {missing_audio} videos without audio. '
                        'Those samples will train video only and contribute no audio tokens or audio loss.'
                    )
            images = images.to('cuda', self.dtype)
            latents, audio_latents = self.vae_encode(images, audio)
            result = {'latents': latents, 'audio_latents': audio_latents}
            if self.i2v:
                if images.shape[2] > 1:
                    # ComfyUI FL2VA encodes the keyframe independently. A
                    # slice of the full video latent is not equivalent because
                    # the H3 VAE's temporal path sees neighboring frames.
                    first_frame_latents, _ = self.vae_encode(
                        images[:, :, :1], [None] * images.shape[0]
                    )
                    result['i2v_first_frame_latents'] = first_frame_latents
                else:
                    # Images remain ordinary T2I/T2V-mode examples even when
                    # the run is configured for I2V. The empty temporal axis is
                    # a batchable sentinel consumed by prepare_inputs().
                    result['i2v_first_frame_latents'] = latents.new_empty(
                        (latents.shape[0], latents.shape[1], 0, latents.shape[-2], latents.shape[-1])
                    )
            return result
        return fn

    def to_layers(self):
        diffusion_model = self.diffusion_model
        depth = len(diffusion_model.blocks)
        student_layer = self.self_flow_config.get('student_layer', None)
        teacher_layer = self.self_flow_config.get('teacher_layer', None)
        if student_layer is None:
            student_layer = max(1, round(float(self.self_flow_config.get('student_layer_ratio', 0.3)) * depth))
        if teacher_layer is None:
            teacher_layer = max(1, round(float(self.self_flow_config.get('teacher_layer_ratio', 0.7)) * depth))
        self.self_flow_student_layer = int(student_layer) - 1
        self.self_flow_teacher_layer = int(teacher_layer) - 1
        if self.self_flow_enabled and not (
            0 <= self.self_flow_student_layer < self.self_flow_teacher_layer < depth
        ):
            raise ValueError(
                f'Self-Flow layers must satisfy 1 <= student_layer < teacher_layer <= {depth}; '
                f'got {student_layer} and {teacher_layer}'
            )

        layers = [InitialLayer(diffusion_model, self)]
        for i, block in enumerate(diffusion_model.blocks):
            layers.append(TransformerLayer(block, i, self.offloader, self))
            if self.self_flow_enabled and i == self.self_flow_teacher_layer:
                layers.append(SelfFlowLossLayer(diffusion_model, self))
        layers.append(FinalLayer(diffusion_model, self.cfg, self))
        return layers

    # def to_layers(self):
    #     return [Wrapper(self.diffusion_model)]

    def get_conds(self, inputs, prefix=''):
        text_embeds = inputs[f'{prefix}text_embeds_0']
        attention_mask = inputs[f'{prefix}attention_mask_0']
        # text embeds are variable length
        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        assert text_embeds.shape[:2] == attention_mask.shape[:2]
        attention_mask = attention_mask.to(torch.bool)
        if not self.i2v:
            return text_embeds, attention_mask

        token_tags = inputs[f'{prefix}minimax_token_tags_0']
        token_tags = torch.stack([
            torch.cat([u, u.new_ones(max_seq_len - u.size(0))]) for u in token_tags
        ]).to(torch.int64)
        assert token_tags.shape == attention_mask.shape
        return text_embeds, attention_mask, token_tags

    def _sample_timesteps(self, batch_size, device, frames, height, width, timestep_quantile=None):
        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')
        if timestep_sample_method == 'logit_normal':
            distribution = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            distribution = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError(f'Unknown timestep_sample_method={timestep_sample_method}')

        if timestep_quantile is not None:
            timesteps = distribution.icdf(torch.full((batch_size,), timestep_quantile, device=device))
        else:
            timesteps = distribution.sample((batch_size,)).to(device)

        if timestep_sample_method == 'logit_normal':
            timesteps = torch.sigmoid(timesteps * self.model_config.get('sigmoid_scale', 1.0))

        shift = self.model_config.get('shift', None)
        if frames == 1:
            shift = self.model_config.get('image_shift', shift)
        if shift:
            timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((height // 2) * (width // 2))
            timesteps = time_shift(mu, 1.0, timesteps)

        # The Self-Flow video ablation found that explicitly retaining a small
        # amount of very-low-SNR training data can help a logit-normal schedule.
        high_noise_fraction = self.self_flow_high_noise_fraction
        if (
            self.self_flow_enabled
            and frames > 1
            and timestep_quantile is None
            and high_noise_fraction > 0
        ):
            low, high = self.self_flow_config.get('high_noise_range', [0.95, 1.0])
            replace = torch.rand(batch_size, device=device) < high_noise_fraction
            high_noise = torch.empty(batch_size, device=device).uniform_(float(low), float(high))
            timesteps = torch.where(replace, high_noise, timesteps)
        return timesteps

    def _micro_batch_size_for_latents(self, frames, height, width):
        config_key = 'image_micro_batch_size_per_gpu' if frames == 1 else 'micro_batch_size_per_gpu'
        configured = self.config.get(config_key, self.config['micro_batch_size_per_gpu'])
        if isinstance(configured, int):
            return configured

        batch_sizes = dict(configured)
        if None in batch_sizes:
            return batch_sizes[None]
        pixel_size = math.sqrt(height * width) * self.spatial_compression
        closest_size = min(batch_sizes, key=lambda size: abs(float(size) - pixel_size))
        return batch_sizes[closest_size]

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        bs, c, f, h, w = latents.shape
        device = latents.device

        i2v_first_frame_latents = None
        if self.i2v:
            i2v_first_frame_latents = inputs['i2v_first_frame_latents']
            if not torch.is_tensor(i2v_first_frame_latents) or i2v_first_frame_latents.ndim != 5:
                raise RuntimeError(
                    'MiniMax H3 I2V expected first-frame latents with shape [B, C, T, H, W]'
                )
            if i2v_first_frame_latents.shape[0] != bs:
                raise RuntimeError(
                    'MiniMax H3 I2V first-frame latent batch did not match the target batch'
                )
            cond_frames = i2v_first_frame_latents.shape[2]
            if cond_frames not in (0, 1):
                raise RuntimeError(
                    f'MiniMax H3 I2V expected zero or one conditioning latent frame, got {cond_frames}'
                )
            if cond_frames == 1 and i2v_first_frame_latents.shape[1:] != (c, 1, h, w):
                raise RuntimeError(
                    'MiniMax H3 I2V first-frame latent shape did not match the target video: '
                    f'{tuple(i2v_first_frame_latents.shape)} vs {tuple(latents.shape)}'
                )
            i2v_first_frame_latents = i2v_first_frame_latents.to(device=device, dtype=torch.float32)
            if cond_frames == 1 and self.i2v_visual_cond_timestep < 1:
                # Match ComfyUI's FL2VA conditioning augmentation. Sample it
                # once here so the primary, Self-Flow teacher, and CFG branches
                # all see exactly the same visual anchor.
                visual_noise = torch.randn_like(i2v_first_frame_latents)
                i2v_first_frame_latents = (
                    self.i2v_visual_cond_timestep * i2v_first_frame_latents
                    + (1 - self.i2v_visual_cond_timestep) * visual_noise
                )

        audio_latents_list = inputs['audio_latents'] if 'audio_latents' in inputs else [None]*bs

        # prepare single global batched audio tensor, some may be None
        # videos are bucketed, so non-None audio should all be same length
        audio_shape = None
        audio_latents = None
        valid_audio = []
        for i, a in enumerate(audio_latents_list):
            if a is not None:
                if audio_shape is None:
                    audio_shape = a.shape
                    audio_latents = torch.empty((bs, *audio_shape[1:]), dtype=a.dtype, device=device)
                assert a.shape == audio_shape
                audio_latents[i, ...] = a
                valid_audio.append(True)
            else:
                valid_audio.append(False)
        valid_audio = torch.tensor(valid_audio, device=device)

        if audio_latents is None:
            audio_latents = torch.empty((bs, 32, 2, 0), dtype=latents.dtype, device=device)

        conds = self.get_conds(inputs)

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)

        t = self._sample_timesteps(bs, device, f, h, w, timestep_quantile=timestep_quantile)

        noise = torch.randn_like(latents)
        target = noise - latents

        # audio
        audio_noise = torch.randn_like(audio_latents)
        # fixed t -> audio_t mapping, matches the shift used in model code to derive the audio t from video t
        audio_t = time_shift_sigma(t, self.diffusion_model.sigma_shift_video, self.diffusion_model.sigma_shift_audio)
        audio_target = audio_noise - audio_latents

        if self.self_flow_enabled:
            training_self_flow = timestep_quantile is None
            if training_self_flow:
                s = self._sample_timesteps(bs, device, f, h, w)
                video_mask_ratio = self.self_flow_image_mask_ratio if f == 1 else self.self_flow_video_mask_ratio
                _, patch_h, patch_w = self.diffusion_model.patch_size
                micro_batch_size = self._micro_batch_size_for_latents(f, h, w)
                if bs % micro_batch_size != 0:
                    raise RuntimeError(
                        f'Self-Flow prepared {bs} examples, which is not divisible by the '
                        f'configured physical microbatch size {micro_batch_size}'
                    )
                mask_groups = bs // micro_batch_size
                # MiniMax's batch-enabled AdaLN path currently shares segment
                # boundaries within a physical microbatch. Sample independently
                # between physical microbatches and share only within each one.
                video_token_mask = sample_bernoulli_mask(
                    (mask_groups, f, h // patch_h, w // patch_w), video_mask_ratio, device
                )
                video_token_mask = video_token_mask.repeat_interleave(micro_batch_size, dim=0)
                audio_token_mask = sample_bernoulli_mask(
                    (mask_groups, 2, audio_latents.shape[-1]), self.self_flow_audio_mask_ratio, device
                )
                audio_token_mask = audio_token_mask.repeat_interleave(micro_batch_size, dim=0)
                self_flow_weight = torch.ones(bs, dtype=torch.float32, device=device)
            else:
                # Quantile evaluation remains homogeneous and reports only the
                # generation objective, while keeping pipeline tensor structure.
                s = t
                _, patch_h, patch_w = self.diffusion_model.patch_size
                video_token_mask = torch.zeros((bs, f, h // patch_h, w // patch_w), dtype=torch.bool, device=device)
                audio_token_mask = torch.zeros((bs, 2, audio_latents.shape[-1]), dtype=torch.bool, device=device)
                self_flow_weight = torch.zeros(bs, dtype=torch.float32, device=device)

            video_mask_expanded = video_token_mask.unsqueeze(1)
            video_mask_expanded = video_mask_expanded.repeat_interleave(patch_h, dim=-2).repeat_interleave(patch_w, dim=-1)
            video_sigmas = torch.where(
                video_mask_expanded,
                s.view(-1, 1, 1, 1, 1),
                t.view(-1, 1, 1, 1, 1),
            )
            noisy_latents = (1 - video_sigmas) * latents + video_sigmas * noise

            audio_s = time_shift_sigma(s, self.diffusion_model.sigma_shift_video, self.diffusion_model.sigma_shift_audio)
            audio_sigmas = torch.where(
                audio_token_mask.unsqueeze(1),
                audio_s.view(-1, 1, 1, 1),
                audio_t.view(-1, 1, 1, 1),
            )
            noisy_audio_latents = (1 - audio_sigmas) * audio_latents + audio_sigmas * audio_noise

            minimum_t = torch.minimum(t, s)
            minimum_t_expanded = minimum_t.view(-1, 1, 1, 1, 1)
            teacher_latents = (1 - minimum_t_expanded) * latents + minimum_t_expanded * noise
            minimum_audio_t = time_shift_sigma(
                minimum_t,
                self.diffusion_model.sigma_shift_video,
                self.diffusion_model.sigma_shift_audio,
            )
            minimum_audio_t_expanded = minimum_audio_t.view(-1, 1, 1, 1)
            teacher_audio_latents = (
                (1 - minimum_audio_t_expanded) * audio_latents
                + minimum_audio_t_expanded * audio_noise
            )
        else:
            t_expanded = t.view(-1, 1, 1, 1, 1)
            noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
            audio_t_expanded = audio_t.view(-1, 1, 1, 1)
            noisy_audio_latents = (1 - audio_t_expanded) * audio_latents + audio_t_expanded * audio_noise
            self_flow_weight = torch.zeros(bs, dtype=torch.float32, device=device)

        if self.cfg > 1:
            if self.i2v:
                # These are cached per example with the same first-frame image
                # tokens as the conditional presentation.
                unconds = self.get_conds(inputs, prefix='uncond_')
            else:
                tmp = {}
                for k in ('text_embeds_0', 'attention_mask_0'):
                    v = self.uncond_dict[k]
                    if k == 'attention_mask_0':
                        v = v.to(torch.bool)
                    tmp[k] = v.unsqueeze(0).repeat(bs, *([1]*v.ndim))
                unconds = self.get_conds(tmp)
        else:
            unconds = tuple()

        roles = inputs.get('nsync_role', None)
        if roles is None:
            roles = torch.full((bs,), -1, dtype=torch.int64, device=device)
        else:
            roles = roles.to(device=device, dtype=torch.int64)

        features = (noisy_latents, noisy_audio_latents, valid_audio, t, roles)
        if self.i2v:
            features = (*features, i2v_first_frame_latents)
        if self.self_flow_enabled:
            features = (
                *features,
                teacher_latents,
                teacher_audio_latents,
                s,
                video_token_mask,
                audio_token_mask,
            )
        features = (*features, *conds, *unconds)
        return features, (target, audio_target, self_flow_weight, mask)

    def get_loss_fn(self):
        @torch.autocast('cuda', enabled=False)
        def single_loss(output, target, mask=None):
            output = output.to(torch.float32)
            target = target.to(output.device, torch.float32)
            if 'huber_delta' in self.config:
                loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
            elif 'smooth_l1_beta' in self.config:
                loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
            else:
                loss = F.mse_loss(output, target, reduction='none')
            # empty tensor means no masking
            if mask is not None and mask.numel() > 0:
                mask = mask.to(output.device, torch.float32)
                loss *= mask
            return loss

        def loss_fn(outputs, label):
            if self.self_flow_enabled:
                output, audio_output, representation_loss = outputs
            else:
                output, audio_output = outputs
                representation_loss = None
            target, audio_target, self_flow_weight, mask = label
            video_loss = single_loss(output, target, mask=mask)
            # audio_target may be padded to match another example's audio length from the same
            # gradient-accumulation group (see MinimaxH3Pipeline.prepare_inputs); audio_output's
            # length is authoritative since the model derives it from this example's own valid_audio flag.
            audio_target = audio_target[..., :audio_output.shape[-1]]
            audio_loss = single_loss(audio_output, audio_target)
            # Make each token count the same by default, while allowing the two
            # modalities to be reweighted for Self-Flow experiments.
            video_tokens = video_loss.numel()
            audio_tokens = audio_loss.numel()
            total_tokens = video_tokens + audio_tokens
            video_weight = float(self.self_flow_config.get('video_loss_weight', 1.0))
            audio_weight = float(self.self_flow_config.get('audio_loss_weight', 1.0))
            video_loss = video_loss.mean() * video_tokens / total_tokens * video_weight
            audio_loss = audio_loss.mean() * audio_tokens / total_tokens * audio_weight
            loss = video_loss
            if audio_tokens > 0:  # avoid NaN for no audio
                loss = loss + audio_loss
            if representation_loss is not None:
                loss = loss + self.self_flow_gamma * representation_loss * self_flow_weight.float().mean()
            return loss
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.blocks = None
        diffusion_model.to('cuda')
        diffusion_model.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


_H3_BRANCH_SIZE = 7


def _mark_no_backward(values):
    for value in values:
        if torch.is_tensor(value):
            value.no_backward = True
    return values


class InitialLayer(nn.Module):
    def __init__(self, model, pipeline):
        super().__init__()
        if model.use_adaln_curves:
            self.adaln_t_table = model.adaln_t_table
        else:
            self.time_embedder = model.time_embedder
        self.audio_patch_proj = model.audio_patch_proj
        self.video_patch_proj = model.video_patch_proj
        self.condition_proj = model.condition_proj
        self.rope = model.rope
        self.token_refiner = model.token_refiner
        self.model = [model]
        self.pipeline = [pipeline]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @property
    def training_pipeline(self):
        return self.pipeline[0]

    def _embed_timesteps(self, values, device, dtype):
        t_vals = torch.tensor(values, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        return self.time_embedder(t_vals).to(dtype)

    @staticmethod
    def _append_mask_runs(mod_segments, output_segments, start, stop, mask,
                          false_row, true_row, modality_tag, output_kind):
        for absolute_start, absolute_stop, row in mask_to_runs(
            mask, start, stop, false_row, true_row
        ):
            mod_segments.append((absolute_start, absolute_stop, row * 3 + modality_tag))
            output_segments.append((output_kind, absolute_start, absolute_stop, row))

    @staticmethod
    def _append_text_tag_runs(mod_segments, start, stop, timestep_row, token_tags):
        if token_tags is None:
            mod_segments.append((start, stop, timestep_row * 3 + 1))
            return

        tags = token_tags.reshape(-1).tolist()
        if len(tags) != stop - start:
            raise RuntimeError(
                'MiniMax H3 text token tags did not match the padded text sequence: '
                f'{len(tags)} tags for {stop - start} tokens'
            )
        run_start = 0
        for index in range(1, len(tags) + 1):
            if index == len(tags) or tags[index] != tags[run_start]:
                tag = int(tags[run_start])
                if tag not in (0, 1):
                    raise RuntimeError(f'Unsupported MiniMax H3 text modality tag: {tag}')
                mod_segments.append((
                    start + run_start,
                    start + index,
                    timestep_row * 3 + tag,
                ))
                run_start = index

    def make_packed_sequence(self, video_x, audio_x, valid_audio, t, context, context_mask,
                             mode='standard', s=None, video_token_mask=None, audio_token_mask=None,
                             first_frame_latents=None, text_token_tags=None):
        assert video_x.shape[0] == 1
        if not valid_audio.item():
            audio_x = torch.empty([1, 32, 2, 0], device=video_x.device)

        transformer_options = {}
        device = video_x.device
        dtype = context.dtype
        latent_t, lat_h, lat_w = video_x.shape[2:]
        audio_t = audio_x.shape[-1]
        if audio_token_mask is not None:
            audio_token_mask = audio_token_mask[..., :audio_t]
        text_len = context.shape[1]
        has_visual_condition = first_frame_latents is not None and first_frame_latents.numel() > 0
        keyframes = [{'resolved_frame_index': 0}] if has_visual_condition else None
        layout = PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t, keyframes=keyframes
        )

        shift_v = float(transformer_options.get('minimax_h3_sigma_shift_video', self.sigma_shift_video))
        shift_a = float(transformer_options.get('minimax_h3_sigma_shift_audio', self.sigma_shift_audio))
        sigma_v = t.flatten()[0].float().clamp(min=1e-6)
        video_time = float(1.0 - sigma_v)
        audio_time = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
        visual_cond_time = max(
            video_time, self.training_pipeline.i2v_visual_cond_timestep
        )

        mod_segments = []
        output_segments = []
        if mode == 'standard':
            # Keep row count/order stable across a batch even when two numeric
            # timestep values happen to be equal.
            timestep_values = [video_time, audio_time]
            kind_rows = {
                'text': 0,
                'cond': 2,
                'video': 0,
                'audio': 1,
            }
            if has_visual_condition:
                timestep_values.append(visual_cond_time)
            kind_tags = {'cond': 0, 'video': 0, 'audio': 2}
            for start, stop, kind in layout.segments:
                row = kind_rows[kind]
                if kind == 'text':
                    self._append_text_tag_runs(
                        mod_segments, start, stop, row, text_token_tags
                    )
                else:
                    mod_segments.append((start, stop, row * 3 + kind_tags[kind]))
                if kind == 'video':
                    output_segments.append((0, start, stop, row))
                elif kind == 'audio':
                    output_segments.append((1, start, stop, row))
        elif mode == 'mixed':
            sigma_s = s.flatten()[0].float().clamp(min=1e-6)
            timestep_values = [
                video_time,
                float(1.0 - sigma_s),
                audio_time,
                float(1.0 - time_shift_sigma(sigma_s, shift_v, shift_a)),
            ]
            visual_cond_row = len(timestep_values)
            if has_visual_condition:
                timestep_values.append(visual_cond_time)
            for start, stop, kind in layout.segments:
                if kind == 'text':
                    self._append_text_tag_runs(
                        mod_segments, start, stop, 0, text_token_tags
                    )
                elif kind == 'cond':
                    mod_segments.append((start, stop, visual_cond_row * 3))
                elif kind == 'video':
                    self._append_mask_runs(
                        mod_segments, output_segments, start, stop, video_token_mask,
                        false_row=0, true_row=1, modality_tag=0, output_kind=0,
                    )
                elif kind == 'audio':
                    self._append_mask_runs(
                        mod_segments, output_segments, start, stop, audio_token_mask,
                        false_row=2, true_row=3, modality_tag=2, output_kind=1,
                    )
                else:
                    raise RuntimeError(f'Unsupported Self-Flow packed segment kind: {kind}')
        elif mode == 'teacher':
            sigma_min = torch.minimum(sigma_v, s.flatten()[0].float().clamp(min=1e-6))
            teacher_video_time = float(1.0 - sigma_min)
            teacher_visual_cond_time = max(
                teacher_video_time, self.training_pipeline.i2v_visual_cond_timestep
            )
            timestep_values = [
                teacher_video_time,
                float(1.0 - time_shift_sigma(sigma_min, shift_v, shift_a)),
            ]
            visual_cond_row = len(timestep_values)
            if has_visual_condition:
                timestep_values.append(teacher_visual_cond_time)
            for start, stop, kind in layout.segments:
                if kind == 'text':
                    self._append_text_tag_runs(
                        mod_segments, start, stop, 0, text_token_tags
                    )
                elif kind == 'cond':
                    mod_segments.append((start, stop, visual_cond_row * 3))
                elif kind == 'video':
                    mod_segments.append((start, stop, 0))
                    output_segments.append((0, start, stop, 0))
                elif kind == 'audio':
                    mod_segments.append((start, stop, 1 * 3 + 2))
                    output_segments.append((1, start, stop, 1))
                else:
                    raise RuntimeError(f'Unsupported Self-Flow packed segment kind: {kind}')
        else:
            raise ValueError(f'Unknown MiniMax H3 packing mode: {mode}')

        img_update = layout.img_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))

        all_video_rows = video_rows
        if has_visual_condition:
            cond_video_rows = patchify_video(
                first_frame_latents.to(device=device, dtype=torch.float32), self.patch_size
            )
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(audio_rows).to(dtype)
        text_states = context
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(
                self.condition_proj(text_states), transformer_options=transformer_options
            )
        text_states = text_states[0]

        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        attention_mask = torch.ones((layout.seq_len,), dtype=context_mask.dtype, device=device)
        video_offset = audio_offset = 0
        for start, stop, kind in layout.segments:
            length = stop - start
            if kind == 'text':
                h[start:stop] = text_states
                attention_mask[start:stop] = context_mask
            elif kind in ('cond', 'ref_img', 'video'):
                h[start:stop] = video_embed[video_offset:video_offset + length]
                video_offset += length
            else:
                h[start:stop] = audio_embed[audio_offset:audio_offset + length]
                audio_offset += length

        t_emb = self._embed_timesteps(timestep_values, device, dtype)
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
        mod_segments = torch.tensor(mod_segments, dtype=torch.int32, device=device)
        output_segments = torch.tensor(output_segments, dtype=torch.int32, device=device)
        extra_ints = torch.tensor([latent_t, lat_h, lat_w], dtype=torch.int32, device=device)
        return h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, output_segments

    def make_layer_inputs(self, video_x, audio_x, valid_audio, t, context, context_mask,
                          mode='standard', s=None, video_token_mask=None, audio_token_mask=None,
                          first_frame_latents=None, text_token_tags=None):
        first_mod_segments = first_output_segments = None
        h_list, attention_mask_list, t_emb_list = [], [], []
        batch_size = video_x.shape[0]
        for index in range(batch_size):
            packed = self.make_packed_sequence(
                video_x[index:index + 1],
                audio_x[index:index + 1],
                valid_audio[index:index + 1],
                t[index:index + 1],
                context[index:index + 1],
                context_mask[index:index + 1],
                mode=mode,
                s=None if s is None else s[index:index + 1],
                video_token_mask=None if video_token_mask is None else video_token_mask[index:index + 1],
                audio_token_mask=None if audio_token_mask is None else audio_token_mask[index:index + 1],
                first_frame_latents=(
                    None if first_frame_latents is None else first_frame_latents[index:index + 1]
                ),
                text_token_tags=(
                    None if text_token_tags is None else text_token_tags[index:index + 1]
                ),
            )
            h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, output_segments = packed
            if first_mod_segments is None:
                first_mod_segments = mod_segments
                first_output_segments = output_segments
            elif not (
                torch.equal(mod_segments, first_mod_segments)
                and torch.equal(output_segments, first_output_segments)
            ):
                raise RuntimeError(
                    'MiniMax H3 requires shared packed segment boundaries within a physical microbatch. '
                    'Use micro_batch_size_per_gpu=1 when mixing videos with and without audio.'
                )
            h_list.append(h)
            attention_mask_list.append(attention_mask)
            t_emb_list.append(t_emb)

        h = torch.stack(h_list)
        attention_mask = torch.stack(attention_mask_list)[:, None, None, :]
        t_emb = torch.stack(t_emb_list)
        outputs = make_contiguous(
            h, attention_mask, t_emb, first_mod_segments, rope_freqs, extra_ints, first_output_segments
        )
        for item in outputs:
            if torch.is_tensor(item) and torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        video_x, audio_x, valid_audio, t, roles, *variable = inputs
        pipeline = self.training_pipeline
        if pipeline.i2v:
            first_frame_latents, *variable = variable
        else:
            first_frame_latents = None
        if pipeline.self_flow_enabled:
            teacher_video_x, teacher_audio_x, s, video_token_mask, audio_token_mask, *variable = variable
            packing_mode = 'mixed'
        else:
            teacher_video_x = teacher_audio_x = s = video_token_mask = audio_token_mask = None
            packing_mode = 'standard'

        context, context_mask, *variable = variable
        if pipeline.i2v:
            text_token_tags, *variable = variable
        else:
            text_token_tags = None
        if pipeline.cfg > 1:
            context_uncond, context_mask_uncond, *variable = variable
            if pipeline.i2v:
                uncond_text_token_tags, *variable = variable
            else:
                uncond_text_token_tags = None
        else:
            context_uncond = context_mask_uncond = None
            uncond_text_token_tags = None
        if variable:
            raise RuntimeError(f'Unexpected MiniMax H3 pipeline inputs: {len(variable)} extra tensors')

        pipeline.set_self_flow_adapter(getattr(pipeline, 'self_flow_student_adapter', 'default'))
        primary = self.make_layer_inputs(
            video_x, audio_x, valid_audio, t, context, context_mask,
            mode=packing_mode, s=s, video_token_mask=video_token_mask, audio_token_mask=audio_token_mask,
            first_frame_latents=first_frame_latents, text_token_tags=text_token_tags,
        )
        if pipeline.nsync_enabled:
            primary = (pipeline.nsync_controller.tag_output(primary[0], roles), *primary[1:])

        outputs = primary
        if pipeline.self_flow_enabled:
            pipeline.set_self_flow_adapter(pipeline.self_flow_teacher_adapter)
            with torch.no_grad():
                teacher = self.make_layer_inputs(
                    teacher_video_x, teacher_audio_x, valid_audio, t, context, context_mask,
                    mode='teacher', s=s,
                    first_frame_latents=first_frame_latents, text_token_tags=text_token_tags,
                )
            outputs = (*outputs, *_mark_no_backward(teacher))
            pipeline.set_self_flow_adapter(pipeline.self_flow_student_adapter)

        if pipeline.cfg > 1:
            with torch.no_grad():
                uncond = self.make_layer_inputs(
                    video_x, audio_x, valid_audio, t, context_uncond, context_mask_uncond,
                    mode=packing_mode, s=s, video_token_mask=video_token_mask, audio_token_mask=audio_token_mask,
                    first_frame_latents=first_frame_latents,
                    text_token_tags=uncond_text_token_tags,
                )
            outputs = (*outputs, *_mark_no_backward(uncond))

        student_feature = primary[0].new_empty((0,))
        teacher_feature = primary[0].new_empty((0,))
        student_feature.no_backward = True
        teacher_feature.no_backward = True
        return (*outputs, roles, student_feature, teacher_feature)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader, pipeline):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader
        self.pipeline = [pipeline]

    @property
    def training_pipeline(self):
        return self.pipeline[0]

    @staticmethod
    def _take_branch(inputs, offset):
        return tuple(inputs[offset:offset + _H3_BRANCH_SIZE]), offset + _H3_BRANCH_SIZE

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        pipeline = self.training_pipeline
        primary, offset = self._take_branch(inputs, 0)
        teacher = None
        if pipeline.self_flow_enabled:
            teacher, offset = self._take_branch(inputs, offset)
        uncond = None
        if pipeline.cfg > 1:
            uncond, offset = self._take_branch(inputs, offset)
        roles, student_feature, teacher_feature = inputs[offset:offset + 3]

        h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, output_segments = primary
        self.offloader.wait_for_block(self.block_idx)

        pipeline.set_self_flow_adapter(getattr(pipeline, 'self_flow_student_adapter', 'default'))
        h = self.layer(h, t_emb, mod_segments.tolist(), rope_freqs, attention_mask=attention_mask)
        if pipeline.nsync_enabled:
            h = pipeline.nsync_controller.tag_output(h, roles)
        if pipeline.self_flow_enabled and self.block_idx == pipeline.self_flow_student_layer:
            student_feature = h
        primary = (h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, output_segments)

        if teacher is not None:
            h_teacher, attention_mask_teacher, t_emb_teacher, mod_segments_teacher, rope_freqs_teacher, extra_ints_teacher, output_segments_teacher = teacher
            if h_teacher is not None and h_teacher.numel() > 0 and self.block_idx <= pipeline.self_flow_teacher_layer:
                pipeline.set_self_flow_adapter(pipeline.self_flow_teacher_adapter)
                with torch.no_grad():
                    h_teacher = self.layer(
                        h_teacher,
                        t_emb_teacher,
                        mod_segments_teacher.tolist(),
                        rope_freqs_teacher,
                        attention_mask=attention_mask_teacher,
                    )
                if self.block_idx == pipeline.self_flow_teacher_layer:
                    teacher_feature = h_teacher.detach()
                    teacher_feature.no_backward = True
                    h_teacher = h_teacher.new_empty((0,))
                teacher = (
                    h_teacher, attention_mask_teacher, t_emb_teacher, mod_segments_teacher,
                    rope_freqs_teacher, extra_ints_teacher, output_segments_teacher,
                )
                teacher = _mark_no_backward(teacher)

        if uncond is not None:
            h_uncond, attention_mask_uncond, t_emb_uncond, mod_segments_uncond, rope_freqs_uncond, extra_ints_uncond, output_segments_uncond = uncond
            if h_uncond is not None:
                pipeline.set_self_flow_adapter(getattr(pipeline, 'self_flow_student_adapter', 'default'))
                with torch.no_grad():
                    h_uncond = self.layer(
                        h_uncond,
                        t_emb_uncond,
                        mod_segments_uncond.tolist(),
                        rope_freqs_uncond,
                        attention_mask=attention_mask_uncond,
                    )
                uncond = (
                    h_uncond, attention_mask_uncond, t_emb_uncond, mod_segments_uncond,
                    rope_freqs_uncond, extra_ints_uncond, output_segments_uncond,
                )
                uncond = _mark_no_backward(uncond)

        pipeline.set_self_flow_adapter(getattr(pipeline, 'self_flow_student_adapter', 'default'))
        self.offloader.submit_move_blocks_forward(self.block_idx)
        if torch.is_tensor(student_feature) and student_feature.numel() == 0:
            student_feature.no_backward = True
        if (
            pipeline.self_flow_enabled
            and self.block_idx <= pipeline.self_flow_teacher_layer
            and torch.is_tensor(teacher_feature)
        ):
            teacher_feature.no_backward = True
        outputs = primary
        if teacher is not None:
            outputs = (*outputs, *teacher)
        if uncond is not None:
            outputs = (*outputs, *uncond)
        return make_contiguous(*outputs, roles, student_feature, teacher_feature)


class SelfFlowLossLayer(nn.Module):
    """Collapse the two captured H3 features into the train-time scalar loss."""

    def __init__(self, model, pipeline):
        super().__init__()
        self.pipeline = [pipeline]
        projection_dim = int(pipeline.self_flow_config.get('projection_dim', 1024))
        if projection_dim <= 0:
            raise ValueError(f'Self-Flow projection_dim must be positive, got {projection_dim}')
        reference = next(model.final_layer.parameters())
        projection_dtype = pipeline.config['adapter']['dtype']
        self.projection = nn.Sequential(
            nn.Linear(model.hidden_size, projection_dim, bias=False, device=reference.device, dtype=projection_dtype),
            nn.SiLU(),
            nn.Linear(projection_dim, model.hidden_size, bias=False, device=reference.device, dtype=projection_dtype),
        )
        for name, parameter in self.projection.named_parameters():
            parameter.original_name = f'self_flow_projection.{name}'
            parameter.skip_adapter_save = True

    @property
    def training_pipeline(self):
        return self.pipeline[0]

    @staticmethod
    def _target_features(hidden, output_segments):
        pieces = [hidden[:, start:stop] for _, start, stop, _ in output_segments.tolist()]
        return torch.cat(pieces, dim=1)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        pipeline = self.training_pipeline
        offset = _H3_BRANCH_SIZE
        teacher = tuple(inputs[offset:offset + _H3_BRANCH_SIZE])
        offset += _H3_BRANCH_SIZE
        if pipeline.cfg > 1:
            offset += _H3_BRANCH_SIZE
        roles, student_feature, teacher_feature = inputs[offset:offset + 3]

        if student_feature is None or teacher_feature is None or student_feature.numel() == 0 or teacher_feature.numel() == 0:
            raise RuntimeError('Self-Flow did not capture its configured student and teacher layers')
        primary_output_segments = inputs[_H3_BRANCH_SIZE - 1]
        teacher_output_segments = teacher[-1]
        student_targets = self._target_features(student_feature, primary_output_segments)
        teacher_targets = self._target_features(teacher_feature, teacher_output_segments)
        representation_loss = representation_cosine_loss(self.projection(student_targets), teacher_targets)

        empty_feature = student_feature.new_empty((0,))
        empty_feature.no_backward = True
        return make_contiguous(*inputs[:offset], roles, empty_feature, representation_loss)


class FinalLayer(nn.Module):
    def __init__(self, model, cfg, pipeline):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]
        self.pipeline = [pipeline]
        self.cfg = cfg

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @property
    def training_pipeline(self):
        return self.pipeline[0]

    @staticmethod
    def _take_branch(inputs, offset):
        return tuple(inputs[offset:offset + _H3_BRANCH_SIZE]), offset + _H3_BRANCH_SIZE

    @staticmethod
    def _segments_by_kind(output_segments, kind):
        return [row[1:] for row in output_segments.tolist() if row[0] == kind]

    def _decode_branch(self, branch):
        h, _, t_emb, _, _, extra_ints, output_segments = branch
        video_segments = self._segments_by_kind(output_segments, 0)
        audio_segments = self._segments_by_kind(output_segments, 1)
        latent_t, lat_h, lat_w = extra_ints
        video_rows, audio_rows = self.final_layer(h, t_emb, video_segments, audio_segments)
        video_out = unpatchify_video(
            video_rows, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size
        )
        return video_out, unpack_audio(audio_rows)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        pipeline = self.training_pipeline
        primary, offset = self._take_branch(inputs, 0)
        teacher = None
        if pipeline.self_flow_enabled:
            teacher, offset = self._take_branch(inputs, offset)
        uncond = None
        if pipeline.cfg > 1:
            uncond, offset = self._take_branch(inputs, offset)
        _, student_feature, representation_loss = inputs[offset:offset + 3]

        pipeline.set_self_flow_adapter(getattr(pipeline, 'self_flow_student_adapter', 'default'))
        video_out, audio_out = self._decode_branch(primary)

        if uncond is not None:
            assert self.cfg > 1
            with torch.no_grad():
                video_out_uncond, audio_out_uncond = self._decode_branch(uncond)
            video_out = (video_out + (self.cfg - 1) * video_out_uncond) / self.cfg
            audio_out = (audio_out + (self.cfg - 1) * audio_out_uncond) / self.cfg

        outputs = (-video_out, -audio_out)
        if pipeline.self_flow_enabled:
            if representation_loss is None or representation_loss.numel() != 1 or student_feature.numel() != 0:
                raise RuntimeError('Self-Flow representation loss was not produced after the teacher layer')
            outputs = (*outputs, representation_loss)

        # Audio velocity is intentionally not schedule-slope-scaled during
        # training; that MiniMax transform is inference-only.
        return outputs


# class Wrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, inputs):
#         video_x, timestep, context, context_mask = inputs
#         audio_x = torch.empty([1, 32, 2, 0], device=video_x.device)
#         out = self.model((video_x, audio_x), timestep*1000, context=context)
#         return out[0]
