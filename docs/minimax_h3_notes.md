# A collection of notes on Minimax H3 implementation and training

## Distillation
**Any standard training gradually undistills the model**. You may need to use CFG for inference. How much CFG you will need depends on the size of the dataset and the amount of training.
### Solutions
- CFG-augmented training (see the example TOML config). This modifies the model's output with its own unconditional prediction when fitting the target, in order to cause the model output to have built-in CFG. This preserves the guidance distillation that the model comes with. You can think of this as "baking in" as CFG value using the model's own uncond prediction during training time, even though you are still fitting the standard flow-matching target. This technique works very well and is recommended.
- Training adapter. Ostris has a [training adapter](https://huggingface.co/ostris/minimax_h3_training_adapter/tree/main) for the model, but I haven't tested this myself.

## General notes
The AdaLN weights are not trained with LoRA. This makes the LoRA compatible with both the full and pruned checkpoints, regardless of which one you trained with.

The dataset caching phase should use ComfyUI dynamic VRAM, meaning the text encoder can be larger than available VRAM. E.g. the int8 convrot TE is 26GB, but can compute text embeddings on a 24GB GPU.

If dataset caching occurs (meaning it loads the VAE / TE), the text encoder memory somehow isn't completely freed from RAM afterwards. The TE is large, and this might OOM you. Just relaunch the training script, since the dataset is now cached. Or do it in 2 phases from the beginning:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --num_gpus=1 train.py --config your_config.toml --trust_cache --cache_only
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --num_gpus=1 train.py --config your_config.toml --trust_cache
```
`--trust_cache` loads the cache faster if it exists, but you won't pick up changes to the underlying data files. If you don't change your dataset files, this is always safe to pass.

You can train LoRAs directly on top of quantized weights, like int8 convrot, and you probably should since the model is large.

Block swapping will be needed for 24GB VRAM. `blocks_to_swap=48` is the maximum allowed for this model. `activation_checkpointing = 'unsloth'` also saves a lot of VRAM for minimal overhead.

Audio is automatically trained if your videos have it.

Training on images works fine. I saw some reports that it won't work; no, it does. The VAE is "asymmetric": the encoder can encode 1 image frame to a valid single latent frame, but the decoder can't decode that single latent frame very well. So the latent space for images is fine, and the model can learn from it. But it does gradually degrade the video/motion understanding in the model if you exclusively train on images (same as any other video model). Joint image/video training will work better.

I don't know what timestep distribution and shift value is optimal. `timestep_sample_method='uniform'` and `shift=12` matches the default inference schedule. Lowering shift to something like 8, or even lower, could help learn details at the expense of large-scale structure and motion.

## I2V training

Set `model.mode = 'i2v'` to train H3's first-frame-to-video path. For every video clip, the trainer:

- independently VAE-encodes the preprocessed first frame as a non-denoised FL2VA keyframe;
- includes that same frame in Qwen's multimodal presentation and preserves its video token tags; and
- reuses the identical visual condition in the normal, Self-Flow teacher, and CFG-augmented branches.

`i2v_visual_cond_timestep = 0.999` matches ComfyUI's visual conditioning timestep/noise augmentation and normally should not be changed. `video_clip_mode` decides which source clip supplies the first frame; `single_beginning` is the usual I2V choice, while `single_middle` anchors a middle clip. Images and videos may be mixed freely: image samples skip both the FL2VA keyframe and Qwen vision condition, while video samples use both. [minimax_h3_i2v_dataset.toml](../examples/minimax_h3_i2v_dataset.toml) shows frame bucket `1` together with video buckets.

I2V text caching is slower and larger than T2V caching because Qwen processes the first frame and stores a matching per-example unconditional presentation for CFG/classifier-free caption dropout. I2V and T2V use separate cache directories.

Videos without an audio track are supported. They still receive normal text/video and, in I2V mode, first-frame training, but H3 omits their audio tokens and they contribute no audio loss. Videos with audio train both video and audio. Keep `micro_batch_size_per_gpu = 1` when the dataset mixes the two: audio-present and audio-absent samples have different packed layouts and cannot share one physical H3 microbatch. Gradient accumulation may still be used to increase the logical batch size.

Last-frame and general reference-to-video training are not implemented.

## Small unbucketed datasets

When a small dataset cannot populate useful aspect-ratio and frame buckets, use [`minimax_h3_unbucketed_dataset.toml`](../examples/minimax_h3_unbucketed_dataset.toml). Set `unbucketed = true` and exactly one `resolutions` entry at the dataset root, then remove the aspect-ratio, frame-bucket, and explicit size-bucket settings. Each sample is scaled to approximately `resolution²` pixels while retaining its native aspect ratio and usable frame length. Samples are shuffled and trained one at a time, so a sparse shape cannot cause that sample to be dropped. H3's required alignment still applies: spatial dimensions are rounded to the nearest multiple of 32 and frame count is rounded down to a multiple of 17. Videos shorter than 17 frames after conversion to H3's 24 fps are skipped rather than being treated as still images.

The physical video and image microbatch sizes must both be `1`; the same applies to eval microbatch sizes when an eval dataset is unbucketed. `gradient_accumulation_steps` can still be greater than one because each sample is prepared independently before accumulation. The only data omitted is the final incomplete accumulated batch across all data-parallel ranks. Increase `num_repeats` if the complete dataset is smaller than that logical batch. NSYNC remains bucket-based and cannot be combined with this mode.

## NSYNC LoRA training

The implementation follows the update in the [NSYNC paper](https://arxiv.org/abs/2511.01517) rather than the typo in one path of its released training code. For every logical microbatch it computes the LoRA gradients for a positive batch, its caption-matched generated-negative batch, and an independently sampled positive anchor batch, then applies

```
g = g_pos - proj(g_pos onto g_neg) + proj(g_pos onto g_anchor)
```

The three H3 passes are sequential, so their activation graphs do not have to coexist. DeepSpeed's physical gradient-accumulation count is multiplied by three automatically; the configured `gradient_accumulation_steps` and reported global batch size remain logical positive-batch values. Gradient clipping and the optimizer see only the projected gradient.

Use [minimax_h3_nsync_self_flow_dataset.toml](../examples/minimax_h3_nsync_self_flow_dataset.toml) as the dataset template. Important data rules:

- Every directory must set `nsync_role = 'positive'` or `'negative'`, and matching directories use the same `nsync_pair`.
- `nsync_pair` is also the group name. On a group's positive directory, optional `nsync_anchor_pairs = ['group_b', 'group_c']` draws anchors from the pooled positive examples in those groups. Every listed group must use matching bucket settings. If omitted, anchors come from the group's own positives, preserving the original behavior.
- Positive and negative media are paired by filename stem and must land in the same resolution/frame bucket. For video, generate the negative with the same frame count and dimensions.
- The negative uses the exact positive caption. `caption_path` lets the negative directory reuse positive `.txt` files or `captions.json`.
- The generated negative should preserve the caption's subject/content while omitting the target concept or style. The trainer does not synthesize negatives for you.
- NSYNC currently requires data-parallel world size 1, `uncond_fraction = 0`, and no `optimizer.gradient_release`. NSYNC by itself can use pipeline parallelism.

Anchor selection is deterministic, not resampled on every access. At dataset initialization, each size bucket builds an independently shuffled anchor order from the configured groups and reuses that order in every epoch. A repeated sample or a media item with multiple captions can occupy multiple dataset positions and therefore have more than one fixed anchor.

### Generating N-Sync negatives with local MiniMax H3 in ComfyUI

Use [`tools/generate_minimax_h3_nsync_negatives.py`](../tools/generate_minimax_h3_nsync_negatives.py) to create the negative directory through a locally running MiniMax H3 ComfyUI workflow. It rejects hosted MiniMax API nodes, removes target phrases from generation prompts, preserves positive filename stems, and normalizes dimensions, frame duration, media type, and audio presence. With `--mode i2v`, every video job is conditioned on frame zero of its matching positive; image jobs remain unconditioned.

The [complete local ComfyUI NSYNC negative-generation guide](minimax_h3_nsync_negative_generation.md) covers workflow construction, caption handling, dry runs, resuming, all important options, dataset configuration, and troubleshooting. The basic command is:

```bash
python tools/generate_minimax_h3_nsync_negatives.py \
  /path/to/target_positive_media \
  /path/to/generated_negative_media \
  --workflow /path/to/minimax_h3_t2va_api.json \
  --remove-text 'your trigger phrase' \
  --dry-run
```

Remove `--dry-run` after checking every cleaned prompt. Add `--mode i2v` when using `model.mode = 'i2v'`, and start from [the ready I2V + NSYNC configs](../examples/minimax_h3_i2v_nsync.toml). Keep the negative dataset's `caption_path` pointed at the positive directory so training uses the exact positive caption, not the cleaned generation prompt. During NSYNC, video examples in the positive, negative, and anchor role batches use their own cached first-frame latent and Qwen vision tokens; image examples in those roles remain text-only.

## Self-Flow LoRA training

[Self-Flow](https://arxiv.org/abs/2603.06507) trains a student on a sequence where randomly selected tokens use an independent timestep `s` and all other tokens use `t`. An EMA teacher receives the homogeneous, cleaner state at `min(t, s)`. A train-only projection MLP aligns the student representation near 30% depth with the teacher representation near 70% depth using negative cosine similarity. The normal flow-matching loss remains active.

For H3 this is implemented separately for its packed modalities:

- video/image patch tokens use H3's per-segment video AdaLN timestep;
- audio tokens use the corresponding timestep after H3's video-to-audio schedule mapping;
- default mask ratios are 0.25 for images, 0.10 for video, and 0.50 for audio;
- the EMA decay defaults to 0.9999 and representation weight `gamma` to 0.8;
- the optional `high_noise_fraction = 0.05` / `high_noise_range = [0.95, 1.0]` adapts the paper's low-SNR video sampling ablation;
- the EMA is only a second LoRA adapter because the base H3 weights are frozen. It is forwarded only through the configured teacher layer and defaults to FP32 so decay `0.9999` does not quantize away small updates (`ema_dtype = 'bfloat16'` is available when memory is tighter).

H3's batched AdaLN requires one token-mask layout inside each physical microbatch, so samples in that microbatch share the Bernoulli mask; a new independent mask is sampled for every physical microbatch. Video training normally uses `micro_batch_size_per_gpu = 1`, which exactly recovers per-example masking.

The EMA adapter and projection head are stored in resumable DeepSpeed checkpoints. Following the paper's evaluation setup, the final inference LoRA writes the EMA adapter values under the normal LoRA keys; the online student and projection head are not added as extra inference adapters. Self-Flow currently requires `pipeline_stages = 1`; block swapping remains supported. `torch.compile` is disabled automatically for either method because adapter selection and NSYNC's backward hooks are dynamic.

Black Forest Labs says [FLUX.3 is built on Self-Flow](https://bfl.ai/blog/flux-3), with joint image, video, and audio pretraining. The public FLUX.3 material does not provide a complete reproducible optimizer/data recipe, so this implementation adapts the concrete mechanisms and ablations published in Self-Flow rather than inventing unpublished FLUX.3 details.

## Combining NSYNC and Self-Flow

The two methods can be enabled independently, or enable both sections as shown in [minimax_h3_nsync_self_flow.toml](../examples/minimax_h3_nsync_self_flow.toml). Each positive, negative, and anchor role gets the full generation plus Self-Flow representation objective. NSYNC surgery is then applied only to the student LoRA gradient vector, while the train-only projection head learns from the ordinary average across all three roles. One optimizer update and one EMA update occur after the combined logical step.
