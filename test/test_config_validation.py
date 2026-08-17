import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.config_validation import ConfigValidationError, load_and_validate_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigValidationTest(unittest.TestCase):
    def _write_valid_config(self, root: Path) -> Path:
        media_dir = root / 'media'
        model_dir = root / 'model'
        output_dir = root / 'output'
        media_dir.mkdir()
        (media_dir / 'example.png').write_bytes(b'not decoded during preflight')
        (media_dir / 'example.txt').write_text('an example caption')
        model_dir.mkdir()
        (model_dir / 'weights.bin').write_bytes(b'placeholder')
        dataset_path = root / 'dataset.toml'
        dataset_path.write_text(
            f'''\
resolutions = [512]

[[directory]]
path = {str(media_dir)!r}
num_repeats = 1
'''
        )
        config_path = root / 'train.toml'
        config_path.write_text(
            f'''\
output_dir = {str(output_dir)!r}
dataset = {str(dataset_path)!r}
epochs = 10
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 1
save_every_n_epochs = 1
trust_cache = true

[model]
type = 'wan'
ckpt_path = {str(model_dir)!r}
dtype = 'bfloat16'

[optimizer]
type = 'adamw'
lr = 1e-4
'''
        )
        return config_path

    def _convert_to_minimax_h3(self, config_path: Path, model_dir: Path) -> None:
        config_path.write_text(
            config_path.read_text()
            .replace("type = 'wan'", "type = 'minimax_h3'")
            .replace(
                f"ckpt_path = {str(model_dir)!r}",
                f"diffusion_model = {str(model_dir)!r}\n"
                f"vae = {str(model_dir)!r}\n"
                f"audio_vae = {str(model_dir)!r}\n"
                f"text_encoders = [{{path = {str(model_dir)!r}, type = 'minimax'}}]",
            )
        )

    def test_valid_config_loads_main_and_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_valid_config(Path(tmp))
            config, datasets = load_and_validate_config(config_path, world_size=1)

        self.assertEqual(config['model']['type'], 'wan')
        self.assertEqual(len(datasets), 1)

    def test_minimax_h3_unbucketed_dataset_accepts_native_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            dataset_path = root / 'dataset.toml'
            media_dir = root / 'media'
            dataset_path.write_text(
                f'''\
unbucketed = true
resolutions = [512]
num_repeats = 2

[[directory]]
path = {str(media_dir)!r}
'''
            )
            self._convert_to_minimax_h3(config_path, root / 'model')

            config, datasets = load_and_validate_config(config_path)

        dataset = next(iter(datasets.values()))
        self.assertEqual(config['model']['type'], 'minimax_h3')
        self.assertTrue(dataset['unbucketed'])
        self.assertEqual(dataset['resolutions'], [512])

    def test_unbucketed_dataset_requires_physical_batch_size_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            dataset_path = root / 'dataset.toml'
            media_dir = root / 'media'
            dataset_path.write_text(
                f'''\
unbucketed = true
resolutions = [512]

[[directory]]
path = {str(media_dir)!r}
'''
            )
            self._convert_to_minimax_h3(config_path, root / 'model')
            config_path.write_text(
                config_path.read_text()
                .replace('micro_batch_size_per_gpu = 1', 'micro_batch_size_per_gpu = 2')
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        self.assertIn('unbucketed requires micro_batch_size_per_gpu=1', str(caught.exception))
        self.assertIn('unbucketed requires image_micro_batch_size_per_gpu=1', str(caught.exception))

    def test_unbucketed_dataset_rejects_bucket_settings_and_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            dataset_path = root / 'dataset.toml'
            media_dir = root / 'media'
            dataset_path.write_text(
                f'''\
unbucketed = true
resolutions = [512]
frame_buckets = [1, 34]

[[directory]]
path = {str(media_dir)!r}
'''
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        message = str(caught.exception)
        self.assertIn('unbucketed is currently supported only for MiniMax H3', message)
        self.assertIn('must omit bucket settings when unbucketed=true: frame_buckets', message)

    def test_unbucketed_dataset_requires_exactly_one_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            dataset_path = root / 'dataset.toml'
            media_dir = root / 'media'
            self._convert_to_minimax_h3(config_path, root / 'model')

            dataset_path.write_text(
                f'''\
unbucketed = true

[[directory]]
path = {str(media_dir)!r}
'''
            )
            with self.assertRaises(ConfigValidationError) as missing:
                load_and_validate_config(config_path)

            dataset_path.write_text(
                f'''\
unbucketed = true
resolutions = [512, 768]

[[directory]]
path = {str(media_dir)!r}
'''
            )
            with self.assertRaises(ConfigValidationError) as multiple:
                load_and_validate_config(config_path)

        self.assertIn('unbucketed requires exactly one target resolution', str(missing.exception))
        self.assertIn('unbucketed requires exactly one target resolution', str(multiple.exception))

    def test_minimax_h3_i2v_mode_and_visual_timestep_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            model_dir = root / 'model'
            minimax_config = (
                config_path.read_text()
                .replace("type = 'wan'", "type = 'minimax_h3'\nmode = 'i2v'\ni2v_visual_cond_timestep = 0.999")
                .replace(
                    f"ckpt_path = {str(model_dir)!r}",
                    f"diffusion_model = {str(model_dir)!r}\n"
                    f"vae = {str(model_dir)!r}\n"
                    f"audio_vae = {str(model_dir)!r}\n"
                    f"text_encoders = [{{path = {str(model_dir)!r}, type = 'minimax'}}]",
                )
            )
            config_path.write_text(minimax_config)
            config, _ = load_and_validate_config(config_path)
            self.assertEqual(config['model']['mode'], 'i2v')

            config_path.write_text(
                minimax_config
                .replace("mode = 'i2v'", "mode = 'reference'")
                .replace('i2v_visual_cond_timestep = 0.999', 'i2v_visual_cond_timestep = 1.1')
            )
            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        message = str(caught.exception)
        self.assertIn('model.mode must be t2v or i2v', message)
        self.assertIn('model.i2v_visual_cond_timestep must be in [0, 1]', message)

    def test_minimax_h3_i2v_bucketed_and_unbucketed_nsync_can_be_enabled_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            positive_dir = root / 'media'
            (positive_dir / 'example.png').unlink()
            (positive_dir / 'example.mp4').write_bytes(b'not decoded during preflight')
            (positive_dir / 'still.png').write_bytes(b'not decoded during preflight')
            (positive_dir / 'still.txt').write_text('a still image caption')
            negative_dir = root / 'negative'
            negative_dir.mkdir()
            (negative_dir / 'example.mp4').write_bytes(b'not decoded during preflight')
            (negative_dir / 'still.png').write_bytes(b'not decoded during preflight')
            dataset_path = root / 'dataset.toml'
            dataset_path.write_text(
                f'''\
unbucketed = true
resolutions = [512]

[[directory]]
path = {str(positive_dir)!r}
nsync_role = 'positive'
nsync_pair = 'i2v'

[[directory]]
path = {str(negative_dir)!r}
caption_path = {str(positive_dir)!r}
nsync_role = 'negative'
nsync_pair = 'i2v'
'''
            )
            model_dir = root / 'model'
            config_path.write_text(
                config_path.read_text()
                .replace("type = 'wan'", "type = 'minimax_h3'\nmode = 'i2v'")
                .replace(
                    f"ckpt_path = {str(model_dir)!r}",
                    f"diffusion_model = {str(model_dir)!r}\n"
                    f"vae = {str(model_dir)!r}\n"
                    f"audio_vae = {str(model_dir)!r}\n"
                    f"text_encoders = [{{path = {str(model_dir)!r}, type = 'minimax'}}]",
                )
                + '''\

[adapter]
type = 'lora'
rank = 8

[training_methods.nsync]
enabled = true
'''
            )

            config, datasets = load_and_validate_config(config_path, world_size=1)
            dataset_path.write_text(
                dataset_path.read_text()
                .replace('unbucketed = true\n', '')
                .replace('resolutions = [512]\n', 'resolutions = [512]\nframe_buckets = [1, 33]\n')
            )
            _, bucketed_datasets = load_and_validate_config(config_path, world_size=1)

        self.assertEqual(config['model']['mode'], 'i2v')
        self.assertTrue(config['training_methods']['nsync']['enabled'])
        self.assertEqual(len(datasets), 1)
        dataset = next(iter(datasets.values()))
        self.assertTrue(dataset['unbucketed'])
        self.assertEqual(len(dataset['directory']), 2)
        self.assertFalse(next(iter(bucketed_datasets.values())).get('unbucketed', False))

    def test_reports_multiple_errors_in_one_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'broken.toml'
            config_path.write_text(
                '''\
epochs = 0
pipeline_stages = 3
lr_scheduler = 'mystery'

[model]
type = 'not-a-model'
dtype = 'half-ish'
'''
            )
            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path, world_size=2)

        message = str(caught.exception)
        self.assertIn('output_dir', message)
        self.assertIn('epochs', message)
        self.assertIn('world size', message)
        self.assertIn('lr_scheduler', message)
        self.assertIn('model.type', message)
        self.assertIn('model.dtype', message)
        self.assertIn('optimizer', message)
        self.assertGreaterEqual(len(caught.exception.errors), 8)

    def test_validate_only_exits_before_training_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_valid_config(Path(tmp))
            result = subprocess.run(
                [sys.executable, 'train.py', '--validate_only', '--config', str(config_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Configuration is valid', result.stdout)
        self.assertNotIn('DeepSpeed', result.stdout + result.stderr)

    def test_toml_syntax_error_exits_before_training_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'broken.toml'
            config_path.write_text('[model\ntype = "wan"')
            result = subprocess.run(
                [sys.executable, 'train.py', '--validate_only', '--config', str(config_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Could not read training config', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_cli_typo_fails_before_training_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_valid_config(Path(tmp))
            result = subprocess.run(
                [sys.executable, 'train.py', '--validate_only', '--config', str(config_path), '--loging_steps', '2'],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unrecognized arguments: --loging_steps 2', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_help_exits_before_training_imports(self):
        result = subprocess.run(
            [sys.executable, 'train.py', '--help'],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--validate_only', result.stdout)
        self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_unknown_keys_and_late_scalar_errors_are_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            config_path.write_text(
                config_path.read_text()
                .replace('trust_cache = true', "trust_cache = true\nloging_steps = 2\nsave_dtype = 'half-ish'\nvideo_clip_mode = 'random'")
                .replace("dtype = 'bfloat16'", "dtype = 'bfloat16'\ntimestep_sample_method = 'mystery'", 1)
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        message = str(caught.exception)
        self.assertIn("unknown key 'loging_steps'", message)
        self.assertIn('save_dtype', message)
        self.assertIn('video_clip_mode', message)
        self.assertIn('timestep_sample_method', message)

    def test_main_config_errors_do_not_scan_dataset_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            config_path.write_text(config_path.read_text().replace('epochs = 10', 'epochs = 0'))

            with patch('utils.config_validation._inspect_dataset_media') as inspect_media:
                with self.assertRaises(ConfigValidationError) as caught:
                    load_and_validate_config(config_path)

        self.assertIn('epochs', str(caught.exception))
        inspect_media.assert_not_called()

    def test_dataset_structure_errors_do_not_scan_dataset_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            (root / 'dataset.toml').write_text(
                f'''\
resolutions = []

[[directory]]
path = {str(root / 'media')!r}
'''
            )

            with patch('utils.config_validation._inspect_dataset_media') as inspect_media:
                with self.assertRaises(ConfigValidationError) as caught:
                    load_and_validate_config(config_path)

        self.assertIn('needs non-empty resolutions', str(caught.exception))
        inspect_media.assert_not_called()

    def test_dataset_metadata_failures_are_reported_before_caching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            media_dir = root / 'media'
            control_dir = root / 'control'
            control_dir.mkdir()
            (media_dir / 'captions.json').write_text('{not json')
            (media_dir / 'example.txt').unlink()
            dataset_path = root / 'dataset.toml'
            dataset_path.write_text(
                f'''\
resolutions = [512]

[[directory]]
path = {str(media_dir)!r}
control_path = {str(control_dir)!r}
'''
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        message = str(caught.exception)
        self.assertIn('captions.json', message)
        self.assertIn('would be empty', message)
        self.assertIn('missing control files', message)

    def test_corrupt_safetensors_header_is_rejected_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            broken_weights = root / 'broken.safetensors'
            broken_weights.write_bytes(b'not a safetensors file')
            config_path.write_text(
                config_path.read_text()
                .replace("type = 'wan'", "type = 'sdxl'")
                .replace(
                    f"ckpt_path = {str(root / 'model')!r}",
                    f"checkpoint_path = {str(broken_weights)!r}",
                )
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        self.assertIn('not a readable safetensors file', str(caught.exception))

    def test_nsync_pairing_is_checked_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            dataset_path = root / 'dataset.toml'
            media_dir = root / 'media'
            dataset_path.write_text(
                f'''\
resolutions = [512]

[[directory]]
path = {str(media_dir)!r}
nsync_role = 'positive'
'''
            )
            config_path.write_text(
                config_path.read_text()
                .replace("type = 'wan'", "type = 'minimax_h3'")
                .replace(
                    f"ckpt_path = {str(root / 'model')!r}",
                    f"diffusion_model = {str(root / 'model')!r}\n"
                    f"vae = {str(root / 'model')!r}\n"
                    f"audio_vae = {str(root / 'model')!r}\n"
                    f"text_encoders = [{{path = {str(root / 'model')!r}, type = 'minimax'}}]",
                )
                + '''\

[adapter]
type = 'lora'
rank = 8

[training_methods.nsync]
enabled = true
'''
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        self.assertIn('exactly one positive and one negative', str(caught.exception))

    def test_nsync_anchor_pairs_are_checked_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            target_positive = root / 'media'
            target_negative = root / 'target_negative'
            anchor_positive = root / 'anchor_positive'
            anchor_negative = root / 'anchor_negative'
            for directory in (target_negative, anchor_positive, anchor_negative):
                directory.mkdir()
            (target_negative / 'example.png').write_bytes(b'not decoded during preflight')
            (anchor_positive / 'anchor.png').write_bytes(b'not decoded during preflight')
            (anchor_positive / 'anchor.txt').write_text('an anchor caption')
            (anchor_negative / 'anchor.png').write_bytes(b'not decoded during preflight')

            dataset_path = root / 'dataset.toml'
            dataset_path.write_text(
                f'''\
unbucketed = true
resolutions = [512]

[[directory]]
path = {str(target_positive)!r}
nsync_role = 'positive'
nsync_pair = 'target'
nsync_anchor_pairs = ['anchors']

[[directory]]
path = {str(target_negative)!r}
caption_path = {str(target_positive)!r}
nsync_role = 'negative'
nsync_pair = 'target'

[[directory]]
path = {str(anchor_positive)!r}
nsync_role = 'positive'
nsync_pair = 'anchors'

[[directory]]
path = {str(anchor_negative)!r}
caption_path = {str(anchor_positive)!r}
nsync_role = 'negative'
nsync_pair = 'anchors'
'''
            )
            model_dir = root / 'model'
            config_path.write_text(
                config_path.read_text()
                .replace("type = 'wan'", "type = 'minimax_h3'")
                .replace(
                    f"ckpt_path = {str(model_dir)!r}",
                    f"diffusion_model = {str(model_dir)!r}\n"
                    f"vae = {str(model_dir)!r}\n"
                    f"audio_vae = {str(model_dir)!r}\n"
                    f"text_encoders = [{{path = {str(model_dir)!r}, type = 'minimax'}}]",
                )
                + '''\

[adapter]
type = 'lora'
rank = 8

[training_methods.nsync]
enabled = true
'''
            )

            _, datasets = load_and_validate_config(config_path)
            dataset_config = next(iter(datasets.values()))
            self.assertEqual(dataset_config['directory'][0]['nsync_anchor_pairs'], ['anchors'])

            dataset_path.write_text(dataset_path.read_text().replace("['anchors']", "['missing_group']"))
            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        self.assertIn("references unknown anchor pair 'missing_group'", str(caught.exception))

    def test_nsync_media_pairing_is_checked_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            positive_dir = root / 'media'
            negative_dir = root / 'negative'
            negative_dir.mkdir()
            (negative_dir / 'different.png').write_bytes(b'not decoded during preflight')
            dataset_path = root / 'dataset.toml'
            dataset_path.write_text(
                f'''\
resolutions = [512]

[[directory]]
path = {str(positive_dir)!r}
nsync_role = 'positive'

[[directory]]
path = {str(negative_dir)!r}
caption_path = {str(positive_dir)!r}
nsync_role = 'negative'
'''
            )
            model_dir = root / 'model'
            config_path.write_text(
                config_path.read_text()
                .replace("type = 'wan'", "type = 'minimax_h3'")
                .replace(
                    f"ckpt_path = {str(model_dir)!r}",
                    f"diffusion_model = {str(model_dir)!r}\n"
                    f"vae = {str(model_dir)!r}\n"
                    f"audio_vae = {str(model_dir)!r}\n"
                    f"text_encoders = [{{path = {str(model_dir)!r}, type = 'minimax'}}]",
                )
                + '''\

[adapter]
type = 'lora'
rank = 8

[training_methods.nsync]
enabled = true
'''
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        self.assertIn('missing 1 negative media files', str(caught.exception))

    def test_self_flow_constraints_are_checked_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            model_dir = root / 'model'
            config_path.write_text(
                config_path.read_text()
                .replace("type = 'wan'", "type = 'minimax_h3'")
                .replace(
                    f"ckpt_path = {str(model_dir)!r}",
                    f"diffusion_model = {str(model_dir)!r}\n"
                    f"vae = {str(model_dir)!r}\n"
                    f"audio_vae = {str(model_dir)!r}\n"
                    f"text_encoders = [{{path = {str(model_dir)!r}, type = 'minimax'}}]",
                )
                + '''\

[adapter]
type = 'lora'
rank = 8

[training_methods.self_flow]
enabled = true
gamma = -1
ema_decay = 1
high_noise_range = [1.0, 0.5]
projection_dim = 0
student_layer = 8
teacher_layer = 4
'''
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        message = str(caught.exception)
        self.assertIn('self_flow.gamma', message)
        self.assertIn('self_flow.ema_decay', message)
        self.assertIn('self_flow.high_noise_range', message)
        self.assertIn('self_flow.projection_dim', message)
        self.assertIn('student_layer must be less than teacher_layer', message)

    def test_resume_target_must_contain_a_deepspeed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_valid_config(root)
            output_dir = root / 'output'
            output_dir.mkdir()
            (output_dir / '20260814_12-00-00').mkdir()

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path, resume_from_checkpoint=True)

        self.assertIn('missing', str(caught.exception))
        self.assertIn('latest', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
