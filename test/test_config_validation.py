import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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

    def test_valid_config_loads_main_and_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_valid_config(Path(tmp))
            config, datasets = load_and_validate_config(config_path, world_size=1)

        self.assertEqual(config['model']['type'], 'wan')
        self.assertEqual(len(datasets), 1)

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

    def test_nsync_pairing_is_checked_without_scanning_media(self):
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

[training_methods.nsync]
enabled = true
'''
            )

            with self.assertRaises(ConfigValidationError) as caught:
                load_and_validate_config(config_path)

        self.assertIn('exactly one positive and one negative', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
