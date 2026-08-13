import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = REPO_ROOT / "tools" / "setup_runpod_minimax_h3.sh"


class RunPodSetupScriptTest(unittest.TestCase):
    def _write_executable(self, path: Path, contents: str) -> None:
        path.write_text(textwrap.dedent(contents), encoding="utf-8")
        path.chmod(0o755)

    def test_reuses_existing_comfy_models_and_cached_environment(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory).resolve()
            comfy_root = workspace / "ComfyUI"
            (comfy_root / "models" / "diffusion_models").mkdir(parents=True)
            (comfy_root / "models" / "text_encoders").mkdir(parents=True)
            (comfy_root / "models" / "vae").mkdir(parents=True)
            (comfy_root / "main.py").touch()

            model_paths = [
                comfy_root / "models" / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                comfy_root / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                comfy_root / "models" / "vae" / "minimax_h3_video_vae_fp16.safetensors",
                comfy_root / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
            ]
            for model_path in model_paths:
                model_path.touch()

            fake_bin = workspace / "fake-bin"
            fake_bin.mkdir()
            self._write_executable(
                fake_bin / "python",
                """
                #!/usr/bin/env bash
                if [[ "${1:-}" == "-" && "$#" == 1 ]]; then
                    printf '%s\\n' '3.12.0' '2.9.1+cu130' '13.0' 'true' 'Fake GPU'
                elif [[ "${1:-}" == "-" && "$#" == 4 ]]; then
                    printf '%s\\n' 'py3_12_0-torch2_9_1-cu13_0'
                fi
                """,
            )
            self._write_executable(
                fake_bin / "git",
                """
                #!/usr/bin/env bash
                if [[ "$*" == *"rev-parse HEAD"* ]]; then
                    printf '%s\\n' '0123456789abcdef0123456789abcdef01234567'
                fi
                """,
            )
            for command in ("git-lfs", "ffmpeg", "ffprobe", "curl", "wget", "jq", "tmux", "rsync"):
                self._write_executable(fake_bin / command, "#!/usr/bin/env bash\nexit 0\n")

            venv = workspace / "venvs" / "diffusion-pipe-py3_12_0-torch2_9_1-cu13_0"
            (venv / "bin").mkdir(parents=True)
            self._write_executable(
                venv / "bin" / "python",
                """
                #!/usr/bin/env bash
                exit 0
                """,
            )
            (venv / "bin" / "activate").write_text(
                f"export VIRTUAL_ENV={venv}\n",
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            command = [
                "bash",
                str(SETUP_SCRIPT),
                "--workspace",
                str(workspace),
                "--comfy-root",
                str(comfy_root),
                "--skip-system-packages",
                "--allow-nonpersistent-workspace",
            ]
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f"Using existing ComfyUI: {comfy_root}", result.stdout)
            self.assertIn(f"Diffusion model: {model_paths[0]}", result.stdout)

            env_file = workspace / "minimax-h3-env.sh"
            self.assertTrue(env_file.is_file())
            env_contents = env_file.read_text(encoding="utf-8")
            self.assertIn(f"export COMFYUI_ROOT={comfy_root}", env_contents)
            self.assertIn(f"export H3_DIFFUSION_MODEL={model_paths[0]}", env_contents)
            self.assertIn(f"export DP_VENV={venv}", env_contents)
            self.assertTrue((venv / ".diffusion-pipe-setup-signature").is_file())

            second_result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(second_result.returncode, 0, second_result.stdout + second_result.stderr)
            self.assertIn("Persistent Python environment is current; skipping pip installation", second_result.stdout)

    def test_help_is_available_without_setup(self):
        result = subprocess.run(
            ["bash", str(SETUP_SCRIPT), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--comfy-root PATH", result.stdout)
        self.assertIn("--rebuild-venv", result.stdout)


if __name__ == "__main__":
    unittest.main()
