import tempfile
import unittest
from pathlib import Path

from tools.generate_minimax_h3_nsync_negatives import (
    GenerationError,
    MediaInfo,
    WorkItem,
    bind_local_h3_workflow,
    find_output_resource,
    fit_generation_dimensions,
    load_api_workflow,
    make_generation_prompt,
    prepare_workflow,
    read_caption,
)


def local_h3_workflow():
    return {
        "12": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["1", 0],
                "vae": ["2", 0],
                "prompt": "template prompt",
                "width": 1344,
                "height": 768,
                "length": 124,
            },
            "_meta": {"title": "MiniMax H3 Image to Video"},
        },
        "20": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 7},
        },
        "30": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["29", 0], "filename_prefix": "video/ComfyUI"},
        },
    }


class GenerateMiniMaxH3NSyncNegativesTest(unittest.TestCase):
    def test_bundled_api_workflow_is_ready_for_negative_generation(self):
        workflow_path = Path(__file__).resolve().parents[1] / "examples" / "minimax_h3_t2va_api.json"
        workflow = load_api_workflow(workflow_path)
        binding = bind_local_h3_workflow(workflow)

        self.assertEqual(binding.prompt_node, "5")
        self.assertEqual(binding.shape_node, "5")
        self.assertEqual(binding.output_node, "14")
        self.assertNotIn("first_frame", workflow["5"]["inputs"])
        self.assertNotIn("last_frame", workflow["5"]["inputs"])
        self.assertEqual(workflow["13"]["inputs"]["fps"], 24.0)
        self.assertEqual(workflow["14"]["class_type"], "SaveVideo")
        self.assertEqual(workflow["13"]["inputs"]["audio"], ["12", 0])
        self.assertEqual(workflow["14"]["inputs"]["video"], ["13", 0])
        self.assertEqual(
            sum(node["class_type"] == "SaveVideo" for node in workflow.values()),
            1,
        )
        for node in workflow.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], workflow)

    def test_bind_and_prepare_local_h3_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = local_h3_workflow()
            binding = bind_local_h3_workflow(workflow)
            media = MediaInfo(root / "shot.mp4", "video", 1920, 1080, 60, 30.0, 48, True)
            item = WorkItem(
                positive=media.path,
                output=root / "shot.mp4",
                caption="a person walking",
                generation_prompt="a person walking",
                media=media,
                generation_width=1344,
                generation_height=768,
                generation_length=48,
                seed=123,
            )

            prepared = prepare_workflow(workflow, binding, item)

            self.assertEqual(binding.prompt_node, "12")
            self.assertEqual(binding.shape_node, "12")
            self.assertEqual(binding.output_node, "30")
            self.assertEqual(prepared["12"]["inputs"]["prompt"], "a person walking")
            self.assertEqual(prepared["12"]["inputs"]["length"], 48)
            self.assertEqual(prepared["20"]["inputs"]["noise_seed"], 123)
            self.assertEqual(prepared["30"]["inputs"]["filename_prefix"], "nsync/shot_123")
            self.assertEqual(workflow["12"]["inputs"]["prompt"], "template prompt")

    def test_hosted_minimax_node_is_rejected(self):
        workflow = local_h3_workflow()
        workflow["40"] = {
            "class_type": "MinimaxHailuo03TextToVideoNode",
            "inputs": {"model": {}},
        }

        with self.assertRaisesRegex(GenerationError, "Hosted MiniMax"):
            bind_local_h3_workflow(workflow)

    def test_connected_positive_frame_is_rejected(self):
        workflow = local_h3_workflow()
        workflow["12"]["inputs"]["first_frame"] = ["5", 0]

        with self.assertRaisesRegex(GenerationError, "first_frame connected"):
            bind_local_h3_workflow(workflow)

    def test_target_text_is_removed_from_generation_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            media = MediaInfo(path, "image", 1024, 768, 1, 0, 1, False)

            prompt = make_generation_prompt(
                "nsyncPerson, walking in NSYNC STYLE, at sunset",
                ["nsyncPerson", "nsync style"],
                "Documentary footage of {caption}",
                media.path,
                media,
            )

            self.assertEqual(prompt, "Documentary footage of walking in, at sunset")
            self.assertNotIn("nsync", prompt.lower())

    def test_caption_sources_and_output_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "shot.mp4"
            positive.write_bytes(b"")
            (root / "shot.txt").write_text("text caption\n", encoding="utf-8")
            self.assertEqual(read_caption(positive, None, 0), "text caption")

            captions = {"shot.mp4": ["first caption", "second caption"]}
            self.assertEqual(read_caption(positive, captions, 1), "second caption")

            history = {
                "outputs": {
                    "30": {
                        "videos": [
                            {"filename": "shot.mp4", "subfolder": "nsync", "type": "output"}
                        ]
                    }
                }
            }
            self.assertEqual(find_output_resource(history, "30")["filename"], "shot.mp4")

    def test_generation_dimensions_preserve_aspect_and_pixel_area(self):
        width, height = fit_generation_dimensions(1920, 1080, 1344 * 768)

        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertAlmostEqual(width / height, 16 / 9, delta=(16 / 9) * 0.03)
        self.assertAlmostEqual(width * height, 1344 * 768, delta=(1344 * 768) * 0.05)


if __name__ == "__main__":
    unittest.main()
