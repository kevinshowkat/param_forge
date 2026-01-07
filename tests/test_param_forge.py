import importlib.util
import pathlib
import tempfile
import unittest
import sys

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("param_forge", ROOT / "scripts" / "param_forge.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Unable to load param_forge module for tests")
param_forge = importlib.util.module_from_spec(SPEC)
sys.modules["param_forge"] = param_forge
SPEC.loader.exec_module(param_forge)


class TestWrapText(unittest.TestCase):
    def test_wrap_text_respects_width(self) -> None:
        lines = param_forge._wrap_text("hello world", 5)
        self.assertEqual(lines, ["hello", "world"])

    def test_wrap_text_long_word(self) -> None:
        lines = param_forge._wrap_text("abcdefgh", 3)
        self.assertEqual(lines, ["abc", "def", "gh"])


class TestCostFormatting(unittest.TestCase):
    def test_format_cost_value(self) -> None:
        self.assertEqual(param_forge._format_cost_value(0.002), "$2/1K")
        self.assertEqual(param_forge._format_cost_value(None), "N/A")


class TestRetrievalScore(unittest.TestCase):
    def test_compute_retrieval_score_full(self) -> None:
        axes = {key: 100 for key in param_forge._RETRIEVAL_AXIS_KEYS}
        self.assertEqual(param_forge._compute_retrieval_score(axes), 100)

    def test_compute_retrieval_score_partial(self) -> None:
        axes = {"text_legibility": 80}
        self.assertEqual(param_forge._compute_retrieval_score(axes), 80)

    def test_compact_retrieval_packet(self) -> None:
        packet = {
            "alt_text_120": "alt",
            "caption_280": "cap",
            "entities": ["e1"],
            "claims": [{"text": "c1", "confidence": 0.5}],
            "questions_answered": ["q1"],
            "flags": ["f1"],
            "extra": "drop",
        }
        compact = param_forge._compact_retrieval_packet(packet)
        self.assertIn("alt_text_120", compact)
        self.assertIn("caption_280", compact)
        self.assertIn("entities", compact)
        self.assertIn("claims", compact)
        self.assertIn("questions_answered", compact)
        self.assertIn("flags", compact)
        self.assertNotIn("extra", compact)


class TestImageQualityMetrics(unittest.TestCase):
    def test_quality_gates(self) -> None:
        gates = param_forge._image_quality_gates(
            {
                "brightness_luma_mean": 30,
                "contrast_luma_std": 10,
                "sharpness_edge_mean": 2,
            }
        )
        self.assertIn("too_dark", gates)
        self.assertIn("low_contrast", gates)
        self.assertIn("low_sharpness", gates)

    @unittest.skipIf(Image is None, "Pillow not available")
    def test_compute_image_quality_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "sample.png"
            Image.new("RGB", (32, 32), (255, 255, 255)).save(path)
            metrics = param_forge._compute_image_quality_metrics(path)
            self.assertIn("brightness_luma_mean", metrics)
            self.assertIn("contrast_luma_std", metrics)
            self.assertIn("sharpness_edge_mean", metrics)
            self.assertIn("colorfulness", metrics)
            self.assertIn("sampled_width", metrics)
            self.assertIn("sampled_height", metrics)


class TestDiffCallSettings(unittest.TestCase):
    def test_output_format_default_annotation(self) -> None:
        prev = {
            "provider": "openai",
            "model": "gpt-image-1.5",
            "size": "1024x1024",
            "n": 1,
            "output_format": None,
        }
        curr = {
            "provider": "openai",
            "model": "gpt-image-1.5",
            "size": "1024x1024",
            "n": 1,
            "output_format": "png",
        }
        diffs = param_forge._diff_call_settings(prev, curr)
        self.assertTrue(
            any(
                diff.startswith("output_format: null (default: png)")
                and diff.endswith("-> png")
                for diff in diffs
            )
        )

    def test_provider_option_default_annotation(self) -> None:
        prev = {
            "provider": "gemini",
            "model": "gemini-2.5-flash-image",
            "size": "1024x1024",
            "n": 1,
            "provider_options": {},
        }
        curr = {
            "provider": "gemini",
            "model": "gemini-2.5-flash-image",
            "size": "1024x1024",
            "n": 1,
            "provider_options": {"image_size": "2K"},
        }
        diffs = param_forge._diff_call_settings(prev, curr)
        self.assertTrue(
            any(
                diff.startswith("provider_options.image_size: null (default: 1K)")
                and diff.endswith("-> 2K")
                for diff in diffs
            )
        )


if __name__ == "__main__":
    unittest.main()
