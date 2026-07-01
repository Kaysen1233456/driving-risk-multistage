import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pipeline = load_module("pipeline_lib", ROOT / "src" / "pipeline" / "pipeline_lib.py")


class PipelineCoreTests(unittest.TestCase):
    def test_split_is_deterministic_and_complete(self):
        records = [{"sample_id": f"sample-{i}"} for i in range(20)]
        first = pipeline.build_split_map(records, seed=42)
        second = pipeline.build_split_map(records, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {r["sample_id"] for r in records})
        self.assertEqual(set(first.values()), {"train", "val", "test"})

    def test_answer_builders_return_json(self):
        record = {
            "transform_label": "疲劳1级_TTS干预",
            "final_parts": {
                "risk_description": "驾驶员持续闭眼",
                "intervention": "TTS干预",
                "risk_decision": "需要干预",
            },
        }
        payload = pipeline.build_final_answer_json(record)
        self.assertEqual(payload["transform_label"], record["transform_label"])
        json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
