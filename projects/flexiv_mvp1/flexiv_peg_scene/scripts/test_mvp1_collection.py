import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from collect_flexiv_rdk import load_commands, validate_approval

PLAN = Path(__file__).resolve().parents[1] / "mvp1_collection/collection_plan.json"


class CollectionChecks(unittest.TestCase):
    def test_all_nine_files_have_bounded_smooth_commands(self):
        plan = json.loads(PLAN.read_text())
        self.assertEqual(len(plan["episodes"]), 9)
        self.assertEqual(sum(e["split"] == "heldout" for e in plan["episodes"]), 2)
        for episode in plan["episodes"]:
            _, _, rows = load_commands(PLAN, episode["name"])
            self.assertGreater(len(rows), 1000)
        self.assertEqual(plan["real_samples"], 0)

    def test_unapproved_execution_rejected(self):
        with self.assertRaises(ValueError):
            validate_approval({}, json.loads(PLAN.read_text()), PLAN, "test-robot")

    def test_tampered_command_rejected(self):
        plan = json.loads(PLAN.read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "commands").mkdir()
            episode = plan["episodes"][0]
            (root / episode["command_file"]).write_text("tampered")
            (root / "plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_commands(root / "plan.json", episode["name"])

    def test_invalid_motion_rejected_even_with_new_hash(self):
        plan = json.loads(PLAN.read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "commands").mkdir()
            episode = plan["episodes"][0]
            original = (PLAN.parent / episode["command_file"]).read_text().splitlines()
            fields = original[1].split(",")
            fields[1] = "nan"
            original[1] = ",".join(fields)
            target = root / episode["command_file"]
            target.write_text("\n".join(original)+"\n")
            episode["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            (root / "plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                load_commands(root / "plan.json", episode["name"])


if __name__ == "__main__":
    unittest.main()
