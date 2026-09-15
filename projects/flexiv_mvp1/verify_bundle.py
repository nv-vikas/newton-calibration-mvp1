"""Offline integrity check of the delivered snapshot, not a new physics test."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "flexiv_peg_scene"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    collection = ROOT / "mvp1_collection"
    record = json.loads((collection / "bundle_record.json").read_text())
    for name, expected in record["hashes"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()), f"Path outside bundle: {name}")
        require(path.is_file(), f"Missing file: {name}")
        require(digest(path) == expected, f"Changed file: {name}")

    plan_path = collection / "collection_plan.json"
    plan = json.loads(plan_path.read_text())
    screen = json.loads((collection / "free_motion_screen.json").read_text())
    require(screen["command_plan_sha256"] == digest(plan_path), "Screen/plan hash mismatch")
    require(plan["asset_sha256"] == digest(ROOT / "assets/Flexiv_Rizon4s_Grav.usd"), "Asset hash mismatch")
    require(screen["physics"] == "Newton/MuJoCo Warp", "Unexpected physics backend")
    require(screen["passed"] is True, "Archived screen did not pass")
    require(screen["real_data"] is False, "Expected simulation-only screening")
    require(screen["real_execution_approved"] is False, "Bundle must not approve hardware")
    require(plan["real_samples"] == 0 and record["fitted"] is False, "Expected unfitted reference")
    episodes = plan["episodes"]
    require(len(episodes) == 9, "Expected nine trials")
    require(sum(e["split"] == "train" for e in episodes) == 7, "Expected seven training trials")
    require(sum(e["split"] == "heldout" for e in episodes) == 2, "Expected two held-out trials")
    require(sum(e["duration_s"] for e in episodes) == 152, "Expected 152 seconds")
    tests = {test["name"]: test for test in screen["tests"]}
    require(len(tests) == len(screen["tests"]) == len(episodes), "Trial count mismatch")
    require(set(tests) == {episode["name"] for episode in episodes}, "Trial names mismatch")
    for episode in episodes:
        command = (collection / episode["command_file"]).resolve()
        require(command.is_relative_to(collection.resolve()), "Command path outside collection")
        require(digest(command) == episode["sha256"], f"Command hash mismatch: {episode['name']}")
        test = tests[episode["name"]]
        require(test["split"] == episode["split"], "Screen split mismatch")
        require(test["passed"] is True and test["finite"] is True, "Archived trial did not pass")
        require(not test["moving_robot_fixture_contact_pairs"], "Archived fixture contact detected")

    print(json.dumps({
        "check": "offline snapshot integrity; no physics or hardware executed",
        "hashed_files_verified": len(record["hashes"]),
        "command_files_verified": len(episodes),
        "motion_duration_seconds": 152,
        "recorded_newton_screen_passed": True,
        "real_samples": 0,
        "calibrated": False,
        "real_execution_approved": False,
    }, indent=2))


if __name__ == "__main__":
    main()
