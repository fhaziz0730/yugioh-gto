"""Resume must reproduce updates, opponents, RNG streams and environment draws."""
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
torch.set_num_threads(1)

from yugioh_gto.checkpoint import read_resume, save_training
from yugioh_gto.training import Settings, train, load_policy


class RandomFixture:
    def __init__(self, seed=17, fail_on_step=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fail_on_step = fail_on_step
        self.steps = 0

    def reset(self):
        self.draw = int(self.rng.integers(1, 200))
        self.n, self.scores = 0, [0, 0]
        return self.observation()

    def observation(self):
        return {"cards_": np.zeros((3, 8), np.uint8),
                "global_": np.array([self.n, self.draw], np.uint8),
                "h_actions_": np.zeros((2, 4), np.uint8),
                "actions_": np.array([[0, 0], [255, 0], [0, 255]], np.uint8)}, self.n % 2, 2

    def step(self, action):
        self.steps += 1
        if self.steps == self.fail_on_step:
            raise RuntimeError("simulated process interruption")
        actor = self.n % 2
        self.scores[actor] += action == ((self.draw + self.n // 2) % 2)
        self.n += 1
        if self.n == 6:
            winner = int(self.scores[1] > self.scores[0])
            return None, 1. if winner == actor else -1., True, {"truncated": False}
        return self.observation(), 0., False, {}

    def state_dict(self):
        return {"version": 1, "seed": self.seed, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        if state["seed"] != self.seed:
            raise ValueError("Environment seed mismatch")
        self.rng.bit_generator.state = state["rng"]


class ResumeTests(unittest.TestCase):
    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def settings(self, **kwargs):
        return Settings(**{**dict(iterations=4, games_per_iteration=2, hidden=16,
                                  save_every=1, pool_size=2, ppo_epochs=2), **kwargs})

    def game_records(self, directory):
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        for row in rows:
            row.pop("seconds", None)
        return [row for row in rows if row["event"] in ("game", "update")]

    def assert_equivalent(self, baseline, resumed):
        a, b = read_resume(baseline / "latest.pt"), read_resume(resumed / "latest.pt")
        for key in ("iteration", "model", "optimizer", "spec"):
            self.assert_state_equal(a[key], b[key])
        for key in ("rng", "environment", "opponent_pool"):
            self.assert_state_equal(a["resume"][key], b["resume"][key])

    def test_resume_restores_pool_optimizer_and_all_random_streams(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            settings = self.settings()
            train(RandomFixture(), root / "full", settings, {"deck_sha256": "fixture"})
            train(RandomFixture(), root / "part", replace(settings, iterations=2), {"deck_sha256": "fixture"})
            random.random()
            np.random.random(13)
            torch.rand(17)
            train(RandomFixture(), root / "resumed", settings, {"deck_sha256": "fixture"},
                  resume=root / "part/latest.pt")
            self.assert_equivalent(root / "full", root / "resumed")
            self.assertEqual(self.game_records(root / "full"),
                             self.game_records(root / "part") + self.game_records(root / "resumed"))
            saved = read_resume(root / "resumed/latest.pt")
            self.assertEqual([x["name"] for x in saved["resume"]["opponent_pool"]],
                             ["checkpoint-000003", "checkpoint-000004"])
            self.assertEqual(saved["metadata"]["resumed_from"]["iteration"], 2)

    def test_stopping_between_archive_intervals_does_not_promote_extra_opponent(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            settings = self.settings(iterations=3, save_every=3)
            train(RandomFixture(), root / "full", settings, {})
            train(RandomFixture(), root / "part", replace(settings, iterations=1), {})
            self.assertEqual(len(read_resume(root / "part/latest.pt")["resume"]["opponent_pool"]), 1)
            train(RandomFixture(), root / "resumed", settings, {}, resume=root / "part/latest.pt")
            self.assert_equivalent(root / "full", root / "resumed")

    def test_interrupted_rollout_restarts_from_last_completed_update(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            settings = self.settings(iterations=2, save_every=5)
            train(RandomFixture(), root / "full", settings, {})
            with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                train(RandomFixture(fail_on_step=14), root / "interrupted", settings, {})
            self.assertEqual(read_resume(root / "interrupted/latest.pt")["iteration"], 1)
            train(RandomFixture(), root / "resumed", settings, {}, resume=root / "interrupted/latest.pt")
            self.assert_equivalent(root / "full", root / "resumed")

    def test_resume_rejects_changed_experiment_settings_runtime_and_code(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            settings = self.settings(iterations=1)
            train(RandomFixture(), root / "part", settings, {"deck_sha256": "original"})
            source = root / "part/latest.pt"
            for field in ("deck_sha256", "native_binary_sha256", "snapshot_sha256"):
                with self.subTest(identity=field), self.assertRaisesRegex(ValueError, "experiment mismatch"):
                    train(RandomFixture(), root / "bad", replace(settings, iterations=2),
                          {"deck_sha256": "original", field: "changed"}, resume=source)
                self.assertFalse((root / "bad").exists())
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                train(RandomFixture(), root / "bad", replace(settings, iterations=2, seed=8),
                      {"deck_sha256": "original"}, resume=source)
            for field, message in (("runtime", "runtime mismatch"), ("trainer_sha256", "code mismatch")):
                saved = read_resume(source)
                saved["resume"][field] = {} if field == "runtime" else "different"
                modified = root / f"changed-{field}.pt"
                torch.save(saved, modified)
                with self.assertRaisesRegex(ValueError, message):
                    train(RandomFixture(), root / "bad", replace(settings, iterations=2),
                          {"deck_sha256": "original"}, resume=modified)

    def test_legacy_and_incomplete_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"model": {}}, path)
            with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
                read_resume(path)
            torch.save({"format_version": 2, "resume": {}}, path)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_resume(path)
            path.write_bytes(b"incomplete checkpoint")
            with self.assertRaisesRegex(ValueError, "Cannot read resume checkpoint"):
                read_resume(path)

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            settings = self.settings(iterations=1)
            train(RandomFixture(), root / "part", settings, {})
            path = root / "part/latest.pt"
            previous = path.read_bytes()
            policy, saved = load_policy(path)
            optimizer = torch.optim.Adam(policy.parameters())
            optimizer.load_state_dict(saved["optimizer"])
            def interrupted_write(value, temporary):
                Path(temporary).write_bytes(b"incomplete save")
                raise OSError("simulated disk failure")
            with patch("yugioh_gto.checkpoint.torch.save", side_effect=interrupted_write):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    save_training(path, policy, optimizer, 2, {}, settings,
                                  rng=np.random.default_rng(7), env=RandomFixture(),
                                  pool=[("initial", policy)], identity={}, device="cpu")
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(read_resume(path)["iteration"], 1)

    def test_output_reuse_and_nonincreasing_target_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            settings = self.settings(iterations=1)
            train(RandomFixture(), root / "part", settings, {})
            with self.assertRaisesRegex(ValueError, "not empty"):
                train(RandomFixture(), root / "part", replace(settings, iterations=2), {}, resume=root / "part/latest.pt")
            with self.assertRaisesRegex(ValueError, "total target"):
                train(RandomFixture(), root / "bad", settings, {}, resume=root / "part/latest.pt")
