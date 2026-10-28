"""Synthetic fixtures test the pipeline only; they never simulate Yu-Gi-Oh!."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
torch.set_num_threads(1)

from yugioh_gto.deck import Deck, supported_codes
from yugioh_gto.engine import sanitize, YgoEnvironment, _NativeYgoEnvironment
from yugioh_gto.legality import validate, Tables
from yugioh_gto.model import Policy, pack
from yugioh_gto.training import Settings, train, evaluate, load_policy, play, ppo_update, wilson


class FixtureEnvironment:
    """Deterministic two-seat, four-action fixture with actor-relative rewards."""
    def reset(self):
        self.n = 0
        return self.observation()

    def observation(self):
        return {"cards_": np.zeros((3, 8), np.uint8), "global_": np.array([self.n], np.uint8),
                "h_actions_": np.zeros((2, 4), np.uint8),
                "actions_": np.array([[0, 0], [255, 0], [0, 255], [255, 255]], np.uint8)}, self.n % 2, 2

    def step(self, action):
        assert action in (0, 1), "Illegal action in fixture"
        self.n += 1
        if self.n == 4:
            # Last actor is seat 1; a loss for that actor is a win for seat 0.
            return None, -1., True, {"truncated": False, "win_reason": 1}
        return self.observation(), 0., False, {}


class PipelineTests(unittest.TestCase):
    def test_script_errors_are_not_game_outcomes(self):
        env = object.__new__(_NativeYgoEnvironment)
        env.native = SimpleNamespace(integration_errors=lambda: ["Card script failed"])
        with self.assertRaisesRegex(RuntimeError, "Card script failed"):
            env.check_errors({"integration_error": [1]})

    def test_native_timeout_is_not_a_win(self):
        env = object.__new__(_NativeYgoEnvironment)
        env.native = SimpleNamespace(integration_errors=lambda: [])
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            env.check_errors({"integration_error": [1]})

    def test_exact_action_capacity_is_valid_without_overflow(self):
        env = object.__new__(_NativeYgoEnvironment)
        env.native = SimpleNamespace(integration_errors=lambda: [])
        env.max_options = 256
        obs = {"cards_": np.zeros((1, 3, 41), dtype=np.uint8)}
        _, seat, count = env.unpack(obs, {"num_options": [256], "to_play": [0], "action_overflow": [0]})
        self.assertEqual((seat, count), (0, 256))
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            env.unpack(obs, {"num_options": [256], "to_play": [0], "action_overflow": [1]})

    def test_supplied_deck_and_side_exclusion(self):
        deck = Deck.read("decks/elfnote.ydk")
        self.assertEqual((len(deck.main), len(deck.extra), len(deck.side)), (42, 15, 15))
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory) / "active.ydk"
            deck.write_active(path)
            self.assertEqual(Deck.read(path).side, [])

    def test_fail_closed_release_and_limits(self):
        deck = Deck([1] * 3 + list(range(2, 39)), [], [])
        cards = {str(code): dict(name=str(code), tcg_release="2026-01-01", limit=3) for code in deck.main}
        snapshot = dict(as_of="2026-10-28", effective="2026-10-19", cards=cards)
        self.assertEqual(validate(deck, snapshot, "2026-10-28"), [])
        cards["1"]["limit"] = 1
        cards["2"]["tcg_release"] = "2026-11-01"
        cards["3"]["tcg_release"] = None
        errors = validate(deck, snapshot, "2026-10-28")
        self.assertEqual(len(errors), 3)

    def test_artwork_aliases_share_limit(self):
        deck = Deck([1, 2] + list(range(3, 41)), [], [])
        cards = {str(c): dict(name=str(c), tcg_release="2026-01-01", limit=3) for c in deck.main}
        cards["1"].update(alias=1, limit=1)
        cards["2"].update(alias=1, limit=1)
        errors = validate(deck, dict(as_of="2026-10-28", cards=cards), "2026-10-28")
        self.assertTrue(any("limit is 1" in e for e in errors))

    def test_banlist_table_parser(self):
        parser = Tables()
        parser.feed('<table><tr><td>Spell</td><td>A &amp; B</td><td>Forbidden</td></tr></table>')
        self.assertEqual(parser.rows, [["Spell", "A & B", "Forbidden"]])

    def test_private_cards_do_not_leak(self):
        cards = np.ones((6, 41), np.uint8) * 99
        cards[:, 2] = [1, 1, 2, 2, 7, 3]
        cards[:, 4] = [0, 1, 0, 1, 1, 1]
        cards[5, 5] = 6
        original = cards.copy()
        result = sanitize({"cards_": cards})["cards_"]
        for row in (0, 1, 3, 4):
            self.assertFalse(result[row, :2].any())
            self.assertFalse(result[row, 7:].any())
        self.assertEqual(result[2, 0], 99)  # Own hand is visible.
        self.assertFalse(result[5, 7:].any())  # Opponent set monster.
        np.testing.assert_array_equal(cards, original)

    def test_action_mask(self):
        obs, _, count = FixtureEnvironment().reset()
        features, actions = pack(obs)
        policy = Policy(len(features), actions.shape[-1], 16)
        for _ in range(20):
            action, *_ = policy.act(features, actions, count)
            self.assertLess(action, count)

    def test_actor_relative_terminal_reward(self):
        _, result = play(FixtureEnvironment(), [None, None], np.random.default_rng(1), 10)
        self.assertEqual(result["winner"], 0)

    def test_truncation_has_no_winner(self):
        _, result = play(FixtureEnvironment(), [None, None], np.random.default_rng(1), 2)
        self.assertTrue(result["truncated"])
        self.assertIsNone(result["winner"])

    def test_ppo_changes_parameters(self):
        env = FixtureEnvironment()
        observation, _, _ = env.reset()
        features, actions = pack(observation)
        policy = Policy(len(features), actions.shape[-1], 16)
        old = {k: v.clone() for k, v in policy.state_dict().items()}
        trajectories, _ = play(env, [policy, policy], np.random.default_rng(1), 10, [0])
        stats = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=.01),
                           [(trajectories[0], 1.)], Settings(ppo_epochs=2))
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertTrue(any(not torch.equal(v, old[k]) for k, v in policy.state_dict().items()))

    def test_training_checkpoint_logs_and_evaluation(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = Path(directory).resolve()
            settings = Settings(iterations=2, games_per_iteration=2, hidden=16, save_every=1, max_decisions=10)
            train(FixtureEnvironment(), path / "train", settings, {"backend": "TEST_FIXTURE"})
            policy, saved = load_policy(path / "train/checkpoint-000002.pt")
            self.assertEqual(saved["iteration"], 2)
            self.assertEqual(saved["metadata"]["backend"], "TEST_FIXTURE")
            summary = evaluate(FixtureEnvironment(), policy, None, path / "eval", games=4)
            self.assertEqual(summary["going_first"]["wins"], 2)
            self.assertEqual(summary["going_second"]["losses"], 2)
            self.assertTrue((path / "train/games.csv").exists())
            self.assertTrue((path / "train/metrics.jsonl").exists())
            self.assertTrue(list((path / "train/tensorboard").iterdir()))

    def test_confidence_interval(self):
        low, high = wilson(50, 100)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertIsNone(wilson(0, 0))

    def test_real_engine_refuses_unsupported_deck(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/code_list.txt").write_text("1 0\n")
            with self.assertRaisesRegex(RuntimeError, "support missing"):
                YgoEnvironment(root, Deck.read("decks/elfnote.ydk"))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
