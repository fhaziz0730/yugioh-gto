"""Deterministic real-core scenarios. Opt in with YGO_NATIVE_AUDIT=1 in Linux.

Debug boards are test fixtures, not deck-legal training episodes. Privileged
snapshots assert outcomes but never enter the learner's observation stream.
"""
import os
from pathlib import Path
import unittest
import numpy as np

from yugioh_gto.deck import Deck
from yugioh_gto.engine import YgoEnvironment
from yugioh_gto.model import pack

LU, REGINA, PATRON = 13597785, 56651978, 12375297
VEILER, ASH, FUWA = 97268402, 14558128, 42141493
BLUE, DARK, ODD = 89631139, 46986414, 16178681
ROOT = Path(__file__).resolve().parent.parent


def board(cards, deck0=None, deck1=None):
    lines = ["Debug.ReloadFieldBegin(0,5)",
             "Debug.SetPlayerInfo(0,8000,0,0)", "Debug.SetPlayerInfo(1,8000,0,0)"]
    for player, deck in enumerate((deck0 or [VEILER]*8, deck1 or [VEILER]*8)):
        for sequence, code in enumerate(deck):
            lines.append(f"Debug.AddCard({code},{player},{player},1,{sequence},8)")
    for code, player, location, sequence, position in cards:
        lines.append(f"Debug.AddCard({code},{player},{player},{location},{sequence},{position})")
    lines.append("Debug.ReloadFieldEnd()")
    return "\n".join(lines)


@unittest.skipUnless(os.environ.get("YGO_NATIVE_AUDIT") == "1", "Requires the source-built native audit engine")
class NativeRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ROOT / "vendor/ygo-agent"
        codes = [int(line.split()[0]) for line in (cls.engine / "scripts/code_list.txt").read_text().splitlines()]
        cls.ids = {code: i+1 for i, code in enumerate(codes)}

    def setUp(self):
        self.envs = []

    def tearDown(self):
        for env in self.envs:
            env.close()

    def start(self, cards, deck0=None, deck1=None):
        env = YgoEnvironment(self.engine, Deck.read(ROOT / "decks/elfnote.ydk"), seed=17,
                             _audit_scenario=board(cards, deck0, deck1))
        self.envs.append(env)
        return env, self.idle(env, env.reset())

    def snapshot(self, env):
        return env._request("audit")["state"]

    def choices(self, state, *, code=None, act=None):
        obs, player, count = state
        rows = obs["actions_"][:count]
        return [i for i, row in enumerate(rows)
                if (code is None or int(row[1])*256+int(row[2]) == self.ids[code])
                and (act is None or int(row[4]) == act)]

    def step(self, env, state, index):
        nxt, reward, done, info = env.step(index)
        self.assertFalse(done, f"Scenario unexpectedly ended: {info}, reward={reward}")
        return nxt

    def idle(self, env, state, player=0, handler=None):
        for _ in range(80):
            obs, seat, count = state
            rows = obs["actions_"][:count]
            if seat == player and int(rows[0, 3]) == 1:
                return state
            selected = handler(state) if handler else None
            if selected is None:
                selected = next((i for i, row in enumerate(rows) if row[4] == 9), 0)
            state = self.step(env, state, selected)
        self.fail("Scenario did not reach expected player's idle prompt")

    def cards(self, env, code=None, player=None, location=None):
        return [c for c in self.snapshot(env)["cards"] if
                (code is None or c[0] == code) and (player is None or c[1] == player)
                and (location is None or c[2] == location)]

    def test_lucina_special_summon_requires_center_and_places_it_there(self):
        env, state = self.start([(LU,0,2,0,8)])
        special = self.choices(state, code=LU, act=3)
        self.assertTrue(special)
        state = self.idle(env, self.step(env, state, special[0]))
        field = self.cards(env, LU, 0, 4)
        self.assertEqual(len(field), 1)
        self.assertEqual(field[0][3], 2)

    def test_lucina_cannot_special_summon_into_occupied_center(self):
        _, state = self.start([(LU,0,2,0,8), (VEILER,0,4,2,1)])
        self.assertFalse(self.choices(state, code=LU, act=3))
        self.assertTrue(self.choices(state, code=LU, act=4))

    def test_lucina_search_and_name_once_per_turn(self):
        env, state = self.start([(LU,0,4,2,1), (LU,0,4,0,1), (VEILER,0,2,0,8)],
                                deck0=[REGINA, PATRON, LU]+[VEILER]*5)
        searches = self.choices(state, code=LU, act=8)
        self.assertEqual(len(searches), 2)
        selected_codes = []
        def choose_target(s):
            if int(s[0]["actions_"][0,3]) == 3:
                ids = {int(r[1])*256+int(r[2]) for r in s[0]["actions_"][:s[2]]}
                self.assertEqual(ids, {self.ids[REGINA], self.ids[PATRON]})
                selected_codes.append(ids)
                return self.choices(s, code=REGINA)[0]
        state = self.idle(env, self.step(env, state, searches[0]), handler=choose_target)
        self.assertTrue(selected_codes)
        self.assertEqual(len(self.cards(env, REGINA, 0, 2)), 1)
        self.assertFalse(self.choices(state, code=LU, act=8))

    def test_ash_discards_and_negates_lucina_search(self):
        env, state = self.start([(LU,0,4,2,1), (VEILER,0,2,0,8), (ASH,1,2,0,8)],
                                deck0=[REGINA, PATRON]+[VEILER]*6)
        state = self.step(env, state, self.choices(state, code=LU, act=8)[0])
        self.assertEqual(state[1], 1)
        ash = self.choices(state, code=ASH, act=8)
        self.assertTrue(ash)
        state = self.idle(env, self.step(env, state, ash[0]))
        self.assertEqual(len(self.cards(env, ASH, 1, 16)), 1)
        self.assertFalse(self.cards(env, REGINA, 0, 2))
        self.assertFalse(self.cards(env, PATRON, 0, 2))
        self.assertFalse(self.choices(state, code=LU, act=8))

    def regina_summon(self, zone):
        env, state = self.start([(REGINA,0,2,0,8), (LU,0,2,1,8), (FUWA,1,2,0,8)],
                                deck0=[REGINA, LU, PATRON]+[VEILER]*5)
        state = self.step(env, state, self.choices(state, code=REGINA, act=8)[0])
        # Cost has already been paid while Regina's effect is still on the chain.
        self.assertEqual(len(self.cards(env, LU, 0, 16)), 1)
        self.assertEqual(len(self.cards(env, REGINA, 0, 2)), 1)
        self.assertGreaterEqual(self.snapshot(env)["chain_size"], 1)
        triggered = []
        def choices(s):
            obs, seat, count = s
            rows = obs["actions_"][:count]
            msg = int(rows[0,3])
            if msg == 11:
                return next((i for i, row in enumerate(rows) if int(row[10]) == zone+1), 0)
            if msg == 6 and seat == 0:
                triggered.append(True)
                return next(i for i, row in enumerate(rows) if row[4] == 8)
            if msg == 3 and seat == 0:
                self.assertFalse(self.choices(s, code=REGINA))
                return self.choices(s, code=LU)[0]
        state = self.idle(env, state, handler=choices)
        return env, state, triggered

    def test_regina_pays_cost_then_center_summon_triggers_deck_summon(self):
        env, _, triggered = self.regina_summon(2)
        self.assertTrue(triggered)
        self.assertEqual(self.cards(env, REGINA, 0, 4)[0][3], 2)
        self.assertEqual(len(self.cards(env, LU, 0, 4)), 1)
        self.assertEqual(len(self.cards(env, LU, 0, 16)), 1)

    def test_regina_off_center_does_not_trigger_deck_summon(self):
        env, _, triggered = self.regina_summon(0)
        self.assertFalse(triggered)
        self.assertEqual(self.cards(env, REGINA, 0, 4)[0][3], 0)
        self.assertFalse(self.cards(env, LU, 0, 4))

    def test_regina_cannot_pay_with_itself(self):
        _, state = self.start([(REGINA,0,2,0,8), (VEILER,0,2,1,8)])
        self.assertFalse(self.choices(state, code=REGINA, act=8))

    def test_lucina_opponent_turn_swap_and_level_limited_bounce(self):
        env, state = self.start([(LU,0,4,0,1), (REGINA,0,4,2,1), (VEILER,0,2,0,8),
                                 (VEILER,1,4,0,1), (REGINA,1,4,1,1), (BLUE,1,4,4,1),
                                 (VEILER,1,2,0,8)])
        self.assertFalse(self.choices(state, code=LU, act=8))
        end = next(i for i,row in enumerate(state[0]["actions_"][:state[2]]) if row[7] == 3)
        state = self.step(env, state, end)
        for _ in range(40):
            actions = self.choices(state, code=LU, act=8) if state[1] == 0 else []
            if actions:
                state = self.step(env, state, actions[0])
                break
            rows = state[0]["actions_"][:state[2]]
            state = self.step(env, state, next((i for i,r in enumerate(rows) if r[4] == 9), 0))
        else:
            self.fail("Lucina quick effect was not offered on the opponent's turn")
        bounce = []
        def choose(s):
            rows = s[0]["actions_"][:s[2]]
            if s[1] == 0 and rows[0,3] == 7:
                return next(i for i,r in enumerate(rows) if r[4] != 9)
            if s[1] == 0 and rows[0,3] == 3:
                self.assertFalse(self.choices(s, code=BLUE))
                self.assertTrue(self.choices(s, code=REGINA))
                bounce.append(True)
                return self.choices(s, code=VEILER)[0]
        state = self.idle(env, state, player=1, handler=choose)
        self.assertTrue(bounce)
        self.assertEqual(self.cards(env, LU, 0, 4)[0][3], 2)
        self.assertEqual(self.cards(env, REGINA, 0, 4)[0][3], 0)
        self.assertFalse(self.cards(env, VEILER, 1, 4))
        self.assertTrue(self.cards(env, BLUE, 1, 4))

    def test_regina_recovers_after_being_synchro_material(self):
        riser = 68431965
        env, state = self.start([(REGINA,0,4,0,1), (VEILER,0,4,1,1),
                                 (VEILER,0,2,0,8), (riser,0,64,0,8)])
        synchro = self.choices(state, code=riser, act=3)
        self.assertTrue(synchro)
        recovered = []
        def choose(s):
            rows = s[0]["actions_"][:s[2]]
            if s[1] == 0 and self.choices(s, code=REGINA, act=8):
                recovered.append(True)
                return self.choices(s, code=REGINA, act=8)[0]
        self.idle(env, self.step(env, state, synchro[0]), handler=choose)
        self.assertTrue(recovered)
        self.assertEqual(len(self.cards(env, REGINA, 0, 2)), 1)
        self.assertFalse(self.cards(env, REGINA, 0, 16))
        self.assertEqual(len(self.cards(env, VEILER, 0, 16)), 1)
        self.assertEqual(len(self.cards(env, riser, 0, 4)), 1)

    def privacy_board(self, hidden=BLUE, public=VEILER):
        return [(VEILER,0,2,0,8), (hidden,1,2,0,8), (hidden,1,4,1,8),
                (hidden,1,16,0,1), (public,1,16,1,1),
                (hidden,1,32,0,8), (59822133 if hidden==BLUE else 68431965,1,64,0,8),
                (ODD,1,64,1,1)]

    def test_hidden_hand_set_banished_extra_and_draw_order_do_not_change_inputs(self):
        # Keep the public graveyard identical; vary only actual hidden cards.
        states = []
        for hidden, deck in ((BLUE,[LU,REGINA]+[VEILER]*6), (DARK,[REGINA,LU]+[VEILER]*6)):
            cards = self.privacy_board(hidden)
            cards[3] = (BLUE,1,16,0,1)
            env, state = self.start(cards, deck0=deck, deck1=list(reversed(deck)))
            self.assertTrue(self.cards(env, hidden, 1, 2))
            states.append(state)
        for a, b in zip(pack(states[0][0]), pack(states[1][0])):
            np.testing.assert_array_equal(a, b)

    def test_faceup_extra_and_graveyard_remain_visible(self):
        _, state = self.start(self.privacy_board())
        cards = state[0]["cards_"]
        ids = cards[:,0].astype(int)*256+cards[:,1]
        self.assertIn(self.ids[ODD], ids)
        self.assertIn(self.ids[BLUE], ids)  # Public graveyard card.
        self.assertNotIn(self.ids[59822133], ids)  # Opponent facedown Extra Deck.

    def test_search_reveal_survives_as_history_without_revealing_hand_slots(self):
        states = []
        for other_card in (BLUE, DARK):
            env, state = self.start([(LU,0,4,2,1), (other_card,0,2,0,8), (VEILER,1,2,0,8)],
                                    deck0=[REGINA, PATRON]+[VEILER]*6)
            def target(s):
                if int(s[0]["actions_"][0,3]) == 3:
                    return self.choices(s, code=REGINA)[0]
            state = self.idle(env, self.step(env, state, self.choices(state, code=LU, act=8)[0]), handler=target)
            end = next(i for i,row in enumerate(state[0]["actions_"][:state[2]]) if row[7] == 3)
            state = self.idle(env, self.step(env,state,end), player=1)
            reveals = state[0]["reveals_"]
            ids = set(reveals[:,0].astype(int)*256+reveals[:,1]) - {0}
            self.assertEqual(ids, {self.ids[REGINA]})
            cards = state[0]["cards_"]
            hand = (cards[:,2] == 2) & (cards[:,4] == 1)
            self.assertFalse(cards[hand,:2].any())
            states.append(state)
        for a, b in zip(pack(states[0][0]), pack(states[1][0])):
            np.testing.assert_array_equal(a, b)

    def test_public_graveyard_change_is_observable(self):
        states = [self.start(self.privacy_board(public=code))[1] for code in (VEILER, ASH)]
        self.assertFalse(np.array_equal(pack(states[0][0])[0], pack(states[1][0])[0]))

    def test_normal_environment_rejects_privileged_inspection(self):
        env = YgoEnvironment(self.engine, Deck.read(ROOT / "decks/elfnote.ydk"), seed=7)
        self.envs.append(env)
        obs, _, _ = env.reset()
        self.assertNotIn("state", obs)
        self.assertNotIn("raw", obs)
        with self.assertRaisesRegex(RuntimeError, "disabled outside audit scenarios"):
            env._request("audit")

    def test_automatic_hidden_target_cannot_leak_through_action_history(self):
        states = []
        cosmic, maxx = 8267140, 23434538
        for trap in (10045474, 15693423):
            env, state = self.start([(cosmic,0,2,0,8), (cosmic,0,2,1,8),
                                     (VEILER,0,2,2,8), (maxx,1,2,0,8), (trap,1,8,0,8)])
            state = self.step(env, state, self.choices(state, code=cosmic, act=8)[0])
            for _ in range(20):
                responses = self.choices(state, code=maxx, act=8) if state[1] == 1 else []
                if responses:
                    state = self.step(env, state, responses[0])
                    break
                rows = state[0]["actions_"][:state[2]]
                index = next((i for i,r in enumerate(rows) if r[4] == 9), 0)
                state = self.step(env, state, index)
            else:
                self.fail("Did not reach the pre-resolution response window")
            self.assertEqual(state[1], 0)
            self.assertTrue(self.cards(env, trap, 1, 8))
            self.assertTrue(self.choices(state, code=cosmic, act=8))
            history = state[0]["h_actions_"]
            target_rows = history[history[:,3] == 3]
            self.assertTrue(len(target_rows))
            self.assertFalse(target_rows[:,1:3].any())
            states.append(state)
        for a, b in zip(pack(states[0][0]), pack(states[1][0])):
            np.testing.assert_array_equal(a, b)

