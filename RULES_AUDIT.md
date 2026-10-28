# Elfnote ruling and visibility audit

The source-built engine now passes 15 controlled native scenarios, in addition
to the 15 existing pipeline tests. Run `scripts/run-audit.py` inside the Docker
image to reproduce the native suite. The saved machine-readable result is
`runs/rules-audit/report.json`, including the engine revisions and binary hash.

## Ruling scenarios

| Scenario | Expected outcome checked in the real core |
| --- | --- |
| Lucina hand summon | Offered when the center zone is free; occupies that zone |
| Occupied center | Lucina's hand Special Summon is unavailable; Normal Summon remains possible |
| Lucina search | Choice labels match eligible Elfnote monsters; Lucina itself is excluded |
| Lucina once per turn | A second copy cannot repeat the search that turn |
| Ash Blossom response | Ash goes to the GY; the search adds no card; the spent search stays spent |
| Regina cost | Another Elfnote is already in the GY while Regina remains in hand on the pending chain |
| Regina self-cost exclusion | Regina alone cannot pay its own summon cost |
| Regina center trigger | Center placement offers a different Elfnote from the Deck |
| Regina off-center control | The same summon outside the center does not offer that trigger |
| Lucina opponent-turn effect | Zone swap occurs; a qualifying opponent monster returns; the Level 8 control is ineligible |
| Regina Synchro material | A real Synchro Summon consumes the materials and Regina's trigger returns it to hand |

Some assertions share a test case, so this table is not a count of individual
test methods. Expectations were checked against the official card text for
[Lucina](https://www.db.yugioh-card.com/yugiohdb/card_search.action?cid=22116&ope=2&request_locale=en),
[Regina](https://www.db.yugioh-card.com/yugiohdb/card_search.action?cid=22523&ope=2&request_locale=en),
and [Ash Blossom](https://www.db.yugioh-card.com/yugiohdb/card_search.action?cid=12950&ope=2&request_locale=en).

## Visibility scenarios

The tests change actual hidden state in controlled boards and compare the
complete packed policy input and legal-action features. Hidden identities are
changed only where doing so leaves the public situation and legal choices
equivalent; a legitimate difference in available responses is not called a leak.

- Opponent hand, set monster, facedown banished card and facedown Extra Deck
  identities can change without changing the observer's inputs.
- Permuting either Main Deck leaves policy inputs unchanged.
- A public graveyard change is observable; face-up Extra Deck cards remain visible.
- An automatic target selection of a set card does not expose its identity in
  action history. The test pauses before resolution makes the card public.
- A searched-and-confirmed Regina is remembered through a reveal-event channel,
  while its hand slot and the other cards in that hand stay hidden.
- Changing an unrelated hidden card after that reveal leaves policy inputs unchanged.
- Ordinary training environments reject privileged inspection requests.

The privacy baseline follows the public/private distinction in the
[official tournament policy](https://www.yugioh-card.com/en/downloads/penalty_guide/YGOTCG_Policy_v_2_3.pdf).
Remembering an observed event does not expose the current location of a private
card or make another card in the same zone visible.

## Bugs found and fixed

1. The automatic-action lookup computed a hidden-card visibility result but
   fell through and returned the real ID anyway. It now returns the filtered ID.
2. A reveal previously disabled hiding for unrelated private zones, and stale
   location records could reveal later cards. Those overrides are removed.
3. The adapter hid all opponent Extra Deck cards, including face-up cards. The
   native encoder and policy filter now preserve face-up Extra Deck information.
4. Search prompts use selection-local deck indices in the current core. The old
   adapter treated them as physical deck positions and displayed the wrong IDs
   to the policy. It now preserves the authorized IDs delivered with the prompt.

An additional action-feature filter removes hidden target IDs before policy
input. Reveal events are stored separately from current private card locations.

## Test isolation and compatibility

Native scenarios use the core's Debug board setup and privileged snapshots to
assert outcomes. These are test fixtures, not deck-legal training episodes.
Only an explicitly configured audit worker enables them; the training CLI has
no audit-scenario option. Snapshots and raw observations are not policy features.

The observation schema is now integration revision 2 and includes `reveals_`.
Earlier checkpoints are incompatible with the corrected engine/observations;
their hashes fail normal evaluation checks. They remain available for historical
reference. A fresh compatible smoke-run checkpoint is saved at
`runs/elfnote-audit-v2/checkpoint-000001.pt`.

The packaged revision-2 image completed four fresh training games, saved and
reloaded that checkpoint, and completed eight balanced evaluation games with
zero Lua errors or truncations. Its native binary hash matches the audit report.
Those games validate the changed observation/training path, not playing strength.

## Remaining limits

This is targeted regression coverage, not certification of every TCG ruling.
The backend's TCG/OCG timing differences, other cards in the supplied list,
continuous hand-reveal effects and top-deck reveal mechanisms still need
dedicated scenarios. Unimplemented visibility mechanisms may conservatively
withhold legitimate knowledge. Passing these tests does not establish playing
strength or equilibrium convergence.
