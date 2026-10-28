# Engine integration verification — October 28, 2026

## Fixed integration

The previous released binary and 2024 assets have been replaced by a source
build. The environment, current Fluorohydride core, matching mycard card scripts
and database are pinned in `engine/lock.json`. The vocabulary now comes from
15,019 database definitions rather than the historical deck support list.

Source compatibility patches cover core duel creation/destruction, set codes,
chain messages with per-option forced flags, and card-reveal skip-panel flags.
The adapter now handles monster-type and multi-attribute declarations, exposes
optional/cancel/unselect actions, preserves End Phase choices, and lets the
policy choose end-phase discards. Visible card IDs are filled correctly.

Engine execution is isolated in a spawned worker. Script errors, native aborts,
disconnects, malformed messages, timeouts and action overflow cannot become
successful game outcomes. Native prompt and Lua-error counts are logged.

## Executed validation

| Check | Result |
| --- | --- |
| Unit tests | 15 passed |
| Elfnote source-engine preflight | Ready; zero missing active cards |
| Kewl Tune engine vocabulary | Zero missing active cards; TCG Rotary restriction still rejected |
| Final CLI training | 2 PPO updates; 8 completed Elfnote mirror games; checkpoint saved |
| Final CLI evaluation | Checkpoint reloaded; 32 completed Elfnote games against random legal policy; 16 in each seat |
| Elfnote native diagnostics | Zero Lua errors and zero truncations in the 40-game final CLI run |
| Historical regression | 8 completed games; 27 card declarations; zero Lua errors |
| Packaged image | Reproduced engine hashes, reloaded the checkpoint and completed 4 additional Elfnote evaluation games |

Earlier integration runs additionally completed Elfnote games while fixes were
being developed. The final run above includes the restored End Phase and
cancel choices. The historical regression uses the unmodified upstream
Blue-Eyes list, including Crossout Designator, solely to test engine prompts.
It is explicitly labeled historical regression, not current-TCG training.

The first eight successful Elfnote integration games exercised chain responses,
reveals, Synchro sums, tribute selection, select/unselect, monster-type and
attribute declarations. The final source build is reproducible from the
Dockerfile; no downloaded precompiled native release is used.

## Local artifacts

- `runs/elfnote-engine-v1/checkpoint-000002.pt`
- `runs/elfnote-engine-v1/manifest.json`
- `runs/elfnote-engine-v1/metrics.jsonl`
- `runs/elfnote-engine-v1/tensorboard/`
- `runs/elfnote-engine-v1/games.csv`
- `runs/elfnote-engine-v1-eval/summary.json`
- `runs/elfnote-engine-v1-eval/metrics.jsonl`
- `runs/engine-regression/report.json`
- `runs/elfnote-packaged-eval/summary.json`

Generated artifacts are ignored by Git. Checkpoint metadata includes the binary,
scripts, database, deck and vocabulary hashes plus source bundle revisions.

## Limits of this evidence

These checks establish source-engine integration for the tested decks and
paths, not optimal play or exhaustive ruling correctness. TCG deck eligibility
is checked separately from the core's MR5 behavior; TCG/OCG timing differences
still need scenario-level review. Other upstream prompt paths and very large
action/combination spaces may require additional support. The current adapter
fails those cases rather than silently accepting partial action lists.

The learner remains a single-environment recurrent PPO baseline, verified on
CPU. GPU throughput and playing strength have not been established. Exact
per-player damage and resolved disruption attribution remain separate work.
