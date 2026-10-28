# Yu-Gi-Oh! self-play pipeline

Recurrent PPO in PyTorch, frozen checkpoint opponents, legal-action masking,
TensorBoard logs, and evaluation split by going first/second. The default
experiment uses the supplied 42-card Ars Magna Elfnote deck in mirror matches,
single games, without siding. This is a research learner, not a proven GTO solver.

## Working engine integration

The latest [ruling and visibility audit](RULES_AUDIT.md) adds 15 deterministic
real-core scenarios and fixes hidden target history, unrelated-card reveal
leaks, face-up Extra Deck visibility, and search-choice IDs. Integration revision
2 includes a reveal-history observation channel. Older checkpoints are retained
for reference but are incompatible; use a fresh run with the current image.

[Checkpoint resume](RESUME.md) now restores the full training and native RNG
state. `latest.pt` is saved after every completed update; exact CPU continuation
was verified against an uninterrupted Elfnote run.

The native adapter and YGOPro core are compiled together from pinned source.
Matching current card scripts and database provide 15,019 card definitions,
including every active card in both submitted decks. There is no dependency
on the outdated upstream binary release. Revisions are in `engine/lock.json`;
checked source patches are applied by `engine/prepare.py`.

Elfnote passes preflight. Kewl Tune's engine card coverage also passes, but its
three Kewl Tune Rotary copies fail the selected TCG banlist. No deck cards were
silently removed or replaced. Side cards are excluded from these single games.

Earlier revision-1 engine validation completed 8 Elfnote training games and 32 balanced
evaluation games, with no truncations or Lua errors. A separate historical
regression completed 8 games and exercised 27 card declarations, covering the
prompt that previously crashed the released binary. See `VERIFICATION.md`.

## Docker quick start (Windows)

Start Docker Desktop with Linux containers, then run from this project:

```powershell
docker build -t yugioh-gto:source .
docker run --rm yugioh-gto:source preflight
docker run --rm --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source train --iterations 2 --games-per-iteration 4 --output runs/my-elfnote-run --device cpu
docker run --rm --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source evaluate --candidate runs/my-elfnote-run/checkpoint-000002.pt --games 32 --output runs/my-elfnote-eval --device cpu
```

Create the host `runs` directory first if it does not exist. Use a fresh output
directory for each CLI run. The generated files appear in your host `runs/`.
The latest compatible resumable smoke-run model is at
`runs/elfnote-resumed-cli/latest.pt` (iteration 2). Earlier audit checkpoints
are preserved as historical artifacts.

The default image uses CPU PyTorch. For GPU training, build with a compatible
CUDA wheel index using `--build-arg TORCH_INDEX_URL=...`, and run with `--gpus all`.
Select the index using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
The laptop GPU has not yet been benchmarked. No paid compute is required for
the verified pipeline, and no cloud jobs are started automatically.

## Native Linux setup

Install Python 3.10+, a C++17 compiler, Git, CMake, Ninja, and the development
packages for Lua 5.3, SQLiteCpp, glog and concurrentqueue. On Debian/Ubuntu the
native dependencies are:

```bash
sudo apt-get install build-essential git cmake ninja-build liblua5.3-dev libsqlitecpp-dev libgoogle-glog-dev libconcurrentqueue-dev
bash scripts/setup-linux.sh
source .venv-linux/bin/activate
python -m yugioh_gto.cli preflight
```

`engine/fetch.py` downloads only pinned snapshots, avoiding the large card
database Git history. CMake fetches pinned fmt/unordered_dense dependencies.
The Windows learner/tests can also run in `.venv`; native simulation runs in
Linux through Docker or WSL. Windows source files alone do not include a native
engine binary: run preflight in the built Linux environment.

## Training, logs and evaluation

```bash
python -m yugioh_gto.cli train --iterations 100 --games-per-iteration 8 --output runs/elfnote-001
python -m yugioh_gto.cli evaluate --candidate runs/elfnote-001/checkpoint-000100.pt --opponent runs/elfnote-001/checkpoint-000050.pt --games 100 --output runs/eval-checkpoint
tensorboard --logdir runs/elfnote-001/tensorboard
```

Continue into a fresh directory, restoring the saved configuration:

```bash
python -m yugioh_gto.cli train --resume runs/elfnote-001/latest.pt --iterations 200 --output runs/elfnote-continued
```

The target includes saved iterations. See [resume behavior and verification](RESUME.md).

| Output | Contents |
| --- | --- |
| `manifest.json` | Configuration, seed, device, source revisions and asset/binary hashes |
| `metrics.jsonl` | Games, PPO updates, resource/action diagnostics and native prompt/error counts |
| `results.sqlite3`, `games.csv` | Match records and export |
| `latest.pt` | Atomic recovery checkpoint after every completed update |
| `checkpoint-*.pt` | Archives with model, optimizer, opponent pool, RNG and experiment state |
| `tensorboard/` | Training metrics |
| `summary.json` | Evaluation outcomes and Wilson 95% intervals by turn order |

The learner alternates seats and samples frozen checkpoint opponents, starting
with an untrained policy. Players have separate recurrent memory. PPO replays
full per-player sequences using undiscounted terminal returns (+1/-1/0).
Winning is the only reward; long sequences can consume substantial memory.
The first implementation runs one native environment per learner.

Summon/attack/activation diagnostics count choices, not successful resolutions.
Hand/board advantage and life points are last-observed values. Native prompt
counts now record engine events such as damage and special summons, but exact
per-player damage/disruption attribution still needs further instrumentation.

## Integration safeguards

- Native gameplay runs in a separate worker process. A native abort, disconnect,
  Lua error or timeout fails the run without inventing a win or training label.
- The adapter checks a source-integration revision and requires its manifest.
  Checkpoints record binary, scripts, database, vocabulary and deck hashes;
  evaluation rejects mismatched experiments.
- Updated chain/reveal protocols are parsed correctly. Monster-type and
  multi-attribute declarations, optional card selections, cancellation,
  unselecting cards, End Phase choices and end-phase discards are exposed.
- Visible card IDs and deck search labels are populated correctly. Both draw
  orders and opponent private/facedown features are filtered before policy input.
  Confirmed-card events are remembered separately, without exposing hand slots
  or unrelated cards. Some other reveal mechanisms remain conservatively hidden.
- Legal choices are masked before sampling. Native overflow is reported
  explicitly, so oversized action lists fail rather than silently dropping
  choices. Very large combination spaces or other unsupported upstream prompt
  paths can still require additional implementation.
- Decision caps are truncations, excluded from training and completed-game win
  rates. Evaluation alternates seats and preserves truncation counts.

The verified backend is the current Fluorohydride YGOPro core with its matching
mycard scripts/database, not EDOPro's distinct API. The core runs MR5; TCG deck
eligibility is enforced separately. These integration games do not certify
every TCG-specific timing/ruling difference, every card interaction, or playing
strength. Broader scenario tests and ruling audits remain research work.

## Tests

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
docker run --rm --entrypoint python --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source scripts/run-audit.py
docker run --rm --entrypoint python --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source scripts/engine-integration.py --output runs/integration-new
docker run --rm --entrypoint python --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source scripts/engine-regression.py
```

The default Windows run skips native scenarios; the Docker audit command runs
all 15 and saves a report. Pipeline unit tests use a synthetic fixture. The integration command uses
the unchanged, eligibility-checked Elfnote deck. The regression command uses
upstream's historical Blue-Eyes list, including Crossout Designator, solely to
test engine prompts; its results are labeled historical engine regression and
are not current-TCG experiments.

## Eligibility snapshots

The default snapshot freezes the official October 19, 2026 banlist and available
TCG release metadata as of October 28. Missing or future release dates fail
validation. Release dates come from YGOPRODeck, a secondary source.

```bash
python -m yugioh_gto.cli refresh-legality --as-of 2026-10-28
```

For a later date, provide its applicable official `--banlist-url`, a new
`--output`, and matching `--as-of`/`--snapshot` arguments in subsequent runs.
Preserve old snapshots for existing experiments. This check is not tournament
certification. `data/preflight-kewl-tune.json` records the source engine's updated
card coverage and remaining banlist restriction. `data/preflight-windows.json`
describes the historical Windows binary setup; use Docker preflight for current
native compatibility.

Sources: [TCG list](https://www.yugioh-card.com/en/limited/list_2026-10-19/),
[YGOPRODeck API](https://ygoprodeck.com/api-guide/),
[ygo-agent](https://github.com/sbl1996/ygo-agent),
[YGOPro core](https://github.com/Fluorohydride/ygopro-core).
The source bundle retains upstream licenses and notices.
