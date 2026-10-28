# Resuming self-play training

Training now atomically replaces `latest.pt` after every completed PPO update.
It includes model and optimizer state, the ordered frozen-opponent pool,
Python/NumPy/PyTorch random streams, and the native engine's next-game RNG state.
No partial game or partially applied update is used as a resume point.

## Continue a run

From the project directory, with Docker Desktop running:

```powershell
docker run --rm --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source train --resume runs/elfnote-resumed-cli/latest.pt --iterations 4 --output runs/elfnote-next
```

The saved example is at iteration 2, so this command performs iterations 3 and
4. `--iterations` always means the total target, not a number of extra updates.
Choose a fresh output directory. The source run is preserved, and the new
manifest and JSONL log record its checkpoint path, hash and iteration.

Saved hyperparameters are restored unless explicitly supplied. Changing them
is rejected for exact resume; only the total target may change. Automatic device
selection preserves the saved device, and CPU thread counts are restored.

For a new run with a specified CPU thread count:

```powershell
docker run --rm --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source train --iterations 100 --games-per-iteration 8 --torch-threads 1 --output runs/elfnote-new --device cpu
```

## Checkpoint behavior

- `latest.pt` is written at initialization and after each complete update. An
  interrupted write leaves the previous valid checkpoint in place.
- `checkpoint-NNNNNN.pt` archives are created at `--save-every` intervals and
  at the requested final iteration.
- Opponent promotion follows the save interval, independently of when a run
  stops. Stopping between intervals does not add an extra opponent or change
  the resumed training schedule.
- Native engine state covers the next reset, including shuffle and duel seeds.
  After restoration the worker requires a reset before any action. It does not
  pretend to restore an in-progress board or recurrent trajectory.
- Logs in a continuation begin at the next saved iteration. If an interrupted
  source log contains later rollout rows, those rows were abandoned; include
  only source iterations up to the chosen checkpoint when combining histories.

## Compatibility checks

Resume requires matching deck/eligibility snapshots, engine assets and binary,
training configuration, trainer source fingerprint, and execution runtime.
Checks include model/input shape, Python/NumPy/PyTorch versions, device, thread
counts and determinism settings. The engine's resume RNG format is versioned.
Preserve the code and Docker image used for a long experiment.

Legacy checkpoints lack the random streams and complete opponent pool required
for exact continuation. They are rejected with a clear message rather than
silently restarted with fresh state. The pre-resume audit checkpoints remain
historical artifacts; use the new `latest.pt` files for continuation.

CUDA RNG states are stored for CUDA runs, but exact CUDA continuation has not
yet been benchmarked or verified. The bitwise equivalence checks below are CPU
results on a matching software/runtime stack.

## Executed verification

Seven resume tests cover restored weights/optimizer/pool/RNG state, stopping
between archive intervals, interruption during a rollout, atomic-write failure,
corrupt and legacy checkpoints, mismatched experiments/configuration/runtime/code,
and output/iteration safeguards. All passed, along with the 15 existing pipeline
tests and 15 native ruling/privacy scenarios.

The packaged native engine also ran a real Elfnote comparison:

- Uninterrupted: three updates, two games per update.
- Interrupted segment: two updates, then a fresh process resumed the third.
- All model tensors, optimizer tensors, RNG states, native shuffle state,
  frozen-opponent tensors/order, game outcomes and update metrics matched
  exactly. Elapsed-time fields and the resume-lineage log event are excluded.
- Twelve games were executed across the comparison's three segments.

Result: `runs/resume-verification-final/report.json`.

The normal CLI was separately verified: a two-game update saved
`runs/elfnote-resumable/latest.pt`, and a fresh CLI invocation restored it and
completed iteration 2 in `runs/elfnote-resumed-cli/latest.pt`.

To reproduce the real-engine comparison in a fresh output directory:

```powershell
docker run --rm --entrypoint python --mount "type=bind,source=${PWD}/runs,target=/app/runs" yugioh-gto:source scripts/verify-resume.py --output runs/resume-check-new
```
