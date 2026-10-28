"""Compare uninterrupted versus checkpoint-resumed real Elfnote training on CPU."""
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left == right


def records(directory):
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        row.pop("seconds", None)
    return [r for r in rows if r["event"] in ("game", "update")]


def main():
    import argparse
    import torch
    from yugioh_gto.checkpoint import read_resume
    from yugioh_gto.deck import Deck
    from yugioh_gto.engine import YgoEnvironment
    from yugioh_gto.legality import validate
    from yugioh_gto.training import Settings, train, digest
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/resume-verification")
    args = parser.parse_args()
    output = root / args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Resume verification output exists; preserve it and use a clean workspace")
    engine = root / "vendor/ygo-agent"
    deck_path = root / "decks/elfnote.ydk"
    snapshot_path = root / "data/tcg-2026-10-28.json"
    deck = Deck.read(deck_path)
    errors = validate(deck, json.loads(snapshot_path.read_text()), "2026-10-28")
    if errors:
        raise ValueError("; ".join(errors))
    identity = {"purpose": "REAL_ELFNOTE_RESUME_EQUIVALENCE", "deck_sha256": digest(deck_path),
                "snapshot_sha256": digest(snapshot_path),
                "native_binary_sha256": digest(engine / "ygoenv/ygoenv/ygopro/ygopro_ygoenv.so"),
                "engine_bundle": json.loads((engine / "integration.json").read_text())}
    settings = Settings(iterations=3, games_per_iteration=2, hidden=32,
                        save_every=1, pool_size=2, ppo_epochs=1)
    def run(name, config, resume=None):
        env = YgoEnvironment(engine, deck, seed=config.seed, max_steps=config.max_decisions)
        try:
            train(env, output / name, config, identity, device="cpu", resume=resume)
        finally:
            env.close()
    run("uninterrupted", settings)
    run("interrupted", replace(settings, iterations=2))
    run("resumed", settings, output / "interrupted/latest.pt")
    a = read_resume(output / "uninterrupted/latest.pt")
    b = read_resume(output / "resumed/latest.pt")
    checks = {"model": equal(a["model"], b["model"]),
              "optimizer": equal(a["optimizer"], b["optimizer"]),
              "random_streams": equal(a["resume"]["rng"], b["resume"]["rng"]),
              "native_environment_rng": equal(a["resume"]["environment"], b["resume"]["environment"]),
              "opponent_pool": equal(a["resume"]["opponent_pool"], b["resume"]["opponent_pool"]),
              "game_and_update_logs": records(output / "uninterrupted") == records(output / "interrupted") + records(output / "resumed")}
    report = {"passed": all(checks.values()), "checks": checks, "iterations": 3,
              "games_executed": 12, "device": "cpu", "identity": identity}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
