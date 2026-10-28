import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

from .deck import Deck, supported_codes
from .legality import validate, refresh, BANLIST

ROOT = Path(__file__).resolve().parent.parent


def preflight(args):
    deck = Deck.read(args.deck)
    errors = deck.errors()
    snapshot_path = Path(args.snapshot)
    snapshot = None
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        errors += validate(deck, snapshot, args.as_of)
    else:
        errors.append("TCG snapshot missing; run refresh-legality before real training")
    engine = Path(args.engine).resolve()
    codes = engine / "scripts/code_list.txt"
    missing = []
    if not codes.exists():
        errors.append("Engine source missing; run scripts/fetch-engine.ps1 or scripts/fetch-engine.sh")
    else:
        missing = sorted(set(deck.active) - supported_codes(codes))
        if missing:
            errors.append(f"Engine supported-card list missing {len(missing)} active card passwords: {missing}")
    binary = engine / "ygoenv/ygoenv/ygopro/ygopro_ygoenv.so"
    if not binary.exists():
        errors.append("Linux ygoenv native binary missing; see README setup")
    if not (engine / "integration.json").exists():
        errors.append("Source integration manifest missing; rebuild the source engine")
    if not (engine / "scripts/script/utility.lua").exists():
        errors.append("Card scripts missing")
    if not (engine / "assets/locale/en/cards.cdb").exists():
        errors.append("Engine card database missing")
    report = {"main": len(deck.main), "extra": len(deck.extra), "side_ignored": len(deck.side),
              "as_of": args.as_of, "ready": not errors, "errors": sorted(set(errors)),
              "unsupported_cards": [{"password": code,
                  "name": snapshot["cards"].get(str(code), {}).get("name", "Unknown") if snapshot else "Unknown"}
                  for code in missing]}
    return deck, report


def main():
    parser = argparse.ArgumentParser(description="Yu-Gi-Oh! recurrent self-play pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    refresh_parser = sub.add_parser("refresh-legality")
    refresh_parser.add_argument("--as-of", default="2026-10-28")
    refresh_parser.add_argument("--banlist-url", default=BANLIST)
    refresh_parser.add_argument("--output", default=str(ROOT / "data/tcg-2026-10-28.json"))
    for name in ("preflight", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--deck", default=str(ROOT / "decks/elfnote.ydk"))
        command.add_argument("--engine", default=str(ROOT / "vendor/ygo-agent"))
        command.add_argument("--snapshot", default=str(ROOT / "data/tcg-2026-10-28.json"))
        command.add_argument("--as-of", default="2026-10-28")
        if name != "preflight":
            command.add_argument("--output", default=str(ROOT / f"runs/{name}"))
            command.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
            command.add_argument("--seed", type=int, default=7)
            command.add_argument("--max-decisions", type=int, default=4000)
        if name == "train":
            command.add_argument("--iterations", type=int, default=100)
            command.add_argument("--games-per-iteration", type=int, default=8)
        if name == "evaluate":
            command.add_argument("--candidate", required=True)
            command.add_argument("--opponent", help="Checkpoint; omit for random legal policy")
            command.add_argument("--games", type=int, default=100)
    args = parser.parse_args()
    try:
        if args.command == "refresh-legality":
            snapshot = refresh(args.output, args.as_of, args.banlist_url)
            print(f"Saved {len(snapshot['cards'])} passwords to {args.output}")
            return 0
        date.fromisoformat(args.as_of)
        deck, report = preflight(args)
        print(json.dumps(report, indent=2))
        if args.command == "preflight":
            return 0 if report["ready"] else 2
        if not report["ready"]:
            return 2
        import torch
        from .engine import YgoEnvironment
        from .training import Settings, train, evaluate, load_policy, digest, digest_scripts
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
        output = Path(args.output).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose a new run directory")
        metadata = {"deck_sha256": digest(args.deck), "snapshot_sha256": digest(args.snapshot),
                    "as_of": args.as_of, "engine_commit": subprocess.check_output(
                        ["git", "-C", args.engine, "rev-parse", "HEAD"], text=True).strip(),
                    "code_list_sha256": digest(Path(args.engine) / "scripts/code_list.txt"),
                    "database_sha256": digest(Path(args.engine) / "assets/locale/en/cards.cdb"),
                    "scripts_sha256": digest_scripts(Path(args.engine) / "scripts/script"),
                    "native_binary_sha256": digest(Path(args.engine) / "ygoenv/ygoenv/ygopro/ygopro_ygoenv.so"),
                    "engine_bundle": json.loads((Path(args.engine) / "integration.json").read_text()),
                    "backend": "ygoenv", "torch_version": str(torch.__version__)}
        if args.command == "train" and (args.iterations < 1 or args.games_per_iteration < 2 or args.games_per_iteration % 2):
            raise ValueError("Use positive iterations and an even games-per-iteration >= 2")
        env = YgoEnvironment(args.engine, deck, args.seed, max_steps=args.max_decisions)
        try:
            if args.command == "train":
                settings = Settings(iterations=args.iterations, games_per_iteration=args.games_per_iteration,
                                    seed=args.seed, max_decisions=args.max_decisions)
                train(env, output, settings, metadata, device)
            else:
                candidate, checkpoint = load_policy(args.candidate, device)
                opponent = load_policy(args.opponent, device)[0] if args.opponent else None
                for label, saved in (("candidate", checkpoint), ("opponent", load_policy(args.opponent, device)[1] if args.opponent else checkpoint)):
                    for key in ("deck_sha256", "snapshot_sha256", "code_list_sha256", "engine_commit",
                                "database_sha256", "scripts_sha256", "native_binary_sha256"):
                        if saved["metadata"].get(key) != metadata[key]:
                            raise ValueError(f"{label} checkpoint {key} does not match this experiment")
                evaluate(env, candidate, opponent, output, args.games, args.seed, args.max_decisions)
                metadata.update(candidate=str(Path(args.candidate).resolve()), candidate_sha256=digest(args.candidate),
                                opponent_sha256=digest(args.opponent) if args.opponent else None, seed=args.seed)
                (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        finally:
            env.close()
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
