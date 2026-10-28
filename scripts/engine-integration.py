"""Real modern-deck engine integration check; never bypass TCG eligibility."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import torch
    from yugioh_gto.deck import Deck
    from yugioh_gto.engine import YgoEnvironment
    from yugioh_gto.legality import validate
    from yugioh_gto.training import Settings, train, evaluate, load_policy
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", default="decks/elfnote.ydk")
    parser.add_argument("--output", default="runs/elfnote-source-integration")
    parser.add_argument("--games", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    deck = Deck.read(root / args.deck)
    snapshot = json.loads((root / "data/tcg-2026-10-28.json").read_text())
    errors = validate(deck, snapshot, "2026-10-28")
    if errors:
        raise RuntimeError("; ".join(errors))
    env = YgoEnvironment(root / "vendor/ygo-agent", deck, seed=17, max_options=512, max_steps=4000)
    output = (root / args.output).resolve()
    try:
        train(env, output / "train", Settings(iterations=1, games_per_iteration=args.games,
              hidden=32, ppo_epochs=1, save_every=1),
              {"backend": "source-ygoenv", "purpose": "REAL_ELFNOTE_ENGINE_INTEGRATION"})
        policy, _ = load_policy(output / "train/checkpoint-000001.pt")
        evaluate(env, policy, None, output / "eval", games=args.games)
    finally:
        env.close()


if __name__ == "__main__":
    main()
