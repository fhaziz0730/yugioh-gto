"""Historical engine regression only; these games are not current-TCG experiments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import json
    import numpy as np
    from yugioh_gto.deck import Deck
    from yugioh_gto.engine import YgoEnvironment
    from yugioh_gto.training import play
    root = Path(__file__).resolve().parent.parent
    deck = Deck.read(root / "vendor/ygo-agent/assets/deck/BlueEyes.ydk")
    env = YgoEnvironment(root / "vendor/ygo-agent", deck, seed=31, max_options=512)
    results = []
    try:
        rng = np.random.default_rng(31)
        for _ in range(8):
            _, game = play(env, [None, None], rng, 4000)
            if game["truncated"]:
                raise RuntimeError("Regression game did not finish")
            results.append(game)
    finally:
        env.close()
    declarations = sum(g["engine_diagnostics"]["prompt_counts"].get("announce_card", 0) for g in results)
    if not declarations:
        raise RuntimeError("Regression failed to exercise card declarations")
    report = dict(purpose="HISTORICAL_ENGINE_REGRESSION_ONLY", games=results,
                  card_declarations=declarations)
    output = root / "runs/engine-regression"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"games_completed": len(results), "card_declarations": declarations,
                      "script_errors": sum(g["engine_diagnostics"]["script_errors"] for g in results)}))


if __name__ == "__main__":
    main()
