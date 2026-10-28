from dataclasses import dataclass, asdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import torch

from .model import Policy, pack


@dataclass
class Settings:
    iterations: int = 100
    games_per_iteration: int = 8
    ppo_epochs: int = 3
    learning_rate: float = 0.0001
    clip: float = 0.2
    entropy: float = 0.01
    hidden: int = 128
    seed: int = 7
    save_every: int = 5
    pool_size: int = 8
    max_decisions: int = 4000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_scripts(directory):
    checksum = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(directory.rglob("*.lua")):
        checksum.update(path.relative_to(directory).as_posix().encode())
        checksum.update(b"\0")
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Results:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path / "results.sqlite3")
        self.db.execute("""CREATE TABLE IF NOT EXISTS games
            (phase TEXT, iteration INTEGER, game INTEGER, candidate_seat INTEGER,
             opponent TEXT, winner INTEGER, truncated INTEGER, decisions INTEGER,
             seconds REAL, reason INTEGER)""")

    def game(self, phase, iteration, index, seat, opponent, result):
        row = (phase, iteration, index, seat, opponent, result["winner"],
               int(result["truncated"]), result["decisions"], result["seconds"], result.get("reason"))
        self.db.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?)", row)
        self.db.commit()
        self.log({"event": "game", "phase": phase, "iteration": iteration,
                  "game": index, "candidate_seat": seat, "opponent": opponent, **result})

    def log(self, event):
        with (self.path / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")

    def export(self):
        cursor = self.db.execute("SELECT * FROM games")
        with (self.path / "games.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([c[0] for c in cursor.description])
            writer.writerows(cursor)

    def close(self):
        self.export()
        self.db.close()


def play(env, policies, rng, max_decisions, learn_seats=()):
    """Keep recurrent memory separate for each player. Rewards are actor-relative."""
    state = [None, None]
    trajectories = {seat: [] for seat in learn_seats}
    observation, seat, count = env.reset()
    started = time.monotonic()
    telemetry = [{"special_summon_choices": 0, "normal_summon_choices": 0,
                  "activation_choices": 0, "attack_choices": 0} for _ in range(2)]
    for decision in range(1, max_decisions + 1):
        features, actions = pack(observation)
        policy = policies[seat]
        if policy is None:
            action, logp, value = int(rng.integers(count)), 0., 0.
        else:
            action, logp, value, state[seat] = policy.act(features, actions, count, state[seat])
        if not 0 <= action < count:
            raise RuntimeError("Policy selected an illegal action")
        raw_actions = observation["actions_"]
        if raw_actions.shape[-1] == 12:
            action_type = int(raw_actions[action, 4])
            key = {3: "special_summon_choices", 4: "normal_summon_choices",
                   8: "activation_choices", 6: "attack_choices", 7: "attack_choices"}.get(action_type)
            if key:
                telemetry[seat][key] += 1
        global_state = observation["global_"]
        if len(global_state) == 23:
            own_resources = sum(int(global_state[i]) for i in (9, 10, 11))
            opponent_resources = sum(int(global_state[i]) for i in (16, 17, 18))
            telemetry[seat]["last_observed_hand_board_advantage"] = own_resources - opponent_resources
            telemetry[seat]["last_observed_lp"] = int(global_state[0]) * 256 + int(global_state[1])
            telemetry[seat]["last_observed_opponent_lp"] = int(global_state[2]) * 256 + int(global_state[3])
        if seat in trajectories:
            trajectories[seat].append((features.copy(), actions.copy(), count, action, logp, value))
        actor = seat
        next_state, reward, done, info = env.step(action)
        if done:
            truncated = bool(info.get("truncated", False))
            winner = None if truncated or reward == 0 else (actor if reward > 0 else 1 - actor)
            return trajectories, dict(winner=winner, truncated=truncated,
                                      decisions=decision, seconds=time.monotonic() - started,
                                      reason=info.get("win_reason"), diagnostics=telemetry,
                                      engine_diagnostics=info.get("engine_diagnostics", {}))
        observation, seat, count = next_state
    return trajectories, dict(winner=None, truncated=True, decisions=max_decisions,
                              seconds=time.monotonic() - started, diagnostics=telemetry)


def ppo_update(policy, optimizer, episodes, settings):
    """Replay full per-player sequences from zero memory on each PPO epoch.

    Undiscounted terminal returns make either player's target +1/-1. No
    cross-player GAE or stale recurrent states are used. Truncated games
    are excluded by the collector rather than inventing terminal labels.
    """
    device = next(policy.parameters()).device
    losses, kls = [], []
    for _ in range(settings.ppo_epochs):
        for trajectory, target in episodes:
            if not trajectory:
                continue
            features, actions, counts, chosen, old_logp, old_values = zip(*trajectory)
            obs = torch.tensor(np.stack(features), device=device)[None]
            acts = torch.tensor(np.stack(actions), device=device)[None]
            mask = torch.arange(acts.shape[-2], device=device)[None, :] < torch.tensor(counts, device=device)[:, None]
            chosen = torch.tensor(chosen, device=device)[None]
            old_logp = torch.tensor(old_logp, device=device)[None]
            advantages = target - torch.tensor(old_values, device=device)[None]
            dist, values, _ = policy(obs, acts, mask[None])
            logp = dist.log_prob(chosen)
            ratio = (logp - old_logp).exp()
            surrogate = torch.minimum(ratio * advantages,
                                      ratio.clamp(1 - settings.clip, 1 + settings.clip) * advantages)
            loss = -surrogate.mean() + .5 * (values - target).square().mean() - settings.entropy * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
            kls.append(float((old_logp - logp).mean().detach()))
    return {"loss": float(np.mean(losses)) if losses else 0.,
            "approx_kl": float(np.mean(kls)) if kls else 0., "training_sequences": len(episodes)}


def save_checkpoint(path, policy, optimizer, iteration, metadata, settings):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(model=policy.state_dict(), spec=policy.spec,
                    optimizer=optimizer.state_dict(), iteration=iteration,
                    settings=asdict(settings), metadata=metadata), temporary)
    temporary.replace(path)


def load_policy(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    policy = Policy(**checkpoint["spec"]).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy, checkpoint


def train(env, output, settings, metadata, device="cpu"):
    seed_everything(settings.seed)
    rng = np.random.default_rng(settings.seed)
    results = Results(output)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(results.path / "tensorboard"))
    try:
        observation, _, _ = env.reset()
        features, actions = pack(observation)
        policy = Policy(len(features), actions.shape[-1], settings.hidden).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
        opponent_pool = [("initial", copy.deepcopy(policy).eval())]
        metadata = {**metadata, "settings": asdict(settings), "device": device,
                    "observation_size": len(features), "action_size": actions.shape[-1]}
        (results.path / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for iteration in range(1, settings.iterations + 1):
            episodes, wins, completed, truncated = [], 0, 0, 0
            for game in range(settings.games_per_iteration):
                seat = game % 2
                label, opponent = opponent_pool[int(rng.integers(len(opponent_pool)))]
                policies = [opponent, opponent]
                policies[seat] = policy
                trajectories, result = play(env, policies, rng, settings.max_decisions, [seat])
                results.game("train", iteration, game, seat, label, result)
                if result["truncated"]:
                    truncated += 1
                    continue
                completed += 1
                wins += result["winner"] == seat
                target = 0. if result["winner"] is None else (1. if result["winner"] == seat else -1.)
                episodes.append((trajectories[seat], target))
            metrics = ppo_update(policy, optimizer, episodes, settings)
            metrics.update(iteration=iteration, completed=completed, truncated=truncated,
                           win_rate=wins / completed if completed else 0.)
            results.log({"event": "update", **metrics})
            for key, value in metrics.items():
                writer.add_scalar(key, value, iteration)
            writer.flush()
            print(json.dumps(metrics), flush=True)
            if not completed:
                raise RuntimeError("No completed games in rollout; refusing to train on artificial outcomes")
            if iteration % settings.save_every == 0 or iteration == settings.iterations:
                name = f"checkpoint-{iteration:06d}"
                save_checkpoint(results.path / f"{name}.pt", policy, optimizer, iteration, metadata, settings)
                opponent_pool.append((name, copy.deepcopy(policy).eval()))
                opponent_pool = opponent_pool[-settings.pool_size:]
        return policy
    finally:
        writer.close()
        results.close()


def wilson(wins, total):
    if not total:
        return None
    z = 1.95996398454
    p = wins / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    width = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [max(0., center-width), min(1., center+width)]


def evaluate(env, candidate, opponent, output, games=100, seed=123, max_decisions=4000):
    if games < 2 or games % 2:
        raise ValueError("Evaluation requires a positive even number of games for balanced seats")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    results = Results(output)
    counts = [{"wins": 0, "losses": 0, "draws": 0, "truncated": 0} for _ in range(2)]
    try:
        for game in range(games):
            seat = game % 2
            policies = [opponent, opponent]
            policies[seat] = candidate
            _, result = play(env, policies, rng, max_decisions)
            results.game("eval", 0, game, seat, "random" if opponent is None else "checkpoint", result)
            key = "truncated" if result["truncated"] else "draws" if result["winner"] is None else "wins" if result["winner"] == seat else "losses"
            counts[seat][key] += 1
        for stats in counts:
            total = stats["wins"] + stats["losses"] + stats["draws"]
            stats["completed"] = total
            stats["win_rate"] = stats["wins"] / total if total else None
            stats["win_rate_95_ci"] = wilson(stats["wins"], total)
        summary = {"games_requested": games, "going_first": counts[0], "going_second": counts[1]}
        (results.path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        results.close()
