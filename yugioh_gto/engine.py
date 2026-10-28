"""Real ygoenv adapter. A synthetic environment is only available to tests."""
import os
from pathlib import Path
import sys
import time
import multiprocessing
import numpy as np

from .deck import supported_codes


def sanitize(obs):
    obs = {k: v.copy() for k, v in obs.items()}
    cards = obs["cards_"]
    # ygoenv location IDs are assigned in ascending core location order:
    # deck=1, hand=2, monster=3, spell=4, grave=5, banished=6, extra=7.
    # Hide our draw order and all opponent private zones conservatively,
    # including temporarily revealed cards; retain only location/controller.
    private = (cards[:, 2] == 1) | ((cards[:, 4] == 1) & np.isin(cards[:, 2], [2, 7]))
    locations, controllers = cards[private, 2].copy(), cards[private, 4].copy()
    cards[private] = 0
    cards[private, 2], cards[private, 4] = locations, controllers
    facedown = (cards[:, 4] == 1) & np.isin(cards[:, 5], [2, 6, 7])
    cards[facedown, :2] = 0
    cards[facedown, 7:] = 0
    return obs


class _NativeYgoEnvironment:
    def __init__(self, root, deck, seed=0, max_options=256, max_steps=4000):
        root = Path(root).resolve()
        self.root = root
        missing = set(deck.active) - supported_codes(root / "scripts/code_list.txt")
        if missing:
            raise RuntimeError(f"Engine card support missing: {sorted(missing)}. Cannot train this deck.")
        sys.path.insert(0, str(root / "ygoenv"))
        try:
            import ygoenv
            from ygoenv.ygopro import init_module
            from ygoenv.ygopro import ygopro_ygoenv as native
        except ImportError as exc:
            raise RuntimeError("Native ygoenv is unavailable. See README Linux setup.") from exc
        if getattr(native, "integration_revision", None) != 1:
            raise RuntimeError("Outdated native binary: rebuild the source integration Docker image")
        self.native = native
        work = root / "scripts"
        if not (work / "script/utility.lua").exists():
            raise RuntimeError("Engine card scripts have not been installed")
        # Upstream loads scripts by relative path. Keep this process in its engine workdir.
        self.previous_cwd = Path.cwd()
        os.chdir(work)
        active = work / f"_mirror_{os.getpid()}.ydk"
        self.active_deck = active
        deck.write_active(active)
        init_module(str(root / "assets/locale/en/cards.cdb"),
                    str(work / "code_list.txt"), {"Mirror": str(active)})
        self.env = ygoenv.make("YGOPro-v1", env_type="gymnasium", num_envs=1,
                              num_threads=1, seed=seed, deck1="Mirror",
                              deck2="Mirror", play_mode="self", player=0,
                              max_options=max_options, n_history_actions=32,
                              greedy_reward=False, oppo_info=False, async_reset=False,
                              max_steps=max_steps, timeout=120)
        self.max_options = max_options
        self.max_steps = max_steps
        self.steps = 0

    def unpack(self, obs, info):
        self.check_errors(info)
        count = int(info["num_options"][0])
        if bool(info.get("action_overflow", [False])[0]):
            raise RuntimeError("Action capacity reached: refusing potentially truncated legal actions")
        if count < 1:
            raise RuntimeError("Engine returned no legal action")
        return sanitize({k: v[0] for k, v in obs.items()}), int(info["to_play"][0]), count

    def check_errors(self, info):
        errors = self.native.integration_errors()
        if errors or bool(info.get("integration_error", [False])[0]):
            raise RuntimeError("Native engine error: " + ("; ".join(errors[-5:]) or "timeout or integration failure"))

    def reset(self):
        self.steps = 0
        self.native.clear_prompt_counts()
        obs, info = self.env.reset()
        return self.unpack(obs, info)

    def step(self, action):
        started = time.monotonic()
        obs, reward, terminated, truncated, info = self.env.step(np.array([action], dtype=np.int32))
        self.check_errors(info)
        self.steps += 1
        # Upstream adjudicates by life points at its step cap. Do not call that a win.
        if bool(truncated[0]) or self.steps >= self.max_steps or time.monotonic() - started >= 119:
            return None, 0.0, True, {"truncated": True}
        done = bool(terminated[0])
        if done:
            return None, float(reward[0]), True, {"truncated": False,
                    "win_reason": int(info["win_reason"][0]),
                    "engine_diagnostics": {"script_errors": len(self.native.integration_errors()),
                                           "prompt_counts": self.native.prompt_counts()}}
        return self.unpack(obs, info), float(reward[0]), False, {}

    def close(self):
        close = getattr(self.env, "close", None)
        if close:
            close()
        self.active_deck.unlink(missing_ok=True)
        os.chdir(self.previous_cwd)


def _engine_worker(connection, options):
    env = None
    try:
        env = _NativeYgoEnvironment(**options)
        connection.send((True, None))
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            if command == "reset":
                result = env.reset()
            elif command == "step":
                result = env.step(value)
            else:
                raise RuntimeError(f"Unknown engine command: {command}")
            connection.send((True, result))
    except (Exception, EOFError) as exc:
        try:
            connection.send((False, str(exc)))
        except (OSError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class YgoEnvironment:
    """Isolate native crashes and global engine state from the learner process."""
    def __init__(self, root, deck, seed=0, max_options=256, max_steps=4000, worker_timeout=180):
        root = Path(root).resolve()
        missing = set(deck.active) - supported_codes(root / "scripts/code_list.txt")
        if missing:
            raise RuntimeError(f"Engine card support missing: {sorted(missing)}. Cannot train this deck.")
        self.timeout = worker_timeout
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_engine_worker, args=(child, dict(
            root=str(root), deck=deck, seed=seed, max_options=max_options, max_steps=max_steps)), daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive()
        except Exception:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(self.timeout):
            self.close()
            raise RuntimeError("Native engine worker timed out; no game outcome recorded")
        try:
            success, result = self.connection.recv()
        except (EOFError, OSError) as exc:
            self.process.join(timeout=1)
            raise RuntimeError(f"Native engine worker exited (code {self.process.exitcode}); no game outcome recorded") from exc
        if not success:
            raise RuntimeError(result)
        return result

    def _request(self, command, value=None):
        try:
            self.connection.send((command, value))
        except (OSError, EOFError) as exc:
            raise RuntimeError("Native engine worker disconnected") from exc
        return self._receive()

    def reset(self):
        return self._request("reset")

    def step(self, action):
        return self._request("step", action)

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(("close", None))
            except (OSError, EOFError):
                pass
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.connection.close()
