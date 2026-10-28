"""Complete, weights-only-loadable training checkpoints at update boundaries."""
from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
import random
import pickle
import shutil
import sys

import numpy as np
import torch

FORMAT_VERSION = 2


def trainer_fingerprint():
    digest = hashlib.sha256()
    for name in ("training.py", "model.py", "engine.py", "checkpoint.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def runtime_signature(device):
    result = {"device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
              "python": list(sys.version_info[:3]), "threads": torch.get_num_threads(),
              "interop_threads": torch.get_num_interop_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if str(device).startswith("cuda"):
        result.update(cuda=torch.version.cuda, cuda_devices=torch.cuda.device_count(),
                      gpu=torch.cuda.get_device_name(device),
                      capability=list(torch.cuda.get_device_capability(device)),
                      cudnn_deterministic=torch.backends.cudnn.deterministic,
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    return result


def capture_rng(rng, device):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy_global": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "sampler": copy.deepcopy(rng.bit_generator.state),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else []}


def restore_rng(state, rng):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy_global"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    rng.bit_generator.state = copy.deepcopy(state["sampler"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def read_resume(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Cannot read resume checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != FORMAT_VERSION or "resume" not in checkpoint:
        raise ValueError("This legacy checkpoint lacks complete resume state; it cannot resume exactly")
    required = {"rng", "environment", "opponent_pool", "identity", "runtime", "trainer_sha256"}
    if not isinstance(checkpoint["resume"], dict) or not required.issubset(checkpoint["resume"]) or not checkpoint["resume"]["opponent_pool"]:
        raise ValueError("Incomplete training checkpoint resume state")
    if not {"optimizer", "model", "spec", "settings", "iteration", "metadata"}.issubset(checkpoint):
        raise ValueError("Incomplete training checkpoint")
    return checkpoint


def validate_resume(checkpoint, settings, identity, device):
    resume = checkpoint["resume"]
    if resume["identity"] != identity:
        differences = sorted(key for key in set(resume["identity"]) | set(identity)
                             if resume["identity"].get(key) != identity.get(key))
        raise ValueError(f"Checkpoint experiment mismatch: {', '.join(differences)}")
    saved = {k: v for k, v in checkpoint["settings"].items() if k != "iterations"}
    current = {k: v for k, v in asdict(settings).items() if k != "iterations"}
    if saved != current:
        differences = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise ValueError(f"Checkpoint training configuration mismatch: {', '.join(differences)}")
    if settings.iterations <= checkpoint["iteration"]:
        raise ValueError("--iterations is the total target and must exceed the saved iteration")
    if resume["trainer_sha256"] != trainer_fingerprint():
        raise ValueError("Checkpoint trainer code mismatch; exact resume requires the same implementation")
    if resume["runtime"] != runtime_signature(device):
        raise ValueError("Checkpoint runtime mismatch; use the same device, versions, thread counts and determinism settings")


def save_training(path, policy, optimizer, iteration, metadata, settings, *, rng, env, pool, identity, device):
    checkpoint = dict(format_version=FORMAT_VERSION, model=policy.state_dict(), spec=policy.spec,
                      optimizer=optimizer.state_dict(), iteration=iteration,
                      settings=asdict(settings), metadata=metadata,
                      resume={"rng": capture_rng(rng, device), "environment": env.state_dict(),
                              "opponent_pool": [{"name": name, "model": opponent.state_dict()}
                                                for name, opponent in pool],
                              "identity": identity, "runtime": runtime_signature(device),
                              "trainer_sha256": trainer_fingerprint()})
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def archive_latest(latest, target):
    temporary = Path(target).with_suffix(".tmp")
    shutil.copyfile(latest, temporary)
    temporary.replace(target)
