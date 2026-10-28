"""Fetch only pinned source snapshots, not large card-database history."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
lock = json.loads((Path(__file__).parent / "lock.json").read_text())
repositories = {
    "environment": ("sbl1996/ygo-agent", "ygo-agent"),
    "core": ("Fluorohydride/ygopro-core", "ygopro-core-current"),
    "scripts": ("mycard/ygopro-scripts", "ygopro-scripts-current"),
    "database": ("mycard/ygopro-database", "ygopro-database-current"),
}
for component, (repo, folder) in repositories.items():
    directory = root / "vendor" / folder
    directory.mkdir(parents=True, exist_ok=True)
    def git(*args):
        subprocess.run(["git", "-C", str(directory), *args], check=True)
    if not (directory / ".git").exists():
        git("init")
        git("remote", "add", "origin", f"https://github.com/{repo}.git")
    git("fetch", "--depth", "1", "origin", lock[component])
    git("checkout", "--detach", lock[component])
