"""Build the cluster (if needed), submit a hello-world job, print output, tear down."""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import docker

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_NAMES = [
    "hpc_emulator-master",
    "hpc_emulator-worker1",
    "hpc_emulator-worker2",
]
# Files that, when changed, should trigger a rebuild.
BUILD_INPUTS = [
    "Dockerfile",
    "entrypoint.sh",
    "slurm.conf",
    "cgroup.conf",
    "munge.key",
]
COMPOSE_CMD = ["docker-compose"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a docker-compose command in the project directory."""
    return subprocess.run(
        [*COMPOSE_CMD, *args],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def images_need_build(client: docker.DockerClient) -> bool:
    """Return True if any image is missing or older than the build inputs."""
    latest_source_mtime: float = 0
    for name in BUILD_INPUTS:
        path = PROJECT_DIR / name
        if path.exists():
            latest_source_mtime = max(latest_source_mtime, path.stat().st_mtime)

    for name in IMAGE_NAMES:
        try:
            img = client.images.get(f"{name}:latest")
        except docker.errors.ImageNotFound:
            print(f"  Image {name}:latest not found — build required.")
            return True

        created = datetime.fromisoformat(img.attrs["Created"]).timestamp()
        if created < latest_source_mtime:
            print(f"  Image {name}:latest is older than source files — build required.")
            return True

    return False


def wait_for_workers(container, timeout: int = 60) -> bool:
    """Poll sinfo until both workers show as idle."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ret = container.exec_run("sinfo -h -o %T")
        states = ret.output.decode().strip().splitlines()
        if states and all(s == "idle" for s in states):
            return True
        time.sleep(1)
    return False


def exec_master(client: docker.DockerClient, cmd: str | list[str]) -> str:
    """Run a command inside the master container and return stdout."""
    master = client.containers.get("master")
    ret = master.exec_run(cmd)
    return ret.output.decode()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = docker.from_env()

    # ---- 1. Build images if needed ----------------------------------------
    print("Checking images...")
    if images_need_build(client):
        print("Building images...")
        result = compose("build")
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(1)
        print("Build complete.")
    else:
        print("Images are up to date.")

    # ---- 2. Start the cluster ---------------------------------------------
    print("Starting cluster...")
    result = compose("up", "-d")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    try:
        # ---- 3. Wait for workers to be ready ------------------------------
        print("Waiting for workers to become ready...")
        master = client.containers.get("master")
        if not wait_for_workers(master):
            print("ERROR: Workers did not become ready in time.", file=sys.stderr)
            print(exec_master(client, "sinfo"), file=sys.stderr)
            sys.exit(1)
        print("Cluster is ready.")
        print()

        # ---- 4. Submit hello-world job ------------------------------------
        #   Use `srun` directly so output streams back to the master.
        print("Running hello-world job on all workers...")
        ret = master.exec_run([
            "srun", "--nodes=2",
            "bash", "-c", 'echo "Hello World from $(hostname)"',
        ])
        output = ret.output.decode().strip()

        if ret.exit_code != 0:
            print(f"ERROR: Job failed (exit {ret.exit_code}).", file=sys.stderr)
            print(output, file=sys.stderr)
            sys.exit(1)

        print()
        print("=== Job Output ===")
        print(output)
        print("==================")

    finally:
        # ---- 7. Tear down -------------------------------------------------
        print()
        print("Tearing down cluster...")
        compose("down")
        print("Done.")


if __name__ == "__main__":
    main()
