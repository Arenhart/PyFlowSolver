"""Build the cluster (if needed), run a distributed Dask array inversion, tear down."""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import docker

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_NAMES = [
    "hpc_emulator-master",
    "hpc_emulator-worker1",
    "hpc_emulator-worker2",
]
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
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ret = container.exec_run("sinfo -h -o %T")
        states = ret.output.decode().strip().splitlines()
        if states and all(s == "idle" for s in states):
            return True
        time.sleep(1)
    return False


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
        # ---- 3. Wait for SLURM workers ------------------------------------
        print("Waiting for SLURM workers to become ready...")
        master = client.containers.get("master")
        if not wait_for_workers(master):
            print("ERROR: Workers did not become ready in time.", file=sys.stderr)
            sys.exit(1)
        print("Cluster is ready.\n")

        # ---- 4. Copy dask job script into master --------------------------
        job_script = PROJECT_DIR / "dask_job.py"
        subprocess.run(
            ["docker", "cp", str(job_script), "master:/tmp/dask_job.py"],
            check=True,
        )

        # ---- 5. Run the dask job ------------------------------------------
        print("Running Dask inversion job...\n")
        ret = master.exec_run(
            ["python", "-u", "/tmp/dask_job.py"],
            demux=True,
        )
        stdout = (ret.output[0] or b"").decode()
        stderr = (ret.output[1] or b"").decode()

        if stdout:
            print(stdout)
        if stderr:
            print("--- stderr ---", file=sys.stderr)
            print(stderr, file=sys.stderr)

        if ret.exit_code != 0:
            print(f"ERROR: Dask job failed (exit {ret.exit_code}).", file=sys.stderr)
            sys.exit(1)

    finally:
        # ---- 6. Tear down -------------------------------------------------
        print("Tearing down cluster...")
        compose("down")
        print("Done.")


if __name__ == "__main__":
    main()
