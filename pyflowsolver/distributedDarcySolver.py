import subprocess
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

from pyflowsolver.darcySolver import DarcySolver

BOUNDARY_ESTIMATE = 0.5

DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker"
COMPOSE_CMD = ["docker-compose"]

BUILD_INPUTS = [
    "Dockerfile",
    "entrypoint.sh",
    "slurm.conf",
    "cgroup.conf",
    "munge.key",
]
IMAGE_NAMES = [
    "pyflowsolver-master",
    "pyflowsolver-worker1",
    "pyflowsolver-worker2",
]
# Also track pyflowsolver source for rebuild detection
SOURCE_DIR = Path(__file__).resolve().parent


class DistributedDarcySolver(DarcySolver):

    def __init__(self, n_partitions=2, pressure_estimate=None, **params):
        super().__init__(**params)
        self.n_partitions = n_partitions
        self.pressure_estimate = pressure_estimate
        self.partitions = None

    def partition_system(self):
        """Split the already-set CSR system into n_partitions independent sub-systems."""
        if self.a_sparse_array is None or self.b_array is None:
            raise Exception("Linear system not set. Call set_linear_system first.")
        self.partitions = partition_csr_system(
            self.a_sparse_array, self.b_array, self.n_partitions,
            pressure_estimate=self.pressure_estimate,
        )
        return self.partitions

    def solve_distributed(self):
        """
        Orchestrate distributed solve via Docker/SLURM/Dask cluster.

        1. Partition the system
        2. Start Docker cluster
        3. Serialize and copy partition data + job script into master
        4. Run the Dask job on the master
        5. Copy results back and reassemble
        """
        import docker

        if self.partitions is None:
            self.partition_system()

        client = docker.from_env()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Serialize partitions as .npz files
            for i, part in enumerate(self.partitions):
                np.savez(
                    tmpdir / f"partition_{i}.npz",
                    val=part["a_sparse_array"]["val"],
                    col_idx=part["a_sparse_array"]["col_idx"],
                    row_ptr=part["a_sparse_array"]["row_ptr"],
                    b_array=part["b_array"],
                    global_row_start=np.array([part["global_row_start"]]),
                    global_row_end=np.array([part["global_row_end"]]),
                )

            np.savez(
                tmpdir / "metadata.npz",
                n_partitions=np.array([self.n_partitions]),
                max_iterations=np.array([self.params["max_iterations"]]),
                target_error=np.array([self.params["target_error"]]),
            )

            # Build images only if source changed, then start cluster
            if _images_need_build(client):
                _compose("build")
            _compose("up", "-d")

            try:
                master = client.containers.get("master")

                # Wait for SLURM workers
                if not _wait_for_workers(master):
                    raise Exception("SLURM workers did not become ready in time.")

                # Copy job script into master
                job_script = DOCKER_DIR / "distributed_cg_job.py"
                subprocess.run(
                    ["docker", "cp", str(job_script), "master:/tmp/distributed_cg_job.py"],
                    check=True,
                )

                # Copy partition data into master
                for i in range(self.n_partitions):
                    src = tmpdir / f"partition_{i}.npz"
                    subprocess.run(
                        ["docker", "cp", str(src), f"master:/tmp/partition_{i}.npz"],
                        check=True,
                    )
                subprocess.run(
                    ["docker", "cp", str(tmpdir / "metadata.npz"), "master:/tmp/metadata.npz"],
                    check=True,
                )

                # Run the distributed solve job
                ret = master.exec_run(
                    ["python", "-u", "/tmp/distributed_cg_job.py"],
                    demux=True,
                )
                stdout = (ret.output[0] or b"").decode()
                stderr = (ret.output[1] or b"").decode()

                if stdout:
                    print(stdout)
                if ret.exit_code != 0:
                    raise Exception(
                        f"Distributed solve failed (exit {ret.exit_code}):\n{stderr}"
                    )

                # Copy results back from master
                subprocess.run(
                    ["docker", "cp", "master:/tmp/results.npz", str(tmpdir / "results.npz")],
                    check=True,
                )

                # Load and reassemble results
                results_data = np.load(tmpdir / "results.npz")
                partition_results = []
                for i in range(self.n_partitions):
                    partition_results.append({
                        "x": results_data[f"x_{i}"].copy(),
                        "error": float(results_data[f"error_{i}"][0]),
                        "iteration": int(results_data[f"iteration_{i}"][0]),
                        "global_row_start": int(results_data[f"start_{i}"][0]),
                        "global_row_end": int(results_data[f"end_{i}"][0]),
                    })
                results_data.close()

                return self.reassemble_solution(partition_results)

            finally:
                _compose("down")

    def reassemble_solution(self, partition_results):
        """Stitch partial solutions from each partition into a full N-length vector."""
        x_full = np.zeros(self.N, dtype=np.float64)
        max_error = 0.0
        max_iteration = 0
        for result in partition_results:
            start = result["global_row_start"]
            end = result["global_row_end"]
            x_full[start:end] = result["x"]
            max_error = max(max_error, result["error"])
            max_iteration = max(max_iteration, result["iteration"])
        self.x = x_full
        self.error = max_error
        self.iteration = max_iteration
        return x_full, max_error, max_iteration


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def partition_csr_system(a_sparse_array, b_array, n_partitions, pressure_estimate=None):
    """
    Partition an N-row CSR system into n_partitions independent sub-systems.

    For each partition, rows that reference columns in other partitions have
    those entries removed from the local A matrix and their contribution
    moved to the local b vector using an assumed x value for those columns.

    When pressure_estimate is None, the flat BOUNDARY_ESTIMATE (0.5) is used
    for all cross-partition references. When provided, pressure_estimate[j]
    is used as the assumed x value for column j.

    Parameters
    ----------
    a_sparse_array : dict
        CSR sparse matrix with keys "val", "col_idx", "row_ptr" (N-element row_ptr).
    b_array : np.ndarray
        Right-hand side vector of length N.
    n_partitions : int
        Number of partitions to create.
    pressure_estimate : np.ndarray or None, optional
        Per-node pressure estimate of length N. When provided, cross-partition
        entries use pressure_estimate[col] instead of BOUNDARY_ESTIMATE.

    Returns
    -------
    list of dict
        Each dict has keys: "a_sparse_array", "b_array",
        "global_row_start", "global_row_end".
    """
    N = b_array.size
    val = a_sparse_array["val"]
    col_idx = a_sparse_array["col_idx"]
    row_ptr = a_sparse_array["row_ptr"]

    boundaries = [k * N // n_partitions for k in range(n_partitions + 1)]

    partitions = []
    for k in range(n_partitions):
        row_start = boundaries[k]
        row_end = boundaries[k + 1]
        local_n = row_end - row_start

        sub_b = b_array[row_start:row_end].copy()
        sub_vals = []
        sub_col_idxs = []
        sub_row_ptrs = []

        for local_row in range(local_n):
            global_row = row_start + local_row
            sub_row_ptrs.append(len(sub_vals))

            # Row bounds in the N-element row_ptr format
            start = row_ptr[global_row]
            if global_row < N - 1:
                stop = row_ptr[global_row + 1]
            else:
                stop = val.size

            for idx in range(start, stop):
                col = col_idx[idx]
                v = val[idx]

                if row_start <= col < row_end:
                    # Intra-partition: keep with remapped column index
                    sub_vals.append(v)
                    sub_col_idxs.append(col - row_start)
                else:
                    # Cross-partition: move contribution to b vector
                    x_est = pressure_estimate[col] if pressure_estimate is not None else BOUNDARY_ESTIMATE
                    sub_b[local_row] -= v * x_est

        sub_sparse = {
            "val": np.array(sub_vals, dtype=np.float64),
            "col_idx": np.array(sub_col_idxs, dtype=np.int32),
            "row_ptr": np.array(sub_row_ptrs, dtype=np.int32),
        }
        partitions.append({
            "a_sparse_array": sub_sparse,
            "b_array": sub_b,
            "global_row_start": row_start,
            "global_row_end": row_end,
        })

    return partitions


# ---------------------------------------------------------------------------
# Docker helpers (following docker/run_dask_invert.py pattern)
# ---------------------------------------------------------------------------

def _compose(*args, check=True):
    return subprocess.run(
        [*COMPOSE_CMD, *args],
        cwd=DOCKER_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _images_need_build(client):
    """Check if Docker images need rebuilding (source newer than images)."""
    from datetime import datetime

    # Collect latest mtime from Docker build inputs and pyflowsolver source
    latest_source_mtime = 0.0
    for name in BUILD_INPUTS:
        path = DOCKER_DIR / name
        if path.exists():
            latest_source_mtime = max(latest_source_mtime, path.stat().st_mtime)
    for path in SOURCE_DIR.glob("*.py"):
        latest_source_mtime = max(latest_source_mtime, path.stat().st_mtime)

    for name in IMAGE_NAMES:
        try:
            img = client.images.get(f"{name}:latest")
        except Exception:
            return True
        created = datetime.fromisoformat(img.attrs["Created"]).timestamp()
        if created < latest_source_mtime:
            return True
    return False


def _wait_for_workers(container, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ret = container.exec_run("sinfo -h -o %T")
        states = ret.output.decode().strip().splitlines()
        if states and all(s == "idle" for s in states):
            return True
        time.sleep(1)
    return False
