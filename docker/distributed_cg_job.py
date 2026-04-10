"""Distributed Darcy solve using Dask on a SLURM cluster.

Loads partition .npz files from /tmp, scatters each partition to a
different SLURM worker, solves via DarcySolver, and gathers results.
"""

import numpy as np
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def solve_partition(partition_data):
    """Worker function: solve a single partition using DarcySolver with PCG."""
    from pyflowsolver.darcySolver import DarcySolver

    a_sparse = {
        "val": partition_data["val"],
        "col_idx": partition_data["col_idx"],
        "row_ptr": partition_data["row_ptr"],
    }
    b = partition_data["b_array"]

    solver = DarcySolver()
    solver.params["max_iterations"] = int(partition_data["max_iterations"])
    solver.params["target_error"] = float(partition_data["target_error"])
    solver.set_linear_system(a_sparse, b)
    solver.generate_preconditioner(preconditioner="inverse_diagonal")
    x, error, iteration = solver.solve_pcg()

    return {
        "x": x,
        "error": error,
        "iteration": iteration,
        "global_row_start": int(partition_data["global_row_start"]),
        "global_row_end": int(partition_data["global_row_end"]),
    }


def main():
    # Load metadata
    metadata = np.load("/tmp/metadata.npz")
    n_partitions = int(metadata["n_partitions"][0])
    max_iterations = int(metadata["max_iterations"][0])
    target_error = float(metadata["target_error"][0])

    # Load partitions
    partitions = []
    for i in range(n_partitions):
        data = np.load(f"/tmp/partition_{i}.npz")
        partitions.append({
            "val": data["val"],
            "col_idx": data["col_idx"],
            "row_ptr": data["row_ptr"],
            "b_array": data["b_array"],
            "global_row_start": data["global_row_start"][0],
            "global_row_end": data["global_row_end"][0],
            "max_iterations": max_iterations,
            "target_error": target_error,
        })

    # Create SLURM cluster via Dask
    n_workers = min(n_partitions, 2)
    cluster = SLURMCluster(
        cores=1,
        memory="1GB",
        processes=1,
        queue="debug",
        walltime="00:10:00",
        job_extra_directives=["--exclusive"],
        scheduler_options={"dashboard_address": ":0"},
    )
    cluster.scale(jobs=n_workers)

    client = Client(cluster)
    print(f"Waiting for {n_workers} Dask workers...")
    client.wait_for_workers(n_workers, timeout=120)

    worker_addrs = sorted(client.scheduler_info()["workers"].keys())
    info = client.scheduler_info()["workers"]
    for addr in worker_addrs:
        print(f"  Worker on {info[addr]['host']} ({addr})")

    # Scatter partition data to workers and submit solves
    futures = []
    for i, partition in enumerate(partitions):
        worker = worker_addrs[i % len(worker_addrs)]
        scattered = client.scatter(partition, workers=[worker])
        future = client.submit(solve_partition, scattered, workers=[worker])
        futures.append(future)

    results = client.gather(futures)

    # Save results
    save_dict = {}
    for i, result in enumerate(results):
        save_dict[f"x_{i}"] = result["x"]
        save_dict[f"error_{i}"] = np.array([result["error"]])
        save_dict[f"iteration_{i}"] = np.array([result["iteration"]])
        save_dict[f"start_{i}"] = np.array([result["global_row_start"]])
        save_dict[f"end_{i}"] = np.array([result["global_row_end"]])
    np.savez("/tmp/results.npz", **save_dict)

    print(f"\nSolved {n_partitions} partitions successfully.")
    for i, result in enumerate(results):
        print(f"  Partition {i}: {result['iteration']} iterations, error={result['error']:.2e}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
