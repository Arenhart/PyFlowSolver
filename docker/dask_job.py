"""Distributed array inversion using Dask on a SLURM cluster.

Creates a random binary 50x50 array with dask, scatters each half to a
different SLURM worker, inverts the values (0->1, 1->0), and gathers
the results back.
"""

import socket

import dask.array as da
import numpy as np
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def invert_chunk(chunk):
    """Invert binary values and report which host executed the task."""
    return {"host": socket.gethostname(), "result": 1 - chunk}


def main():
    cluster = SLURMCluster(
        cores=1,
        memory="1GB",
        processes=1,
        queue="debug",
        walltime="00:10:00",
        job_extra_directives=["--exclusive"],
        scheduler_options={"dashboard_address": ":0"},
    )
    cluster.scale(jobs=2)

    client = Client(cluster)
    print("Waiting for 2 Dask workers...")
    client.wait_for_workers(2, timeout=120)

    worker_addrs = sorted(client.scheduler_info()["workers"].keys())
    info = client.scheduler_info()["workers"]
    for addr in worker_addrs:
        print(f"  Worker on {info[addr]['host']} ({addr})")

    # -- Create random binary array using dask ------------------------------
    rng = da.random.RandomState(42)
    darr = rng.randint(0, 2, size=(50, 50), chunks=(25, 50))
    arr = darr.compute()
    print(f"\nCreated {arr.shape} binary array via dask")
    print(f"  First row : {arr[0]}")
    print(f"  Ones count: {arr.sum()} / {arr.size}")

    # -- Scatter each half to a specific worker -----------------------------
    top_future = client.scatter(arr[:25], workers=[worker_addrs[0]])
    bottom_future = client.scatter(arr[25:], workers=[worker_addrs[1]])

    # -- Invert on the worker that holds the data ---------------------------
    inv_top = client.submit(invert_chunk, top_future, workers=[worker_addrs[0]])
    inv_bottom = client.submit(invert_chunk, bottom_future, workers=[worker_addrs[1]])

    result_top, result_bottom = client.gather([inv_top, inv_bottom])

    # -- Display results ----------------------------------------------------
    print(f"\n--- Top half (rows 0-24) inverted by '{result_top['host']}' ---")
    print(f"  Shape    : {result_top['result'].shape}")
    print(f"  First row: {result_top['result'][0]}")

    print(f"\n--- Bottom half (rows 25-49) inverted by '{result_bottom['host']}' ---")
    print(f"  Shape    : {result_bottom['result'].shape}")
    print(f"  First row: {result_bottom['result'][0]}")

    # -- Verify -------------------------------------------------------------
    combined = np.vstack([result_top["result"], result_bottom["result"]])
    expected = 1 - arr
    assert np.array_equal(combined, expected), "Inversion verification FAILED!"
    print(f"\nVerification passed: all {arr.size} values correctly inverted.")
    print(f"  Original ones: {arr.sum()},  Inverted ones: {combined.sum()},  Total: {arr.size}")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
