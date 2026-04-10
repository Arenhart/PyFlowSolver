FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    slurm-wlm \
    munge \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy numba dask[distributed] dask-jobqueue

# Install pyflowsolver from source
COPY pyflowsolver/ /app/pyflowsolver/
COPY setup.py /app/setup.py
RUN pip install --no-cache-dir /app/

RUN mkdir -p /var/spool/slurm /var/spool/slurmd /var/run/slurm /var/log/slurm \
             /var/run/munge /var/log/munge /etc/munge \
    && chown slurm:slurm /var/spool/slurm /var/spool/slurmd /var/run/slurm /var/log/slurm

COPY docker/munge.key /etc/munge/munge.key
COPY docker/cgroup.conf /etc/slurm/cgroup.conf
RUN chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key

COPY docker/slurm.conf /etc/slurm/slurm.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
