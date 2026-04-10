FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    slurm-wlm \
    munge \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy dask[distributed] dask-jobqueue

RUN mkdir -p /var/spool/slurm /var/spool/slurmd /var/run/slurm /var/log/slurm \
             /var/run/munge /var/log/munge /etc/munge \
    && chown slurm:slurm /var/spool/slurm /var/spool/slurmd /var/run/slurm /var/log/slurm

COPY munge.key /etc/munge/munge.key
COPY cgroup.conf /etc/slurm/cgroup.conf
RUN chown munge:munge /etc/munge/munge.key && chmod 400 /etc/munge/munge.key

COPY slurm.conf /etc/slurm/slurm.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
