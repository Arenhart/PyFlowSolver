#!/bin/bash
set -e

# Fix munge socket dir ownership (needed in some slim images)
chown munge:munge /var/run/munge
chmod 755 /var/run/munge

# Start munge on every node
gosu munge /usr/sbin/munged --foreground &
sleep 1

ROLE=${SLURM_ROLE:-worker}

if [ "$ROLE" = "master" ]; then
    echo "[entrypoint] Starting slurmctld (master)..."
    exec /usr/sbin/slurmctld -D
else
    echo "[entrypoint] Starting slurmd (worker)..."
    exec /usr/sbin/slurmd -D -N "$SLURM_NODENAME"
fi