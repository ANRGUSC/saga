#!/usr/bin/env bash
# Launch one CCTools work_queue_worker per Chameleon bare-metal node, tagging
# each with a unique feature so SAGA can pin tasks to the right worker.
#
# Assumptions:
#   * You have reserved a Chameleon lease and have SSH access to each node
#     as ``cc`` (the default user) via a key at ~/.ssh/id_rsa.
#   * Each worker has CCTools installed (``conda install -c conda-forge
#     ndcctools`` on a miniconda env, or ``apt install cctools``).
#   * Your driver machine is running the SAGA Work Queue manager with the
#     project name given as PROJECT below.
#
# Usage:
#   provision_workers.sh PROJECT NODE1 NODE2 ... NODEn
#
# Example:
#   provision_workers.sh saga-demo \
#       saga-worker-0 saga-worker-1 saga-worker-2
#
# Each worker gets a feature tag matching the SAGA node name.  The tag name
# is derived from the node name by replacing non-alphanumeric characters with
# underscores and prefixing with "saga-node-" -- this matches the function
# ``feature_for`` in ``saga.execution.workqueue``.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 PROJECT NODE1 [NODE2 ...]" >&2
    exit 1
fi

PROJECT="$1"
shift

for NODE in "$@"; do
    # Mirror saga.execution.workqueue.feature_for().
    SAFE=$(echo "$NODE" | sed 's/[^[:alnum:]]/_/g')
    FEATURE="saga-node-${SAFE}"

    echo "[+] Starting work_queue_worker on $NODE with feature=$FEATURE"
    ssh -o StrictHostKeyChecking=no "cc@$NODE" \
        "nohup work_queue_worker -N ${PROJECT} --feature ${FEATURE} \
         --cores 0 --memory 0 --disk 0 \
         > work_queue_worker.log 2>&1 &"
done

echo "[+] All workers launched.  Check individual logs at ~/work_queue_worker.log on each node."
