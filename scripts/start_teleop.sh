#!/usr/bin/env bash
set -euo pipefail

NUC_HOST="192.168.1.6"
NUC_CONFIG_DIR="~/fr3/franka_direct/config"
LAPTOP_CTR="laptop-franka_direct_laptop-1"

# === 1. Sync controller.yaml to NUC ==========================================
echo "=== Syncing controller.yaml to NUC ==="
scp config/controller.yaml "$NUC_HOST:$NUC_CONFIG_DIR/controller.yaml"

# === 3. Start teleoperation ===================================================
echo "=== Starting teleoperation ==="
docker exec -it "$LAPTOP_CTR" bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate robot && \
    pip install -q grpcio grpcio-tools protobuf==3.20.1 && \
    bash /app/python/generate_stubs.sh && \
    python /app/scripts/simple_teleop_direct_torque.py --config /app/config/teleop.yaml
"
