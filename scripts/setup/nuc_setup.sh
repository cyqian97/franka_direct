#!/bin/bash
set -e

ROOT_DIR="$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel)"
DOCKER_COMPOSE_FILE="$ROOT_DIR/.docker/nuc/docker-compose-nuc.yaml"

echo "============================================"
echo "  franka_direct  —  NUC setup"
echo "============================================"

read -p "Is this your first time setting up this machine? (yes/no): " first_time

if [ "$first_time" = "yes" ]; then
    # pull submodules (oculus_reader APK needs git-lfs)
    echo -e "\nPulling submodules..."
    read -p "Enter the username whose SSH key to use: " USERNAME
    eval "$(ssh-agent -s)"
    ssh-add /home/$USERNAME/.ssh/id_ed25519
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt update && apt install -y git-lfs
    git lfs install
    cd "$ROOT_DIR" && git submodule update --recursive --remote --init

    echo -e "\nInstalling Docker..."
    apt-get update
    apt-get install -y ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable docker

    echo -e "\nApplying real-time kernel patch..."
    apt update && apt install -y ubuntu-advantage-tools
    pro attach "$UBUNTU_PRO_TOKEN"
    pro enable realtime-kernel

    echo -e "\nSetting CPU governor to performance..."
    apt install -y cpufrequtils
    systemctl disable ondemand
    systemctl enable cpufrequtils
    sh -c 'echo "GOVERNOR=performance" > /etc/default/cpufrequtils'
    systemctl daemon-reload && systemctl restart cpufrequtils
else
    echo -e "\nWelcome back!"
fi

# ── Read parameters.py ────────────────────────────────────────────────────────
echo -e "\nReading parameters.py..."
PARAMETERS_FILE="$ROOT_DIR/parameters.py"
awk -F'[[:space:]]*=[[:space:]]*' \
    '/^[[:space:]]*([[:alnum:]_]+)[[:space:]]*=/ { gsub("\"", "", $2); print "export " toupper($1) "=" $2 }' \
    "$PARAMETERS_FILE" > /tmp/fd_env_vars.sh
source /tmp/fd_env_vars.sh
rm /tmp/fd_env_vars.sh
export ROOT_DIR="$ROOT_DIR"

# ── Static IP ─────────────────────────────────────────────────────────────────
echo -e "\nConfiguring static IP ($NUC_IP) on Ethernet interface..."
interfaces=$(ip -o link show | grep -Eo '^[0-9]+: (en|eth|ens|eno|enp)[a-z0-9]*' | awk -F' ' '{print $2}')
echo "Select an Ethernet interface:"
select interface_name in $interfaces; do
    [ -n "$interface_name" ] && break
    echo "Invalid selection, try again."
done
nmcli connection delete "fd_nuc_static" 2>/dev/null || true
nmcli connection add con-name "fd_nuc_static" ifname "$interface_name" type ethernet
nmcli connection modify "fd_nuc_static" ipv4.method manual ipv4.address "$NUC_IP/24"
nmcli connection up "fd_nuc_static"
echo "Static IP set: $NUC_IP on $interface_name"

# ── Start container ───────────────────────────────────────────────────────────
read -p "Rebuild Docker image? (yes/no): " rebuild
if [ "$rebuild" = "yes" ]; then
    echo -e "\nBuilding NUC image..."
    docker compose -f "$DOCKER_COMPOSE_FILE" build
fi

echo -e "\nStarting franka_direct NUC container..."
docker compose -f "$DOCKER_COMPOSE_FILE" up -d
echo -e "\nDone. Container is running."
echo ""
echo "Next step — build the C++ server inside the container:"
echo "  docker exec franka_direct_nuc bash /app/build.sh"
echo ""
echo "Then launch the server (pick one):"
echo "  docker exec franka_direct_nuc bash /app/launch_server.sh             # joint torque"
echo "  docker exec franka_direct_nuc bash /app/launch_server_cartesian.sh   # Cartesian"