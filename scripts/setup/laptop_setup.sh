#!/bin/bash
set -e

function confirm_oculus {
    echo "ADB devices:"
    adb devices
    read -p "Is your Oculus Quest visible in the list above? (y/n): " c
    [ "$c" = "y" ] || [ "$c" = "Y" ]
}

ROOT_DIR="$(git -C "$(dirname "$(realpath "$0")")" rev-parse --show-toplevel)"
DOCKER_COMPOSE_FILE="$ROOT_DIR/.docker/laptop/docker-compose-laptop.yaml"

echo "============================================"
echo "  franka_direct  —  Laptop setup"
echo "============================================"

read -p "Is this your first time setting up this machine? (yes/no): " first_time

if [ "$first_time" = "yes" ]; then
    echo -e "\nInstalling git-lfs and pulling submodules (oculus_reader APK)..."
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

    echo -e "\nInstalling NVIDIA container toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
else
    echo -e "\nWelcome back!"
fi

# ── Install oculus_reader APK ─────────────────────────────────────────────────
read -p "Have you already installed the oculus_reader APK on your Quest? (yes/no): " apk_done
if [ "$apk_done" = "no" ]; then
    echo -e "\nInstalling ADB and oculus_reader APK..."
    apt install -y android-tools-adb android-sdk-platform-tools-common
    pip3 install -e "$ROOT_DIR/oculus_reader"
    echo "Connect your Quest 3 via USB-C and enable USB debugging."
    read -p "Press Enter when ready..."
    adb start-server
    max_retries=3; retry_count=0
    while ! confirm_oculus; do
        ((retry_count++))
        [ "$retry_count" -ge "$max_retries" ] && { echo "Max retries reached."; exit 1; }
        echo "Retrying..."
    done
    python3 "$ROOT_DIR/oculus_reader/oculus_reader/reader.py"
    echo "Waiting for APK install to complete..."
    sleep 5
    adb kill-server
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
echo -e "\nConfiguring static IP ($LAPTOP_IP) on Ethernet interface..."
interfaces=$(ip -o link show | grep -Eo '^[0-9]+: (en|eth|ens|eno|enp)[a-z0-9]*' | awk -F' ' '{print $2}')
echo "Select an Ethernet interface:"
select interface_name in $interfaces; do
    [ -n "$interface_name" ] && break
    echo "Invalid selection, try again."
done
nmcli connection delete "fd_laptop_static" 2>/dev/null || true
nmcli connection add con-name "fd_laptop_static" ifname "$interface_name" type ethernet
nmcli connection modify "fd_laptop_static" ipv4.method manual ipv4.address "$LAPTOP_IP/24"
nmcli connection up "fd_laptop_static"
echo "Static IP set: $LAPTOP_IP on $interface_name"

# ── X11 forwarding for container ──────────────────────────────────────────────
export DOCKER_XAUTH=/tmp/.docker.xauth
rm -f "$DOCKER_XAUTH"
touch "$DOCKER_XAUTH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$DOCKER_XAUTH" nmerge -

# ── ADB server on host (Quest connects from inside container) ─────────────────
echo -e "\nStarting ADB server on host..."
read -p "Plug in and out any USB cameras, then press Enter..."
adb kill-server
adb -a nodaemon server start &>/dev/null &
max_retries=3; retry_count=0
while ! confirm_oculus; do
    ((retry_count++))
    [ "$retry_count" -ge "$max_retries" ] && { echo "Max retries reached."; exit 1; }
    echo "Retrying..."
done

# ── Start container ───────────────────────────────────────────────────────────
read -p "Rebuild Docker image? (yes/no): " rebuild
if [ "$rebuild" = "yes" ]; then
    echo -e "\nBuilding laptop image..."
    docker compose -f "$DOCKER_COMPOSE_FILE" build
fi

echo -e "\nStarting franka_direct laptop container..."
mkdir -p "$ROOT_DIR/recordings"
docker compose -f "$DOCKER_COMPOSE_FILE" up -d
echo -e "\nDone. Container is running."
echo ""
echo "To start teleoperation:"
echo "  docker exec -it franka_direct_laptop python /app/scripts/simple_teleop_direct_torque.py"