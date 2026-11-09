#!/usr/bin/env bash
# Step 1 on a fresh Oracle Cloud Ubuntu 24.04 (ARM) VM (deploy/README.md). Copy it over and run:
#   scp deploy/scripts/01-bootstrap-vm.sh ubuntu@<vm-ip>:
#   ssh ubuntu@<vm-ip> 'bash 01-bootstrap-vm.sh'
# Safe to run again. It installs Docker, rclone and updates; opens the ports in the VM's own
# firewall; adds swap; and clones the repository to /opt/claimguard with a read-only deploy key.
set -euo pipefail

REPO_SSH="${REPO_SSH:-git@github.com:victorgpt0/claimguard.git}"
BRANCH="${BRANCH:-main}"   # main is the deployed branch (.github/workflows/deploy.yml)
APP_DIR=/opt/claimguard

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
[ "$(id -u)" -ne 0 ] || { echo "Run as the ubuntu user (it uses sudo), not as root."; exit 1; }

step "System updates and base packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates curl git openssl fuse3 unattended-upgrades fail2ban \
    iptables-persistent netfilter-persistent jq unzip
sudo dpkg-reconfigure -f noninteractive unattended-upgrades

step "Docker (Docker's own apt repository)"
if ! command -v docker >/dev/null; then
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
# Keep container logs from filling the disk.
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{ "log-driver": "json-file", "log-opts": { "max-size": "20m", "max-file": "5" } }
JSON
sudo systemctl enable --now docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"

step "The VM's own firewall"
# Oracle's Ubuntu images ship iptables rules that REJECT everything but SSH, on top of the
# VCN security list. Open the stack's ports here; the security list still decides who reaches them.
for port in 80 443 5432 6380; do
    if ! sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null; then
        # Just above the image's catch-all REJECT, or at the end if there is none.
        pos=$(sudo iptables -L INPUT --line-numbers -n | awk '$2 == "REJECT" {print $1; exit}')
        if [ -n "$pos" ]; then
            sudo iptables -I INPUT "$pos" -p tcp --dport "$port" -m state --state NEW -j ACCEPT
        else
            sudo iptables -A INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT
        fi
    fi
done
sudo netfilter-persistent save

step "4 GB swap (headroom for builds and the classifier)"
if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

step "SSH: keys only, fail2ban on"
sudo tee /etc/ssh/sshd_config.d/90-claimguard.conf >/dev/null <<'CONF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
CONF
sudo systemctl reload ssh
sudo systemctl enable --now fail2ban

step "rclone (for the Google Drive mount, step 2)"
if ! command -v rclone >/dev/null; then
    curl -fsSL https://rclone.org/install.sh | sudo bash
fi
grep -q '^user_allow_other' /etc/fuse.conf || echo 'user_allow_other' | sudo tee -a /etc/fuse.conf
sudo mkdir -p /srv/gdrive

step "The code: a read-only deploy key for the private repository"
KEY="$HOME/.ssh/claimguard_deploy"
if [ ! -f "$KEY" ]; then
    ssh-keygen -t ed25519 -N '' -C "claimguard-oracle-vm" -f "$KEY"
fi
grep -q 'claimguard_deploy' "$HOME/.ssh/config" 2>/dev/null || cat >> "$HOME/.ssh/config" <<CONF
Host github.com
    IdentityFile $KEY
    IdentitiesOnly yes
CONF
ssh-keyscan -t ed25519 github.com >> "$HOME/.ssh/known_hosts" 2>/dev/null
sort -u -o "$HOME/.ssh/known_hosts" "$HOME/.ssh/known_hosts"

if [ ! -d "$APP_DIR/.git" ]; then
    echo
    echo "Add this public key on GitHub: repository -> Settings -> Deploy keys -> Add deploy key"
    echo "(title: oracle-vm, leave 'Allow write access' OFF):"
    echo
    cat "$KEY.pub"
    echo
    read -rp "Press Enter once the key is added... "
    sudo mkdir -p "$APP_DIR"
    sudo chown "$USER:$USER" "$APP_DIR"
    git clone --branch "$BRANCH" "$REPO_SSH" "$APP_DIR"
else
    git -C "$APP_DIR" pull --ff-only
fi

step "Done"
echo "Log out and back in (so the docker group applies), then run:"
echo "  bash $APP_DIR/deploy/scripts/02-drive-mount.sh"
