#!/usr/bin/env bash
# Step 2: mount Google Drive at /srv/gdrive, as on the laptop (~/gdrive), so the stack reads
# what Colab writes (the active classifier, the datasets, the ICD-11 cache) and the Feedback
# Dataset lands back in Drive.
#
# It reuses the laptop's rclone login. On the LAPTOP, print the remote's config:
#   rclone config show gdrive
# and paste it (the [gdrive] line and everything under it) when this script asks.
# The pasted text holds a Google refresh token: treat it like a password.
set -euo pipefail

CONF_DIR=/root/.config/rclone
CONF="$CONF_DIR/rclone.conf"

if ! sudo test -f "$CONF" || ! sudo grep -q '^\[gdrive\]' "$CONF"; then
    echo "Paste the output of 'rclone config show gdrive' from the laptop, then press Ctrl-D:"
    pasted=$(cat)
    grep -q '^\[gdrive\]' <<<"$pasted" || { echo "That does not start with [gdrive]; nothing written."; exit 1; }
    sudo mkdir -p "$CONF_DIR"
    printf '%s\n' "$pasted" | sudo tee "$CONF" >/dev/null
    sudo chmod 600 "$CONF"
fi

echo "Checking the login..."
sudo rclone lsd gdrive:claimguard >/dev/null || { echo "rclone cannot list gdrive:claimguard"; exit 1; }

sudo tee /etc/systemd/system/rclone-gdrive.service >/dev/null <<'UNIT'
[Unit]
Description=Google Drive (rclone) at /srv/gdrive for the ClaimGuard containers
Wants=network-online.target
After=network-online.target
Before=docker.service

[Service]
Type=notify
ExecStart=/usr/bin/rclone mount gdrive: /srv/gdrive \
    --allow-other \
    --vfs-cache-mode full \
    --vfs-cache-max-size 5G \
    --vfs-cache-max-age 72h \
    --poll-interval 30s \
    --dir-cache-time 1h \
    --log-level NOTICE
ExecStop=/bin/fusermount3 -uz /srv/gdrive
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

# Docker starts after the mount, so containers bind the mounted Drive and not an empty folder.
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/10-after-gdrive.conf >/dev/null <<'UNIT'
[Unit]
Wants=rclone-gdrive.service
After=rclone-gdrive.service
UNIT

# install.sh puts rclone in /usr/bin; fall back to wherever it is.
RCLONE=$(command -v rclone)
[ "$RCLONE" = /usr/bin/rclone ] || sudo sed -i "s#/usr/bin/rclone#$RCLONE#" /etc/systemd/system/rclone-gdrive.service

sudo systemctl daemon-reload
sudo systemctl enable --now rclone-gdrive
sleep 3
ls /srv/gdrive/claimguard >/dev/null && echo "Drive mounted: $(ls /srv/gdrive/claimguard | tr '\n' ' ')"
echo
echo "If the stack is already running, restart it so the containers see the mount:"
echo "  cd /opt/claimguard/deploy && docker compose restart"
echo "Next: bash /opt/claimguard/deploy/scripts/03-configure.sh"
