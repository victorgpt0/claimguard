#!/usr/bin/env bash
# Step 9: install the nightly database backup (02:30 Nairobi time) and the monthly refresh of
# Cloudflare's address ranges as systemd timers. Check them with: systemctl list-timers 'claimguard-*'
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

sudo tee /etc/systemd/system/claimguard-backup.service >/dev/null <<UNIT
[Unit]
Description=ClaimGuard database backup
Requires=docker.service
After=docker.service rclone-gdrive.service

[Service]
Type=oneshot
ExecStart=/bin/bash $SCRIPTS/backup.sh
UNIT

sudo tee /etc/systemd/system/claimguard-backup.timer >/dev/null <<'UNIT'
[Unit]
Description=Nightly ClaimGuard database backup

[Timer]
OnCalendar=*-*-* 02:30:00 Africa/Nairobi
Persistent=true

[Install]
WantedBy=timers.target
UNIT

sudo tee /etc/systemd/system/claimguard-cf-ips.service >/dev/null <<UNIT
[Unit]
Description=Refresh Cloudflare address ranges for Caddy
After=docker.service network-online.target

[Service]
Type=oneshot
User=$USER
ExecStart=/bin/bash $SCRIPTS/04-cloudflare-ips.sh
UNIT

sudo tee /etc/systemd/system/claimguard-cf-ips.timer >/dev/null <<'UNIT'
[Unit]
Description=Monthly refresh of Cloudflare address ranges

[Timer]
OnCalendar=monthly
Persistent=true

[Install]
WantedBy=timers.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now claimguard-backup.timer claimguard-cf-ips.timer
echo "Running one backup now to check it works..."
sudo systemctl start claimguard-backup.service
sudo journalctl -u claimguard-backup.service -n 5 --no-pager
systemctl list-timers 'claimguard-*' --no-pager
