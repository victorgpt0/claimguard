#!/usr/bin/env bash
# Nightly backup of both databases (run by the claimguard-backup timer, 06-backups.sh).
# Keeps 14 days on the VM (/var/backups/claimguard) and 30 days on Drive
# (MyDrive/claimguard/backups/oracle), so one copy survives losing either.
# Restore: docker compose exec -T db pg_restore -U claimguard -d <db> --clean --no-owner < <file>
set -euo pipefail

DEPLOY="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL=/var/backups/claimguard
DRIVE=/srv/gdrive/claimguard/backups/oracle
stamp=$(date +%Y%m%d-%H%M)
mkdir -p "$LOCAL"

for db in claimguard sha_cdp; do
    out="$LOCAL/$db-$stamp.dump"
    docker compose -f "$DEPLOY/compose.yaml" exec -T db pg_dump -U claimguard -d "$db" --format=custom > "$out"
    echo "$(date -Is) $db: $(du -h "$out" | cut -f1)"
done
find "$LOCAL" -name '*.dump' -mtime +14 -delete

if mountpoint -q /srv/gdrive; then
    mkdir -p "$DRIVE"
    cp "$LOCAL"/*-"$stamp".dump "$DRIVE"/
    find "$DRIVE" -name '*.dump' -mtime +30 -delete
else
    echo "Drive is not mounted: local copy only." >&2
fi
