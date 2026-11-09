#!/usr/bin/env bash
# Step 4 (and monthly, from the timer that 06-backups.sh installs): write Cloudflare's current
# address ranges into deploy/.env as CF_IPS. Caddy accepts web traffic only from these
# addresses and reads the visitor's real address from CF-Connecting-IP only when it comes from them.
# Ranges: https://www.cloudflare.com/ips/
set -euo pipefail

DEPLOY="$(cd "$(dirname "$0")/.." && pwd)"
ENV="$DEPLOY/.env"
[ -f "$ENV" ] || { echo "No $ENV yet: run 03-configure.sh first."; exit 1; }

v4=$(curl -fsS https://www.cloudflare.com/ips-v4)
v6=$(curl -fsS https://www.cloudflare.com/ips-v6)
ips=$(printf '%s\n%s\n' "$v4" "$v6" | grep -E '^[0-9a-fA-F:.]+/[0-9]+$' | tr '\n' ' ' | sed 's/ $//')
count=$(wc -w <<<"$ips")
[ "$count" -ge 10 ] || { echo "Got only $count ranges from Cloudflare; leaving $ENV unchanged."; exit 1; }

if grep -q '^CF_IPS=' "$ENV"; then
    sed -i "s#^CF_IPS=.*#CF_IPS=\"$ips\"#" "$ENV"
else
    echo "CF_IPS=\"$ips\"" >> "$ENV"
fi
echo "CF_IPS: $count ranges written to $ENV"

# Pick up the change if Caddy is already running.
if docker compose -f "$DEPLOY/compose.yaml" ps --status running caddy 2>/dev/null | grep -q caddy; then
    docker compose -f "$DEPLOY/compose.yaml" up -d caddy
fi
