#!/usr/bin/env bash
# Step 10: let GitHub Actions deploy (deploy/README.md). Makes an SSH key that can do exactly one
# thing on this VM, run scripts/deploy.sh (no shell, no port forwarding), and prints the three
# values to store as GitHub Actions secrets. The private key is printed once and deleted.
# Run it again to replace the key (the old one stops working).
#   bash 07-deploy-access.sh [reserved-ip]
set -euo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
TAG="claimguard-github-deploy"
ip="${1:-$(curl -fsS https://ifconfig.me || true)}"
[[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Pass the reserved public IP as an argument."; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
ssh-keygen -q -t ed25519 -N '' -C "$TAG" -f "$tmp/key"

auth="$HOME/.ssh/authorized_keys"
touch "$auth" && chmod 600 "$auth"
grep -v "$TAG" "$auth" > "$tmp/auth" || true
printf 'command="%s/deploy.sh",restrict %s\n' "$SCRIPTS" "$(cat "$tmp/key.pub")" >> "$tmp/auth"
cat "$tmp/auth" > "$auth"

cat <<EOF

Add these on GitHub: repository -> Settings -> Secrets and variables -> Actions -> New repository secret

DEPLOY_HOST
$ip

DEPLOY_KNOWN_HOSTS
$ip $(cut -d' ' -f1,2 /etc/ssh/ssh_host_ed25519_key.pub)

DEPLOY_SSH_KEY   (the whole block, including the BEGIN and END lines)
$(cat "$tmp/key")

The private key above is now deleted from this VM; it exists only in what you copy.
EOF
