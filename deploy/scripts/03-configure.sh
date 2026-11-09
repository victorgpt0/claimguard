#!/usr/bin/env bash
# Step 3: the deployment's settings and certificates. Safe to run again: it keeps an existing
# deploy/.env and existing certificates (delete them first to start over).
#   deploy/.env            secrets and settings, read by docker compose (never commit it)
#   config/config.yaml     the pipeline's config, config.example.yaml with Drive paths
#   deploy/tls/            the PostgreSQL/Redis certificate, signed by the deployment's own CA
#   deploy/ca-private/     that CA's private key (stays on the VM; not mounted anywhere)
#
# Usage: bash 03-configure.sh [public-ip]   (default: asks a public what's-my-IP service)
set -euo pipefail

DEPLOY="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(dirname "$DEPLOY")"
ENV="$DEPLOY/.env"
DOMAIN_DEFAULT=claimsguard.tech

rand() { openssl rand -hex 32; }

# ---- deploy/.env -----------------------------------------------------------------------------
if [ ! -f "$ENV" ]; then
    read -rp "Domain [$DOMAIN_DEFAULT]: " domain
    domain=${domain:-$DOMAIN_DEFAULT}
    (umask 077; cat > "$ENV") <<EOF
# ClaimGuard production settings (deploy/README.md). Never commit or screenshot this file.
DOMAIN=$domain
DJANGO_SECRET_KEY=$(openssl rand -base64 60 | tr -d '\n/+=')
POSTGRES_PASSWORD=$(rand)
SHA_CDP_DB_PASSWORD=$(rand)
REDIS_PASSWORD=$(rand)
SHA_CDP_API_KEY=$(rand)
SHA_CDP_ADMIN_TOKEN=$(rand)
SESSION_TIMEOUT_MINUTES=30

# Optional. Google sign-in: add https://$domain/accounts/google/login/callback/ as an
# authorised redirect URI on the OAuth client first.
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
# Optional, for the ML code (the CPU worker writes no narratives, so usually unused here).
HF_TOKEN=
OPENAI_API_KEY=

# Written by scripts/04-cloudflare-ips.sh
CF_IPS=
EOF
    echo "Wrote $ENV"
else
    echo "Keeping the existing $ENV"
fi

# ---- config/config.yaml ----------------------------------------------------------------------
CFG="$REPO/config/config.yaml"
if [ ! -f "$CFG" ]; then
    sed 's#/content/drive/MyDrive#/srv/gdrive#g' "$REPO/config/config.example.yaml" > "$CFG"
    echo "Wrote $CFG (paths under /srv/gdrive/claimguard)"
else
    echo "Keeping the existing $CFG"
fi
grep -n '/srv/gdrive' "$CFG" | head -5

# ---- TLS for PostgreSQL and Redis -------------------------------------------------------------
# A private CA rather than a public certificate: the Colab worker connects by IP, which no
# public CA will certify. The worker trusts ca.crt and checks the IP in the certificate.
TLS="$DEPLOY/tls"; CA="$DEPLOY/ca-private"
if [ ! -f "$TLS/server.crt" ]; then
    ip="${1:-$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')}"
    [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Could not tell the public IP; pass it as an argument."; exit 1; }
    echo "Certificate for $ip (it must be the RESERVED public IP, or the certificate breaks when it changes)"
    mkdir -p "$TLS" "$CA"; chmod 700 "$CA"
    # The extensions below are what strict X.509 checking (Python 3.13's default) demands;
    # without keyUsage on the CA, redis-py and Celery refuse the certificate.
    openssl req -x509 -new -nodes -newkey rsa:3072 -sha256 -days 3650 \
        -keyout "$CA/ca.key" -out "$TLS/ca.crt" -subj "/CN=ClaimGuard deployment CA" \
        -addext "basicConstraints=critical,CA:TRUE" \
        -addext "keyUsage=critical,keyCertSign,cRLSign" \
        -addext "subjectKeyIdentifier=hash"
    openssl req -new -nodes -newkey rsa:2048 -keyout "$TLS/server.key" -out "$CA/server.csr" -subj "/CN=$ip"
    printf '%s\n' "subjectAltName=IP:$ip" "basicConstraints=critical,CA:FALSE" \
        "keyUsage=critical,digitalSignature,keyEncipherment" "extendedKeyUsage=serverAuth" \
        "subjectKeyIdentifier=hash" "authorityKeyIdentifier=keyid,issuer" > "$CA/server.ext"
    openssl x509 -req -in "$CA/server.csr" -CA "$TLS/ca.crt" -CAkey "$CA/ca.key" -CAcreateserial -CAserial "$CA/ca.srl" \
        -days 825 -sha256 -extfile "$CA/server.ext" -out "$TLS/server.crt"
    # PostgreSQL and Redis both run as uid 999 in their images; PostgreSQL insists the key is
    # owned by its user and not readable by others.
    sudo chown 999:999 "$TLS/server.key" "$TLS/server.crt"
    sudo chmod 600 "$TLS/server.key"
    chmod 644 "$TLS/ca.crt"
    echo "Copy $TLS/ca.crt to Drive for the Colab worker (README, step 8)."
else
    echo "Keeping the existing certificates in $TLS"
fi

mkdir -p "$DEPLOY/origin"
echo
echo "Next: bash $DEPLOY/scripts/04-cloudflare-ips.sh"
