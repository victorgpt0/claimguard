#!/usr/bin/env bash
# Step 6: copy the audit database from Neon into the VM's PostgreSQL, ONCE, before the web
# container first starts (it runs the migrations, which would create empty tables in the way).
#
# Usage: NEON_URL='postgresql://...@ep-....neon.tech/neondb?sslmode=require' bash 05-migrate-from-neon.sh
# Use the DIRECT (unpooled) URL, DATABASE_URL_UNPOOLED in the laptop's .env: pg_dump cannot
# run through Neon's PgBouncer pooler.
#
# The claims platform's database is not copied: seed it fresh from the feed file (README, step 7).
set -euo pipefail

DEPLOY="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DEPLOY"
: "${NEON_URL:?set NEON_URL to the Neon direct (unpooled) connection string}"
[[ "$NEON_URL" != *-pooler.* ]] || { echo "That is the pooled URL; use the direct one (no -pooler)."; exit 1; }

DUMP="$DEPLOY/neon-$(date +%Y%m%d-%H%M).dump"
psql_vm() { docker compose exec -T db psql -U claimguard -d claimguard -Atq "$@"; }

echo "==> Starting only the database"
docker compose up -d db
for _ in $(seq 60); do
    docker compose exec -T db pg_isready -U claimguard -d claimguard >/dev/null 2>&1 && break
    sleep 2
done

tables=$(psql_vm -c "select count(*) from information_schema.tables where table_schema='public'")
if [ "$tables" != "0" ]; then
    echo "The VM database already has $tables tables; refusing to restore over it."
    echo "(Fresh start: docker compose down, docker volume rm claimguard_pgdata, run this again.)"
    exit 1
fi

echo "==> Dumping from Neon to $DUMP (pg_dump 18, the same major version as Neon)"
docker run --rm postgres:18 pg_dump "$NEON_URL" --format=custom --no-owner --no-acl > "$DUMP"
ls -lh "$DUMP"

echo "==> Restoring into the VM database"
# Neon adds a few objects of its own that plain PostgreSQL may reject; those errors are
# printed and the row counts below decide whether the data arrived.
docker compose exec -T db pg_restore -U claimguard -d claimguard --no-owner --role=claimguard < "$DUMP" \
    || echo "pg_restore reported errors (above); checking the row counts."

echo "==> Row counts, Neon vs VM"
count_sql="select table_name from information_schema.tables where table_schema='public' and table_type='BASE TABLE' order by 1"
mismatch=0
printf '%-40s %10s %10s\n' table neon vm
for t in $(psql_vm -c "$count_sql"); do
    n=$(docker run --rm postgres:18 psql "$NEON_URL" -Atqc "select count(*) from public.\"$t\"")
    v=$(psql_vm -c "select count(*) from public.\"$t\"")
    flag=""; [ "$n" = "$v" ] || { flag="  <-- differs"; mismatch=1; }
    printf '%-40s %10s %10s%s\n' "$t" "$n" "$v" "$flag"
done
[ "$mismatch" = 0 ] && echo "All tables match." || echo "Some counts differ: check before going live."
echo
echo "The dump stays at $DUMP (it holds the accounts' password hashes: delete it once you are happy)."
echo "Next: docker compose up -d --build"
