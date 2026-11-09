#!/bin/bash
# Runs once, when the PostgreSQL volume is first created: the simulated claims platform's own
# database and role (#65), kept apart from the audit system's.
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v pw="$SHA_CDP_DB_PASSWORD" <<'SQL'
CREATE ROLE sha_cdp LOGIN PASSWORD :'pw';
CREATE DATABASE sha_cdp OWNER sha_cdp;
SQL
