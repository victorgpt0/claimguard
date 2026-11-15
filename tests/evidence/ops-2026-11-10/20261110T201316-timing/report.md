# ClaimGuard operational check: timing

- Run: 2026-11-10T20:13:16+00:00 on claimguard-instance-vnic, deployed commit 49046c6
- Target: https://claimsguard.tech (requests from the VM go out through Cloudflare and back, as a visitor's do)
- Settings: {"session_timeout_minutes": 30, "session_save_every_request": true, "debug": false, "conn_max_age": 60, "conn_health_checks": true}

## The tested setup (for docs/environment.md)

- OS: Ubuntu 24.04.5 LTS x86_64, kernel 7.0.0-1013-oracle
- CPU: 2 cores (AMD EPYC 9J14 96-Core Processor); memory: 11Gi total, 9Gi available; swap 4.0Gi
- Disk /: 48G total, 32G free
- Docker version 29.9.3, build 5457199; Docker Compose version v5.7.0

```
CONTAINER                    REPOSITORY          TAG                 PLATFORM            IMAGE ID            SIZE                CREATED
claimguard-caddy-1           caddy               2.10                linux/amd64         c3d7ee5d2b11        82.8MB              9 months ago
claimguard-db-1              postgres            18                  linux/amd64         e3d24f6906d0        644MB               5 days ago
claimguard-redis-1           redis               7.4                 linux/amd64         2e3eb11462c4        167MB               5 days ago
claimguard-sha-cdp-1         claimguard-web      latest              linux/amd64         39b7a6e56d79        1.2GB               About a minute ago
claimguard-web-1             claimguard-web      latest              linux/amd64         39b7a6e56d79        1.2GB               About a minute ago
claimguard-worker-cpu-1      claimguard-worker   latest              linux/amd64         bf540a37b58a        2.59GB              About a minute ago
claimguard-worker-ingest-1   claimguard-web      latest              linux/amd64         39b7a6e56d79        1.2GB               About a minute ago
postgres (PostgreSQL) 18.6 (Debian 18.6-1.pgdg13+2); Redis server v=7.4.12
Python 3.13.18 Django 5.2.18
```

## Response times (30 sequential requests per page)

| Page | As | n | Not 200 | Median s | p95 s | Max s |
|---|---|---|---|---|---|---|
| `/accounts/login/` | anon | 30 | 0 | 0.069 | 0.982 | 1.014 |
| `/audit/` | Administrator | 30 | 0 | 0.146 | 0.265 | 0.875 |
| `/audit/batches/` | Administrator | 30 | 0 | 0.072 | 0.094 | 0.109 |
| `/pipeline/` | Administrator | 30 | 0 | 0.107 | 0.229 | 0.293 |
| `/audit/model/` | Administrator | 30 | 0 | 0.097 | 0.126 | 2.364 |
| `/audit/queue/` | Auditor | 30 | 0 | 0.102 | 0.158 | 0.379 |
| `/audit/batches/<id>/` | Administrator | 30 | 0 | 0.084 | 0.107 | 0.116 |
| `/audit/reports/<id>/` | Auditor | 30 | 0 | 0.070 | 0.097 | 0.125 |

All pages: 240 requests, 0 not answered 200 (0.0%).

