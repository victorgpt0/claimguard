# ClaimGuard operational check: after-reboot

- Run: 2026-11-10T20:21:13+00:00 on claimguard-instance-vnic, deployed commit 49046c6
- Target: https://claimsguard.tech (requests from the VM go out through Cloudflare and back, as a visitor's do)
- Settings: {"session_timeout_minutes": 30, "session_save_every_request": true, "debug": false, "conn_max_age": 60, "conn_health_checks": true}

## After a reboot

- Booted at 2026-11-10 20:15:45; checked 2026-11-10 20:21:16.
- Google Drive mounted at /srv/gdrive: yes.

```
SERVICE         STATE     STATUS
caddy           running   Up 5 minutes
db              running   Up 5 minutes (healthy)
redis           running   Up 5 minutes
sha-cdp         running   Up 5 minutes
web             running   Up 5 minutes
worker-cpu      running   Up 5 minutes
worker-ingest   running   Up 5 minutes
```
- Login page 200 in 1.084213s; workers online (cpu, ingest) after 2.6s more; rows {"batches": 10, "claims": 3075, "reports": 819, "jobs": 9, "ingestion_runs": 1991}. Nothing was started by hand. **Pass**

