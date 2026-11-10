# ClaimGuard operational check: restart

- Run: 2026-11-10T20:14:50+00:00 on claimguard-instance-vnic, deployed commit 49046c6
- Target: https://claimsguard.tech (requests from the VM go out through Cloudflare and back, as a visitor's do)
- Settings: {"session_timeout_minutes": 30, "session_save_every_request": true, "debug": false, "conn_max_age": 60, "conn_health_checks": true}

## Restarts

- Web application (docker compose restart web): restart command took 10.4 s; 1.2 s unavailable (answers seen: 502); 3 of 21 polls failed.
- Whole stack (docker compose restart): restart command took 10.5 s; 9.8 s unavailable (answers seen: 502, 521); 5 of 18 polls failed.
- Workers online again 2.6s after the stack was back. Rows before {"batches": 10, "claims": 3075, "reports": 819, "jobs": 9, "ingestion_runs": 1979}; after {"batches": 10, "claims": 3075, "reports": 819, "jobs": 9, "ingestion_runs": 1980} (unchanged expected). **FAIL**

Next: `sudo reboot`, log back in, then run this script with `after-reboot`.

