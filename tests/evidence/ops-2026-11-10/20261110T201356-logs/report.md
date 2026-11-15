# ClaimGuard operational check: logs

- Run: 2026-11-10T20:13:56+00:00 on claimguard-instance-vnic, deployed commit 49046c6
- Target: https://claimsguard.tech (requests from the VM go out through Cloudflare and back, as a visitor's do)
- Settings: {"session_timeout_minutes": 30, "session_save_every_request": true, "debug": false, "conn_max_age": 60, "conn_health_checks": true}

## Failure rates from the access log (last 7 days; Caddy's log starts at its container's start, 2026-11-09T23:04:10 UTC)

- Requests (pages, not static files): 2238
- Server errors (5xx): 1 (0.04%); client errors (4xx): 781 (34.90%)
- Server time, all requests: median 0.004 s, p95 0.080 s, max 6.910 s
  - 502 on `/pipeline/live/`: 1

## Jobs and claims-feed pulls (last 30 days, from the database)

```
{
    "days": 30,
    "jobs": {
        "cpu": {
            "succeeded": 9
        }
    },
    "job_durations": {
        "cpu": {
            "succeeded": 9,
            "mean_seconds": 41.0,
            "mean_seconds_per_claim": 0.767,
            "max_seconds_per_claim": 5.748
        }
    },
    "recent_job_failures": [],
    "ingestion_runs": {
        "succeeded": 1977,
        "failed": 2
    }
}
```

