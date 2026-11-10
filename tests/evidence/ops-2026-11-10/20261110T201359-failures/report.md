# ClaimGuard operational check: failures

- Run: 2026-11-10T20:13:59+00:00 on claimguard-instance-vnic, deployed commit 49046c6
- Target: https://claimsguard.tech (requests from the VM go out through Cloudflare and back, as a visitor's do)
- Settings: {"session_timeout_minutes": 30, "session_save_every_request": true, "debug": false, "conn_max_age": 60, "conn_health_checks": true}

## Failure scenarios

### F1. Expired and invalid sessions
- Before expiry /audit/: 200. After expiry: 302 https://claimsguard.tech/accounts/login/?next=/audit/. Unknown session on /audit/batches/: 302 https://claimsguard.tech/accounts/login/?next=/audit/batches/. POST without a CSRF token: 403 (403 expected, not 500). **Pass**

### F2. Duplicate submission (two POSTs at once to start the same batch)
- Processed batch #5: both answered 302 / 302 (302 expected); jobs before 1, after 1 (no new job expected). **Pass**

### F3. CPU worker unavailable
- Worker stopped: queues online {"ingest": ["ingest@d0acb92cb2e4"]} (no cpu expected); /audit/batches/ 200 in 0.082311s. Started again: online after 7.9s. **Pass**
- A job started while no worker is online waits in the queue and the page says so (tests: test_a_batch_is_processed_once).

### F4. Job queue (Redis) unavailable
- Redis stopped: `/audit/` 200 in 0.164898s; `/audit/batches/` 200 in 0.094563s; `/pipeline/` 200 in 0.108293s; queues online {} ({} expected). Started again: workers reconnected after 7.2s. **Pass**
- Starting a job while Redis is down marks the job and batch failed with the reason, and the batch can be started again (fixed 10 Nov; tests: test_a_job_the_queue_cannot_take_fails_visibly_and_can_be_retried, test_a_closed_period_the_queue_cannot_take_fails_visibly).

### F5. Claims platform (sha-cdp) unavailable
- Platform stopped: /pipeline/ 200 in 0.077196s; live panel 200 and says "unreachable": yes. **Pass**

### F6. Database unavailable
- Database stopped: /accounts/login/ 500 in 0.115450s, /audit/ 500 in 0.095469s; the error page is ClaimGuard's own: yes; a traceback shown: no. Started again: /audit/ answered 200 after 0.8s without restarting the web container, the session still valid; workers online after 2.6s. **Pass**

