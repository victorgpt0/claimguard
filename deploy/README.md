# Deploying ClaimGuard on Oracle Cloud

The whole platform on one Oracle Cloud **Always Free** ARM VM (home region UK South,
`uk-london-1`), served at **https://claimsguard.tech** with Cloudflare in front. You run every
step yourself; each script is safe to run again, so a new VM is rebuilt by following the same
steps (keep a copy of `deploy/.env`, the CA key and the origin key, or new ones are made).

**`main` is what is deployed.** Work happens on `dev`; a pull request from `dev` into `main`
runs the tests, and merging it deploys automatically (`.github/workflows/deploy.yml`, step 10).

```
Visitors ─HTTPS─▶ Cloudflare (proxied: claimsguard.tech, www) ─HTTPS─▶ VM :443 Caddy ─▶ web (gunicorn)
                                                  Cloudflare Origin Certificate, "Full (strict)"
VM, docker compose (deploy/compose.yaml):
  caddy · web · sha-cdp (internal only) · postgres 18 · redis
  worker-ingest (claims feed + beat) · worker-cpu (stage 1, SHAP, classifier)
  /srv/gdrive = Google Drive (rclone), mounted into the app containers
Colab GPU worker ─TLS─▶ VM :5432 PostgreSQL, :6380 Redis (password + the deployment's own CA)
```

| Port | Open to | Why |
|---|---|---|
| 22 | anywhere (keys only, fail2ban); GitHub Actions deploys through it | SSH |
| 443 | anywhere in the security list; Caddy drops all but Cloudflare's addresses | the site |
| 5432, 6380 | anywhere, TLS + password required | the Colab GPU worker |

## Limits to respect

- **Always Free Ampere A1 is 2 OCPU and 12 GB in total** (Oracle cut it from 4/24;
  [Oracle's page](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)).
  Never create more than that: forum reports say over-limit A1 instances are disabled when the
  trial ends. Block storage: 200 GB in total, in the home region.
- **Idle reclamation:** Oracle may reclaim an Always Free VM whose CPU, network and memory all
  stay under 20% for 7 days. A quiet demo can hit that. Upgrading the account to **Pay As You
  Go** removes the risk and costs nothing within the free limits; add a budget alert
  (Billing → Budgets, e.g. USD 1) so anything outside them is noticed.
- **The trial's credits** end after 30 days; everything here is Always Free, so nothing stops.

## 0. Before you start (laptop)

1. Put the code on `main` (the VM clones it): push `dev`, open a pull request `dev` → `main`
   on GitHub and merge it once the tests pass. The deploy job is skipped until step 10.
2. Have ready: the Neon **direct** URL (`DATABASE_URL_UNPOOLED` in the laptop's `.env`) and
   `rclone config show gdrive` output (step 3).

## 1. Oracle Cloud console

1. **Network:** Networking → Virtual cloud networks → *Start VCN Wizard* → *Create VCN with
   Internet Connectivity* → name `claimguard-vcn`, defaults otherwise.
2. **Instance:** Compute → Instances → *Create instance*:
   - Name `claimguard`; Image: *Canonical Ubuntu 24.04* (the aarch64 build is chosen with the shape).
   - Shape: *Ampere* → `VM.Standard.A1.Flex`, **2 OCPU, 12 GB**. The "Always Free-eligible"
     label must show.
   - Networking: `claimguard-vcn`, its **public** subnet, assign a public IPv4 address.
   - SSH keys: upload your public key (`~/.ssh/id_ed25519.pub`).
   - Boot volume: *Specify a custom size*, **100 GB**.
   - "Out of host capacity": try another availability domain, or retry later; PAYG accounts get capacity more readily.
3. **Reserved public IP** (so the address, and the database certificate, never change):
   Networking → IP management → Reserved public IPs → *Reserve* (`claimguard-ip`). Then
   Instance → Attached VNICs → the VNIC → IPv4 addresses → ⋯ Edit → *No public IP* → Update,
   then Edit again → *Reserved public IP* → `claimguard-ip`.
4. **Security list:** VCN → Security Lists → *Default Security List* → Add ingress rules
   (source `0.0.0.0/0`, TCP): destination ports `443`, `5432`, `6380`. Leave the existing
   port-22 rule open to `0.0.0.0/0`: GitHub's runners connect from changing addresses.

## 2. Bootstrap the VM

```bash
scp claimguard/deploy/scripts/01-bootstrap-vm.sh ubuntu@<ip>:
ssh ubuntu@<ip> 'bash 01-bootstrap-vm.sh'
```

Updates, Docker, rclone, 4 GB swap, keys-only SSH with fail2ban, and the VM's own iptables
opened for 80/443/5432/6380 (Oracle's Ubuntu images reject everything but SSH there, on top of
the security list). It prints a **deploy key**: add it on GitHub (repository → Settings → Deploy
keys, read-only) and press Enter; it clones `main` into `/opt/claimguard`. Log out and back in.

## 3. Google Drive

```bash
bash /opt/claimguard/deploy/scripts/02-drive-mount.sh
```

Paste the laptop's `rclone config show gdrive` output when asked (it contains a Google refresh
token: treat it as a password). Drive is mounted at `/srv/gdrive` by the `rclone-gdrive`
service, before Docker starts. The pipeline reads the active classifier, datasets and ICD-11
cache from there; without it the claims feed fails ("icd11_cache.json not found").

## 4. Settings and certificates

```bash
bash /opt/claimguard/deploy/scripts/03-configure.sh <reserved-ip>
```

Writes `deploy/.env` (random secrets; domain `claimsguard.tech`), `config/config.yaml`
(config.example.yaml with Drive paths), and a private CA plus a PostgreSQL/Redis certificate for
the reserved IP. Optional: fill in `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET` in `deploy/.env`.

## 5. Cloudflare

1. Cloudflare dashboard → *Add a domain* → `claimsguard.tech` → Free plan. At your registrar,
   replace the nameservers with the two Cloudflare gives you; wait for "Active".
2. DNS → records (both **Proxied**, orange cloud): `A  claimsguard.tech → <reserved-ip>` and
   `CNAME  www → claimsguard.tech`. Delete any old A/AAAA/CNAME for these names.
3. SSL/TLS → Overview → **Full (strict)**. Edge Certificates → *Always Use HTTPS* on, minimum
   TLS 1.2. (Leave HSTS off until the site has run cleanly for a while: it cannot be undone quickly.)
4. SSL/TLS → Origin Server → *Create Certificate* (RSA, hostnames `claimsguard.tech`,
   `*.claimsguard.tech`, 15 years). On the VM, paste the two blocks into:
   ```bash
   nano /opt/claimguard/deploy/origin/origin.pem   # Origin Certificate
   nano /opt/claimguard/deploy/origin/origin.key   # Private key (shown once)
   chmod 600 /opt/claimguard/deploy/origin/origin.key
   ```
5. `bash /opt/claimguard/deploy/scripts/04-cloudflare-ips.sh` (writes Cloudflare's ranges into `.env`).
6. Google sign-in, if used: add `https://claimsguard.tech/accounts/google/login/callback/` to the
   OAuth client's authorised redirect URIs.

## 6. Move the data from Neon (once, before the first full start)

```bash
cd /opt/claimguard/deploy
NEON_URL='postgresql://...neon.tech/neondb?sslmode=require' bash scripts/05-migrate-from-neon.sh
```

Starts only PostgreSQL, refuses if it already has tables, dumps Neon with pg_dump 18, restores,
and prints row counts side by side. Delete the `neon-*.dump` file afterwards (it holds password
hashes). Starting fresh instead? Skip this step and run `createsuperuser` in step 7.

## 7. Start

```bash
cd /opt/claimguard/deploy
docker compose up -d --build        # the first build takes a while on 2 OCPU (torch)
docker compose ps
docker compose logs -f web worker-ingest worker-cpu
```

- Fresh database only: `docker compose exec web python web/manage.py createsuperuser`.
- Claims for the feed, once `live-v1` exists on Drive:
  `docker compose run --rm sha-cdp python -m sha_cdp.seed /srv/gdrive/claimguard/data/<feed>.parquet`

## 8. The Colab GPU worker

1. Copy the CA certificate to Drive (the VM sees Drive at `/srv/gdrive`):
   `mkdir -p /srv/gdrive/claimguard/deploy && cp /opt/claimguard/deploy/tls/ca.crt /srv/gdrive/claimguard/deploy/`
2. In Colab's Secrets (or `MyDrive/claimguard/.env`), with the passwords from `deploy/.env`:
   ```
   DATABASE_URL=postgres://claimguard:<POSTGRES_PASSWORD>@<reserved-ip>:5432/claimguard?sslmode=verify-full&sslrootcert=/content/drive/MyDrive/claimguard/deploy/ca.crt
   REDIS_URL=rediss://:<REDIS_PASSWORD>@<reserved-ip>:6380/0?ssl_ca_certs=/content/drive/MyDrive/claimguard/deploy/ca.crt
   ```
   Both the notebook's redis ping and Celery read `ssl_ca_certs` from the URL (tested).
3. Run `notebooks/05_gpu_worker.ipynb`; its check cell should show the migrations and `redis ping: True`.

## 9. Backups

```bash
bash /opt/claimguard/deploy/scripts/06-backups.sh
```

Nightly at 02:30 (Nairobi) both databases are dumped to `/var/backups/claimguard` (14 days) and
`MyDrive/claimguard/backups/oracle` (30 days); Cloudflare's ranges refresh monthly.
Restore: `docker compose exec -T db pg_restore -U claimguard -d claimguard --clean --no-owner < <file>.dump`.

## 10. Automatic deploys from `main`

```bash
bash /opt/claimguard/deploy/scripts/07-deploy-access.sh <reserved-ip>
```

It makes an SSH key that can only run `deploy/scripts/deploy.sh` on the VM (no shell), and prints
three values. Add each on GitHub → repository → Settings → Secrets and variables → Actions →
*New repository secret*: `DEPLOY_HOST`, `DEPLOY_KNOWN_HOSTS`, `DEPLOY_SSH_KEY`. Then Actions →
*Deploy* → *Run workflow* once to check it.

From then on:

1. Work on `dev` and push it (the tests run on every push).
2. Open a pull request `dev` → `main`. The tests run on the pull request.
3. Merge it. The *Deploy* workflow runs the tests on `main`, then over SSH the VM fetches that
   commit, builds the images, restarts what changed (the web container migrates first), waits
   for the login page, and the workflow checks https://claimsguard.tech.

If the build fails, the site keeps running the previous version. If the new version starts but
does not answer, the job fails with the web logs and the new version stays up (no automatic
rollback, because it may already have migrated the database): fix forward with another pull
request, or go back by hand with `bash /opt/claimguard/deploy/scripts/deploy.sh <older-sha>`
(only for commits without migrations). `deploy/.deploy-history` lists what was deployed when.

Optional, if your plan offers it for private repositories: Settings → Rules → a ruleset for
`main` requiring a pull request and the *Tests* check, so nothing reaches the site unreviewed.

## 11. Check before announcing the address

- [ ] https://claimsguard.tech loads with a valid padlock; `www` redirects to it.
- [ ] Skipping Cloudflare fails: `curl -kv https://<reserved-ip>/ -H 'Host: claimsguard.tech'` is cut off.
- [ ] A plain database connection is refused: `psql "postgres://claimguard:...@<ip>:5432/claimguard?sslmode=disable"` → *pg_hba.conf rejects*.
- [ ] Sign in; upload a small batch on the CPU queue; the job finishes (`docker compose logs worker-cpu`).
- [ ] Data pipeline page: pulls succeed (`docker compose logs worker-ingest`).
- [ ] Users page: no test accounts, nobody with a known password.
- [ ] One GPU job from Colab (step 8).
- [ ] After a week without problems, retire Neon and Upstash.

## Day to day

| Task | Command (in `/opt/claimguard/deploy`) |
|---|---|
| Deploy | merge a pull request into `main` (automatic); by hand: `bash scripts/deploy.sh` |
| Logs | `docker compose logs -f --tail 100 <service>` |
| Restart one service | `docker compose restart <service>` |
| After rclone restarts | `docker compose restart` (containers keep the old, empty mount otherwise) |
| What is deployed | `tail .deploy-history` |
| Shell / Django admin commands | `docker compose exec web python web/manage.py <command>` |
| Disk use | `df -h /; docker system df` (`docker image prune` after updates) |
| Rotate a secret | edit `.env`, `docker compose up -d`; for the DB password also `ALTER ROLE ... PASSWORD` |

Certificates: the Cloudflare origin certificate lasts 15 years; the database certificate 825 days
(delete `deploy/tls/` and rerun `03-configure.sh`, then re-copy `ca.crt` to Drive).
rclone's shared Google client ID is being retired during 2026: if the mount stops authenticating,
create your own client ID (`rclone config`) on the laptop and redo step 3.

## For Chapter 5

Describe what runs: "the web application, database, queue and CPU worker on an Oracle Cloud
Always Free VM (2 OCPU, 12 GB, UK South) behind Cloudflare; GPU jobs on Colab on demand". Do
not show `deploy/.env`, the origin key, the CA key or any password in screenshots or the report.
