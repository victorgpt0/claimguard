#!/usr/bin/env bash
# Operational evidence and failure scenarios on the deployed release (Course Sprint III;
# tests/test_plan.md TC-053, TC-054). Run on the VM from anywhere:
#
#   bash /opt/claimguard/deploy/scripts/08-ops-check.sh timing           # response times (safe)
#   bash /opt/claimguard/deploy/scripts/08-ops-check.sh logs             # failure rates from the logs (safe)
#   bash /opt/claimguard/deploy/scripts/08-ops-check.sh failures --yes   # stops services one at a time
#   bash /opt/claimguard/deploy/scripts/08-ops-check.sh restart --yes    # restarts web, then the whole stack
#   sudo reboot; (log back in)
#   bash /opt/claimguard/deploy/scripts/08-ops-check.sh after-reboot     # everything came back by itself
#
# Each run writes ~/claimguard-ops/<time>/report.md: send that file back. It holds no secret.
# "failures" and "restart" take the site down for a minute or two: run them at a quiet time,
# never while a job is running (they refuse if one is queued or running) or while the Colab
# GPU worker is processing. They create no accounts: the checks borrow a session for the
# first active Administrator and Auditor, kept in a 0600 file and deleted at the end.
#
# Options (environment): N=30 requests per page for timing; DAYS=7 for logs;
# DUP_BATCH=<id> of an uploaded batch never started, to also test a double click that would
# really start it (it does start one job, on DUP_QUEUE, default gpu).
set -uo pipefail

DEPLOY="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DEPLOY"
mode="${1:-}"
confirmed="${2:-}"
DOMAIN="$(grep -E "^DOMAIN=" .env 2>/dev/null | cut -d= -f2-)"
BASE="${BASE:-https://${DOMAIN}}"   # BASE overrides it (testing only)
N="${N:-30}"
DAYS="${DAYS:-7}"
OUT="$HOME/claimguard-ops/$(date +%Y%m%dT%H%M%S)-${mode}"
REPORT="$OUT/report.md"
SECRETS="$(umask 077; mktemp -d)"
SESSIONS=()

now() { date +%s.%N; }
since() { python3 -c "import sys; print(f'{float(sys.argv[2]) - float(sys.argv[1]):.1f}')" "$1" "$(now)"; }
rep() { printf '%s\n' "$*" >> "$REPORT"; printf '%s\n' "$*"; }
helper() { docker compose exec -T web python - "$@" < scripts/ops_helper.py 2>/dev/null | tail -1; }
field() { python3 -c "import json,sys
try: v = json.loads(sys.stdin.read()).get(sys.argv[1])
except Exception: v = None
print('' if v is None else v)" "$1"; }
verdict() { if [ "$1" = 0 ]; then echo "**Pass**"; else echo "**FAIL**"; fi; }

cleanup() {
    if [ "${#SESSIONS[@]}" -gt 0 ]; then helper drop "${SESSIONS[@]}" >/dev/null || true; fi
    rm -rf "$SECRETS"
}
trap cleanup EXIT
# Ctrl-C mid-check: delete the borrowed sessions and bring back anything a check stopped.
trap 'echo; echo "Interrupted: starting every service again"; docker compose start >/dev/null 2>&1; exit 130' INT TERM

# A session for the first active user with that role; its Cookie header goes in a 0600 file,
# never on a command line or in the report.
signin() {
    local role="$1" json key
    json="$(helper session "$role")"
    key="$(printf '%s' "$json" | field key)"
    if [ -z "$key" ]; then rep "- No active $role: checks needing one are skipped."; return 1; fi
    SESSIONS+=("$key")
    printf 'Cookie: sessionid=%s\n' "$key" > "$SECRETS/$role.hdr"
    printf '%s' "$key" > "$SECRETS/$role.key"
    return 0
}

# status code and total seconds of one GET; $2 = role whose session to send, or "anon"
get() {
    local url="$1" role="${2:-anon}" hdr=()
    [ "$role" != anon ] && hdr=(-H "@$SECRETS/$role.hdr")
    curl -s -o "$SECRETS/body" -m 30 -w '%{http_code} %{time_total}' "${hdr[@]}" "$BASE$url" || echo "000 30"
}

# seconds until GET $1 (as $2) answers 200; gives up after $3 seconds
wait_200() {
    local url="$1" role="$2" limit="$3" t0 code
    t0="$(now)"
    while :; do
        code="$(get "$url" "$role" | cut -d' ' -f1)"
        [ "$code" = 200 ] && { since "$t0"; return 0; }
        if python3 -c "import sys; sys.exit(float(sys.argv[1]) < float(sys.argv[2]))" "$(since "$t0")" "$limit"; then
            echo "timeout"; return 1
        fi
        sleep 0.5
    done
}

# seconds until the queues named in $1 (comma separated) have a worker; gives up after $2
wait_workers() {
    local want="$1" limit="$2" t0
    t0="$(now)"
    while :; do
        if helper workers | python3 -c "import json,sys
online = json.loads(sys.stdin.read() or '{}')
sys.exit(0 if all(q in online for q in sys.argv[1].split(',')) else 1)" "$want"; then
            since "$t0"; return 0
        fi
        if python3 -c "import sys; sys.exit(float(sys.argv[1]) < float(sys.argv[2]))" "$(since "$t0")" "$limit"; then
            echo "timeout"; return 1
        fi
        sleep 2
    done
}

require_quiet() {
    if [ "$confirmed" != "--yes" ]; then
        echo "'$mode' stops or restarts services on the live site. Run it again with --yes."; exit 2
    fi
    local active
    active="$(helper active | field active)"
    if [ "$active" != 0 ]; then
        echo "Refusing: $active job(s) queued or running. Wait until they finish."; exit 2
    fi
}

header() {
    mkdir -p "$OUT"
    rep "# ClaimGuard operational check: $mode"
    rep ""
    rep "- Run: $(date -Is) on $(hostname), deployed commit $(git -C .. rev-parse --short HEAD)"
    rep "- Target: $BASE (requests from the VM go out through Cloudflare and back, as a visitor's do)"
    rep "- Settings: $(helper config)"
    rep ""
}

inventory() {
    rep "## The tested setup (for docs/environment.md)"
    rep ""
    rep "- OS: $(. /etc/os-release; echo "$PRETTY_NAME") $(uname -m), kernel $(uname -r)"
    rep "- CPU: $(nproc) cores ($(lscpu | sed -n 's/^Model name: *//p' | head -1)); memory: $(free -h | awk '/^Mem:/ {print $2 " total, " $7 " available"}'); swap $(free -h | awk '/^Swap:/ {print $2}')"
    rep "- Disk /: $(df -h / | awk 'NR == 2 {print $2 " total, " $4 " free"}')"
    rep "- $(docker --version); $(docker compose version)"
    rep ""
    rep '```'
    rep "$(docker compose images)"
    rep "$(docker compose exec -T db postgres --version 2>/dev/null); $(docker compose exec -T redis redis-server --version 2>/dev/null | cut -d' ' -f1-3)"
    rep "$(docker compose exec -T web python -c 'import django, sys; print("Python", sys.version.split()[0], "Django", django.get_version())' 2>/dev/null)"
    rep '```'
    rep ""
}

timing() {
    inventory
    signin Administrator || true
    signin Auditor || true
    local batch report
    batch="$(helper batch | field batch)"
    report="$(helper report | field report)"
    : > "$OUT/timing.txt"
    local pages=("/accounts/login/ anon" "/audit/ Administrator" "/audit/batches/ Administrator"
                 "/pipeline/ Administrator" "/audit/model/ Administrator" "/audit/queue/ Auditor")
    [ -n "$batch" ] && pages+=("/audit/batches/$batch/ Administrator")
    [ -n "$report" ] && pages+=("/audit/reports/$report/ Auditor")
    rep "## Response times ($N sequential requests per page)"
    rep ""
    for entry in "${pages[@]}"; do
        set -- $entry
        [ "$2" != anon ] && [ ! -f "$SECRETS/$2.hdr" ] && continue
        for _ in $(seq "$N"); do printf '%s %s %s\n' "$1" "$2" "$(get "$1" "$2")" >> "$OUT/timing.txt"; done
    done
    python3 - "$OUT/timing.txt" <<'EOF' | tee -a "$REPORT"
import re, statistics, sys
rows = {}
for line in open(sys.argv[1]):
    url, role, code, secs = line.split()
    rows.setdefault((re.sub(r"/\d+/", "/<id>/", url), role), []).append((code, float(secs)))
def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]
print("| Page | As | n | Not 200 | Median s | p95 s | Max s |")
print("|---|---|---|---|---|---|---|")
total = bad = 0
for (url, role), xs in rows.items():
    times = [s for _, s in xs]; failed = sum(c != "200" for c, _ in xs)
    total += len(xs); bad += failed
    print(f"| `{url}` | {role} | {len(xs)} | {failed} | {statistics.median(times):.3f} | {pct(times, 95):.3f} | {max(times):.3f} |")
print(f"\nAll pages: {total} requests, {bad} not answered 200 ({100 * bad / max(total, 1):.1f}%).")
EOF
    rep ""
}

logs() {
    local started
    started="$(docker inspect -f '{{.State.StartedAt}}' "$(docker compose ps -q caddy)" 2>/dev/null | cut -c1-19)"
    rep "## Failure rates from the access log (last $DAYS days; Caddy's log starts at its container's start, $started UTC)"
    rep ""
    docker compose logs caddy --since "$((DAYS * 24))h" --no-log-prefix 2>/dev/null | python3 -c '
import json, re, statistics, sys
reqs = []
for line in sys.stdin:
    if "handled request" not in line or "{" not in line:
        continue
    try:
        e = json.loads(line[line.index("{"):])
    except ValueError:
        continue
    uri = e.get("request", {}).get("uri", "")
    if uri.startswith("/static/"):
        continue
    reqs.append((int(e.get("status", 0)), float(e.get("duration", 0)), re.sub(r"\d+", "<n>", uri.split("?")[0])))
if not reqs:
    print("No access-log lines found."); sys.exit()
n = len(reqs); s5 = sum(s >= 500 for s, _, _ in reqs); s4 = sum(400 <= s < 500 for s, _, _ in reqs)
d = sorted(x for _, x, _ in reqs)
print(f"- Requests (pages, not static files): {n}")
print(f"- Server errors (5xx): {s5} ({100 * s5 / n:.2f}%); client errors (4xx): {s4} ({100 * s4 / n:.2f}%)")
print(f"- Server time, all requests: median {statistics.median(d):.3f} s, p95 {d[int(0.95 * (n - 1))]:.3f} s, max {d[-1]:.3f} s")
errors = {}
for s, _, u in reqs:
    if s >= 500: errors[(s, u)] = errors.get((s, u), 0) + 1
for (s, u), k in sorted(errors.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  - {s} on `{u}`: {k}")
' | tee -a "$REPORT"
    rep ""
    rep "## Jobs and claims-feed pulls (last 30 days, from the database)"
    rep ""
    rep '```'
    rep "$(helper stats 30 | python3 -m json.tool)"
    rep '```'
    rep ""
}

failures() {
    require_quiet
    signin Administrator || { rep "An Administrator is needed for these checks."; exit 1; }
    local r code secs body t
    rep "## Failure scenarios"
    rep ""

    rep "### F1. Expired and invalid sessions"
    signin Administrator   # a second session, only for this test
    local key; key="$(cat "$SECRETS/Administrator.key")"
    code="$(get /audit/ Administrator | cut -d' ' -f1)"
    helper expire "$key" >/dev/null
    r="$(curl -s -o /dev/null -m 30 -w '%{http_code} %{redirect_url}' -H "@$SECRETS/Administrator.hdr" "$BASE/audit/")"
    printf 'Cookie: sessionid=notarealsession\n' > "$SECRETS/bogus.hdr"
    local r2; r2="$(curl -s -o /dev/null -m 30 -w '%{http_code} %{redirect_url}' -H "@$SECRETS/bogus.hdr" "$BASE/audit/batches/")"
    local r3; r3="$(curl -s -o /dev/null -m 30 -w '%{http_code}' -X POST -H "@$SECRETS/bogus.hdr" -d queue=cpu "$BASE/audit/batches/1/process/")"
    [[ "$code" = 200 && "$r" == 302*"/accounts/login/?next=/audit/"* && "$r2" == 302*"/accounts/login/"* && "$r3" = 403 ]]; t=$?
    rep "- Before expiry /audit/: $code. After expiry: $r. Unknown session on /audit/batches/: $r2. POST without a CSRF token: $r3 (403 expected, not 500). $(verdict $t)"
    signin Administrator   # a fresh session for the rest
    rep ""

    rep "### F2. Duplicate submission (two POSTs at once to start the same batch)"
    local batch jar csrf before after c1 c2
    batch="$(helper batch | field batch)"
    if [ -z "$batch" ]; then rep "- No processed batch: skipped."; else
        jar="$SECRETS/jar"
        curl -s -o /dev/null -m 30 -c "$jar" -H "@$SECRETS/Administrator.hdr" "$BASE/audit/batches/$batch/"
        csrf="$(awk '$6 == "csrftoken" {print $7}' "$jar" | tail -1)"
        if [ -z "$csrf" ]; then rep "- No CSRF cookie on the batch page: skipped."; else
            start_twice() {   # batch queue -> "code1 code2"
                local hdr="$SECRETS/post.hdr"
                printf 'Cookie: sessionid=%s; csrftoken=%s\nX-CSRFToken: %s\nOrigin: %s\nReferer: %s/audit/batches/%s/\n' \
                    "$(cat "$SECRETS/Administrator.key")" "$csrf" "$csrf" "$BASE" "$BASE" "$1" > "$hdr"
                curl -s -o /dev/null -m 30 -w '%{http_code}' -H "@$hdr" -d "queue=$2" "$BASE/audit/batches/$1/process/" > "$SECRETS/c1" &
                curl -s -o /dev/null -m 30 -w '%{http_code}' -H "@$hdr" -d "queue=$2" "$BASE/audit/batches/$1/process/" > "$SECRETS/c2" &
                wait
                echo "$(cat "$SECRETS/c1") $(cat "$SECRETS/c2")"
            }
            before="$(helper jobs "$batch" | field jobs)"
            read -r c1 c2 <<< "$(start_twice "$batch" cpu)"
            after="$(helper jobs "$batch" | field jobs)"
            [[ "$c1" = 302 && "$c2" = 302 && "$before" = "$after" ]]; t=$?
            rep "- Processed batch #$batch: both answered $c1 / $c2 (302 expected); jobs before $before, after $after (no new job expected). $(verdict $t)"
            if [ -n "${DUP_BATCH:-}" ]; then
                before="$(helper jobs "$DUP_BATCH" | field jobs)"
                read -r c1 c2 <<< "$(start_twice "$DUP_BATCH" "${DUP_QUEUE:-gpu}")"
                after="$(helper jobs "$DUP_BATCH" | field jobs)"
                [[ "$c1" = 302 && "$c2" = 302 && $((after - before)) = 1 ]]; t=$?
                rep "- Uploaded batch #$DUP_BATCH, never started: $c1 / $c2; jobs before $before, after $after (exactly one new job expected, on ${DUP_QUEUE:-gpu}). $(verdict $t)"
            fi
        fi
    fi
    rep ""

    rep "### F3. CPU worker unavailable"
    docker compose stop worker-cpu >/dev/null 2>&1
    local online; online="$(helper workers)"
    read -r code secs <<< "$(get /audit/batches/ Administrator)"
    docker compose start worker-cpu >/dev/null 2>&1
    local back; back="$(wait_workers cpu 180)"
    [[ "$online" != *'"cpu"'* && "$code" = 200 && "$back" != timeout ]]; t=$?
    rep "- Worker stopped: queues online $online (no cpu expected); /audit/batches/ $code in ${secs}s. Started again: online after ${back}s. $(verdict $t)"
    rep "- A job started while no worker is online waits in the queue and the page says so (tests: test_a_batch_is_processed_once)."
    rep ""

    rep "### F4. Job queue (Redis) unavailable"
    docker compose stop redis >/dev/null 2>&1
    local pages_ok=0 line=""
    for u in /audit/ /audit/batches/ /pipeline/; do
        read -r code secs <<< "$(get "$u" Administrator)"
        line="$line \`$u\` $code in ${secs}s;"
        [ "$code" = 200 ] || pages_ok=1
    done
    online="$(helper workers)"
    docker compose start redis >/dev/null 2>&1
    back="$(wait_workers cpu,ingest 240)"
    [[ "$pages_ok" = 0 && "$online" = "{}" && "$back" != timeout ]]; t=$?
    rep "- Redis stopped:$line queues online $online ({} expected). Started again: workers reconnected after ${back}s. $(verdict $t)"
    rep "- Starting a job while Redis is down marks the job and batch failed with the reason, and the batch can be started again (fixed 10 Nov; tests: test_a_job_the_queue_cannot_take_fails_visibly_and_can_be_retried, test_a_closed_period_the_queue_cannot_take_fails_visibly)."
    rep ""

    rep "### F5. Claims platform (sha-cdp) unavailable"
    docker compose stop sha-cdp >/dev/null 2>&1
    read -r code secs <<< "$(get /pipeline/ Administrator)"
    local live; live="$(get /pipeline/live/ Administrator | cut -d' ' -f1)"
    grep -q unreachable "$SECRETS/body"; local said=$?
    docker compose start sha-cdp >/dev/null 2>&1
    [[ "$code" = 200 && "$live" = 200 && "$said" = 0 ]]; t=$?
    rep "- Platform stopped: /pipeline/ $code in ${secs}s; live panel $live and says \"unreachable\": $([ $said = 0 ] && echo yes || echo no). $(verdict $t)"
    rep ""

    rep "### F6. Database unavailable"
    docker compose stop db >/dev/null 2>&1
    read -r code secs <<< "$(get /accounts/login/ anon)"
    grep -q "Something went wrong on our side" "$SECRETS/body"; local friendly=$?
    grep -q -i "traceback" "$SECRETS/body"; local leak=$?
    local code2 secs2; read -r code2 secs2 <<< "$(get /audit/ Administrator)"
    docker compose start db >/dev/null 2>&1
    back="$(wait_200 /audit/ Administrator 180)"
    local wback; wback="$(wait_workers cpu,ingest 180)"
    [[ "$code" = 500 && "$friendly" = 0 && "$leak" = 1 && "$back" != timeout && "$wback" != timeout ]]; t=$?
    rep "- Database stopped: /accounts/login/ $code in ${secs}s, /audit/ $code2 in ${secs2}s; the error page is ClaimGuard's own: $([ $friendly = 0 ] && echo yes || echo no); a traceback shown: $([ $leak = 0 ] && echo YES || echo no). Started again: /audit/ answered 200 after ${back}s without restarting the web container, the session still valid; workers online after ${wback}s. $(verdict $t)"
    rep ""
}

restart() {
    require_quiet
    local before after
    before="$(helper counts)"
    rep "## Restarts"
    rep ""
    restart_one() {   # label, compose args...
        local label="$1"; shift
        local log="$SECRETS/poll" t0 cmd_done
        : > "$log"
        ( while :; do printf '%s %s\n' "$(date +%s.%N)" "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$BASE/accounts/login/" || echo 000)" >> "$log"; sleep 0.5; done ) &
        local poller=$!
        t0="$(now)"
        docker compose restart "$@" >/dev/null 2>&1
        cmd_done="$(since "$t0")"
        local up; up="$(wait_200 /accounts/login/ anon 300)"
        kill "$poller" 2>/dev/null; wait "$poller" 2>/dev/null
        python3 - "$log" "$t0" "$label" "$cmd_done" <<'EOF' | tee -a "$REPORT"
import sys
rows = [l.split() for l in open(sys.argv[1]) if len(l.split()) == 2]
t0 = float(sys.argv[2]); bad = [(float(t), c) for t, c in rows if c != "200"]
codes = sorted({c for _, c in bad})
if bad:
    first_bad = bad[0][0]; last_bad = bad[-1][0]
    ok_after = next((float(t) for t, c in rows if c == "200" and float(t) > last_bad), None)
    down = f"{(ok_after or last_bad) - first_bad:.1f} s unavailable" + (f" (answers seen: {', '.join(codes)})" if codes else "")
else:
    down = "never unavailable to the poller"
print(f"- {sys.argv[3]}: restart command took {sys.argv[4]} s; {down}; {len(bad)} of {len(rows)} polls failed.")
EOF
    }
    restart_one "Web application (docker compose restart web)" web
    restart_one "Whole stack (docker compose restart)"
    local wback; wback="$(wait_workers cpu,ingest 300)"
    after="$(helper counts)"
    [[ "$before" = "$after" && "$wback" != timeout ]]; local t=$?
    rep "- Workers online again ${wback}s after the stack was back. Rows before $before; after $after (unchanged expected). $(verdict $t)"
    rep ""
    rep "Next: \`sudo reboot\`, log back in, then run this script with \`after-reboot\`."
    rep ""
}

after_reboot() {
    signin Administrator || true
    rep "## After a reboot"
    rep ""
    rep "- Booted at $(uptime -s); checked $(date '+%F %T')."
    local mounted; mountpoint -q /srv/gdrive && mounted=yes || mounted=no
    rep "- Google Drive mounted at /srv/gdrive: $mounted."
    rep ""
    rep '```'
    rep "$(docker compose ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}')"
    rep '```'
    local code secs; read -r code secs <<< "$(get /accounts/login/ anon)"
    local wback; wback="$(wait_workers cpu,ingest 300)"
    local counts; counts="$(helper counts)"
    [[ "$mounted" = yes && "$code" = 200 && "$wback" != timeout ]]; local t=$?
    rep "- Login page $code in ${secs}s; workers online (cpu, ingest) after ${wback}s more; rows $counts. Nothing was started by hand. $(verdict $t)"
    rep ""
}

case "$mode" in
    timing) header; timing ;;
    logs) header; logs ;;
    failures) header; failures ;;
    restart) header; restart ;;
    after-reboot) header; after_reboot ;;
    *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
echo
echo "Report: $REPORT"
