#!/usr/bin/env bash
# Deploys one commit of main on the VM: check it out, build the images, restart what changed,
# and wait until the site answers. GitHub Actions runs it over SSH after a merge to main
# (.github/workflows/deploy.yml; the key 07-deploy-access.sh makes can run nothing else).
# By hand:  bash deploy.sh            the latest main
#           bash deploy.sh <sha>      a given commit of main (e.g. to go back to an older one)
#
# A failed build changes nothing: the running containers stay as they were and the checkout
# goes back. A failed health check leaves the new version running and fails loudly; it is not
# rolled back automatically, because the new version may already have migrated the database.
set -euo pipefail

main() {
    local app_dir deploy branch sha prev domain
    app_dir="$(cd "$(dirname "$0")/../.." && pwd)"
    deploy="$app_dir/deploy"
    branch="${DEPLOY_BRANCH:-main}"
    # Over the restricted SSH key the commit arrives as the "command" the client asked for.
    sha="${1:-${SSH_ORIGINAL_COMMAND:-}}"

    exec 9>/tmp/claimguard-deploy.lock
    flock -w 1800 9 || { echo "Another deploy is still running."; exit 1; }

    cd "$app_dir"
    git fetch -q origin "$branch"
    if [ -z "$sha" ]; then
        sha="$(git rev-parse "origin/$branch")"
    fi
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || { echo "Expected a full 40-character commit hash, got: $sha"; exit 2; }
    git merge-base --is-ancestor "$sha" "origin/$branch" \
        || { echo "$sha is not on $branch; only $branch is deployed."; exit 2; }
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "Tracked files were edited on the VM; commit those changes in the repository instead:"
        git status --short --untracked-files=no
        exit 3
    fi

    prev="$(git rev-parse HEAD)"
    echo "==> Deploying $(git log -1 --format='%h %s' "$sha")  (was ${prev:0:7})"
    git checkout -q -B "$branch" "$sha"

    cd "$deploy"
    echo "==> Building images"
    if ! docker compose build; then
        echo "Build failed; going back to ${prev:0:7}. The running site was not touched."
        git -C "$app_dir" checkout -q -B "$branch" "$prev"
        exit 4
    fi

    echo "==> Starting (the web container applies migrations before it serves)"
    docker compose up -d --remove-orphans

    domain="$(sed -n 's/^DOMAIN=//p' .env | tr -d '"')"
    echo "==> Waiting for the web application"
    for _ in $(seq 60); do
        if docker compose exec -T web python - "$domain" >/dev/null 2>&1 <<'PY'
import sys, urllib.request
req = urllib.request.Request("http://127.0.0.1:8000/accounts/login/",
                             headers={"Host": sys.argv[1], "X-Forwarded-Proto": "https"})
assert urllib.request.urlopen(req, timeout=5).status == 200
PY
        then
            printf '%s %s %s\n' "$(date -Is)" "$sha" "$(git -C "$app_dir" log -1 --format=%s "$sha")" \
                >> "$deploy/.deploy-history"
            docker compose ps --format 'table {{.Service}}\t{{.Status}}'
            docker image prune -f >/dev/null
            echo "==> Deployed ${sha:0:7}"
            exit 0
        fi
        sleep 5
    done

    echo "The web application did not answer within 5 minutes. Recent logs:"
    docker compose ps
    docker compose logs --tail 60 web
    exit 5
}

# Everything runs inside main(), which bash reads in full before starting, so the checkout
# replacing this file mid-run cannot change what runs.
main "$@"
