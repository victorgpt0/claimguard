# Free Hosting Guide

Two legitimate, genuinely $0 paths to put the ClaimGuard dashboard on your own domain.
Both were checked against current provider docs before writing this (not guessed) —
specifically: **Hugging Face Spaces' free tier does not support custom domains** (that
needs a paid Pro plan), which is why Option B below routes through a Cloudflare Worker
instead of pointing your domain at Spaces directly.

Pick based on what you actually need:

| | Option A: Cloudflare Tunnel | Option B: HF Spaces + Cloudflare Worker |
|---|---|---|
| Uptime | Only while your machine + tunnel are running | Always-on |
| Setup complexity | Low | Medium |
| Where it runs | Your own machine | Hugging Face's servers (2 vCPU/16GB free tier — more headroom than your laptop) |
| Good for | Thesis defense demos, supervisor walkthroughs | A dashboard you want reachable any time |
| Cost | $0 (domain already owned) | $0 (domain already owned) |

Whichever you pick, **be accurate about it in Chapter 5** — `research_writing.txt` line
994 is explicit: don't describe a demo-only or tunnel-when-running setup as "publicly
available" if it isn't actually up all the time.

## How training, storage, and deployment fit together

```
Colab (GPU, training)  ──writes──>  Google Drive (models/, data/, experiments/ — DR-05)
        │
        └──pushes trained adapter──>  Hugging Face Hub (private model repo)
                                              │
                                              ▼
                              Deployed dashboard loads the LoRA adapter from HF Hub,
                              not from Drive directly — neither Cloudflare Tunnel's
                              target machine nor an HF Space can mount your Google
                              Drive, but both can pull from a private HF Hub repo via
                              HF_TOKEN.
```

Drive stays your durable artifact store (per DR-05 and the implementation roadmap) —
this doesn't change. HF Hub is just the hand-off point that gets the *specific active
checkpoint* to wherever the dashboard actually runs. After `ClassifierRetrainer` promotes
a candidate (FR-14), also run:

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path=config["paths"]["model_dir"] + "/classifier_lora/active",
    repo_id="<your-username>/claimguard-classifier",
    repo_type="model",
    private=True,
)
```

The ICD-11 cache (`artifacts/icd11_cache.json`, built via `src/icd11/refresh_cache.py`)
travels the same way — it's small (plain JSON), commit it into the deployed app's
own repo/Space directly rather than routing it through HF Hub.

## Option A: Cloudflare Tunnel (self-hosted, simplest)

Exposes the Streamlit app running **on your own machine** at your domain, with
Cloudflare's SSL/CDN/DDoS protection in front of it, without opening any router ports.

1. **Point your domain at Cloudflare** (skip if already done): add the domain to your
   Cloudflare account, update your registrar's nameservers to Cloudflare's.

2. **Install `cloudflared`:**
   ```bash
   # Debian/Ubuntu:
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared.deb
   ```

3. **Authenticate and create the tunnel:**
   ```bash
   cloudflared tunnel login
   cloudflared tunnel create claimguard
   cloudflared tunnel route dns claimguard claimguard.yourdomain.com
   ```

4. **Configure it** (`~/.cloudflared/config.yml`):
   ```yaml
   tunnel: claimguard
   credentials-file: /home/<you>/.cloudflared/<tunnel-id>.json
   ingress:
     - hostname: claimguard.yourdomain.com
       service: http://localhost:8501
     - service: http_status:404
   ```

5. **Run it** (in one terminal, alongside `streamlit run dashboard/app.py` in another):
   ```bash
   cloudflared tunnel run claimguard
   ```

Your dashboard is now live at `https://claimguard.yourdomain.com` for as long as both
processes keep running. For something more persistent than a manually-run terminal,
`cloudflared service install` sets it up as a systemd service — but remember the
resource-constraint conversation from `docs/environment.md`: running Streamlit + the
tunnel + (occasionally) the ICD-11 container all at once on this machine is the
scenario worth watching RAM for.

## Option B: Hugging Face Spaces + Cloudflare Worker (always-on)

1. **Create the Space:** on huggingface.co, New Space -> SDK: Streamlit -> connect it to
   a GitHub repo (or push directly via `git`). Add `HF_TOKEN` (to pull the private model
   repo above) as a Space secret — never commit it.

2. **Space-specific entrypoint:** HF Spaces expects the app at the repo root; either
   symlink/copy `dashboard/app.py` to the repo root as `app.py`, or set the Space's "App
   file" setting to `dashboard/app.py` — either works, pick one and note it in the Space
   README.

3. **Confirm it's live** at the auto-assigned `https://<username>-claimguard.hf.space` URL
   first, before wiring the custom domain — easier to debug without the proxy in the way.

4. **Add the Cloudflare Worker reverse proxy** (this is the part that gets you the custom
   domain, since Spaces' free tier won't take it directly). In the Cloudflare dashboard:
   Workers & Pages -> Create Worker -> paste:
   ```javascript
   export default {
     async fetch(request) {
       const url = new URL(request.url);
       url.hostname = "<username>-claimguard.hf.space";
       return fetch(new Request(url, request));
     },
   };
   ```
   Then add a route: `claimguard.yourdomain.com/*` -> this Worker. Cloudflare's free
   Worker tier (100,000 requests/day) is far more than a thesis-project dashboard needs.

5. **Streamlit + WebSockets caveat:** Streamlit's live UI updates use WebSockets, which a
   naive `fetch()`-based Worker proxy (above) does NOT forward correctly. If the Space
   loads but interactions (buttons, live updates) don't work through the custom domain,
   test directly against the `*.hf.space` URL to confirm the app itself works, then look
   into Cloudflare's WebSocket-aware proxy patterns (`Workers` support upgrading
   connections, but the snippet above is the simple HTTP-only version — a fuller
   Worker script or a plain DNS-only "grey-clouded" CNAME may be needed depending on how
   Streamlit is configured). Flagging this now rather than presenting it as
   guaranteed-to-work; test before demoing rather than assuming.

## Notes for Chapter 5

- Record whichever option you actually use in `docs/environment.md`'s "Deployment /
  demonstration environment" section, with the real tested details (research_writing.txt
  5.2.3: "a concise, reproducible account of how the solution is installed, configured,
  started and verified").
- Never expose `HF_TOKEN`, Cloudflare API tokens, or `.env` contents in screenshots,
  commits, or the report itself (5.2.2/5.5) — if one leaks, rotate it, don't just delete
  the file.
