# Hosting the study server publicly

`app/study_app.py` needs to be reachable from the public internet (raters are
off-site), but where it actually runs was undecided when this was written —
possibly a local machine now, possibly an IT-provisioned VM later. Either
way, the app itself doesn't change; only how you expose it does.

## Run the app

```bash
conda activate dementia-speech
shiny run app/study_app.py --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` makes it listen on all network interfaces, not just
localhost — necessary for anything in front of it (tunnel or reverse proxy)
to reach it.

## Option A — Cloudflare Tunnel (no IT ticket needed)

Good fit if you're running this from your own machine and don't want to deal
with router port-forwarding or a firewall exception. Free, gives a stable
subdomain, no inbound ports need opening.

1. Install `cloudflared` and log in with a (free) Cloudflare account:
   `cloudflared tunnel login`.
2. Create a named tunnel: `cloudflared tunnel create study-app`.
3. Point a DNS record at it: `cloudflared tunnel route dns study-app study.yourdomain.example`.
4. Run it alongside the Shiny process:
   `cloudflared tunnel run --url http://localhost:8000 study-app`.
5. `STUDY_BASE_URL` in `study/consent_form/onSubmit.gs` becomes
   `https://study.yourdomain.example/`.

Keep both the `shiny run` process and the `cloudflared tunnel run` process
alive for the study's duration (e.g. under `systemd`, `tmux`, or `screen`).

## Option B — Sheffield IT-Services virtual server

Sheffield IT Services offers self-requestable Linux VMs for exactly this kind
of thing (research/departmental application hosting). Confirmed from
`staff.sheffield.ac.uk/it-services/storage/virtual-servers` (2026-08-11):

- **Request via the Virtual Server Request Form** — built within 5 working
  days. You'll need to read and agree to the Server Policy and Privileged
  Account Management Policy first.
- **SSH access**, via a special `sa_username` admin account created for the
  VM (not your normal university login).
- **IT Services doesn't support app-level config** — installing/maintaining
  conda, nginx, this repo etc. is entirely on you, which matches the setup
  below.
- **Puppet manages baseline config every 2 hours** and can revert changes it
  considers drift (including password changes) — if nginx/systemd config
  mysteriously reverts, that's likely why; contact the IT Service Desk rather
  than fighting it.
- **Automatic reboots**: Linux VMs patch daily and reboot on a random day
  2–5am during the week containing the second Tuesday of the month. This is
  exactly why the `systemd` unit below needs `Restart=on-failure` and to be
  `enable`d (not just started) — the app must come back up on its own after
  an unattended reboot mid-study.
- **Backups**: IT takes a daily disaster-recovery backup, kept 28 days — a
  useful safety net, but don't rely on it alone if you'd care about losing
  more than 4 weeks of `data/study/sessions/`; keep your own copy too (see
  "Either way" below). On-demand snapshot backups are also available via a
  Service Desk request before risky changes.
- **What the page doesn't cover**: public internet reachability (vs.
  university-network-only) and firewall/port-opening. Ask about this
  explicitly when requesting the VM — don't assume 80/443 are open to the
  public internet by default.
- **Decommissioning** at the end of the study is a separate explicit step
  (power off, submit the decommissioning form) — the VM is *not* deleted
  automatically, but also isn't guaranteed recoverable once you do request
  deletion, so copy out `data/study/sessions/` and your aggregated CSVs
  first.

Once you have SSH access to the VM:

1. Clone the repo (or just `app/` + `data/` + the conda env) onto the VM's
   **local disk** — do not run the study server against a network-mounted
   working copy; SQLite/file-locking behaviour on network filesystems is
   unreliable (see `docs/study_protocol.md`'s persistence note — the current
   per-token JSON-file store was chosen partly to tolerate this, but local
   disk is still the safer bet if you have the choice).
2. Keep the Shiny process running with `systemd` so it survives reboots and
   restarts on crash. `/etc/systemd/system/dementia-speech-study.service`:

   ```ini
   [Unit]
   Description=Dementia speech study Shiny app
   After=network.target

   [Service]
   Type=simple
   User=<your-vm-user>
   WorkingDirectory=/path/to/dementia-speech
   ExecStart=/path/to/miniconda3/envs/dementia-speech/bin/shiny run app/study_app.py --host 127.0.0.1 --port 8000
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

   Then: `sudo systemctl daemon-reload && sudo systemctl enable --now dementia-speech-study`.
   Note `--host 127.0.0.1` here, not `0.0.0.0` — nginx is the only thing that
   should reach it directly; the outside world goes through nginx.

3. Put a reverse proxy in front of it, terminating TLS and forwarding to
   `http://127.0.0.1:8000`. Shiny's websocket connection needs the
   `Upgrade`/`Connection` headers passed through explicitly — nginx doesn't
   do this by default for proxied requests. `/etc/nginx/sites-available/study`:

   ```nginx
   server {
       listen 443 ssl;
       server_name study.yourdomain.example;

       # TLS cert paths — e.g. from certbot / IT's cert
       ssl_certificate     /etc/letsencrypt/live/study.yourdomain.example/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/study.yourdomain.example/privkey.pem;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_read_timeout 86400; # keep long-lived websocket connections open
       }
   }

   server {
       listen 80;
       server_name study.yourdomain.example;
       return 301 https://$host$request_uri;
   }
   ```

   Then: `sudo ln -s /etc/nginx/sites-available/study /etc/nginx/sites-enabled/`
   and `sudo nginx -t && sudo systemctl reload nginx`. Ask IT for the TLS
   cert (or `certbot --nginx -d study.yourdomain.example` if they'll let you
   run Let's Encrypt directly on the box).

4. `STUDY_BASE_URL` becomes whatever public hostname IT assigns, e.g.
   `https://study.yourdomain.example/`.

## Either way

- `data/study/sessions/` accumulates one small JSON file per rater — back it
  up periodically once the study is live.
- Restarting the `shiny run` process is safe mid-study: all progress is
  read from `data/study/sessions/` on each request, nothing is held only in
  memory.
- To see results, run `python pipelines/aggregate_study_results.py` — reads
  every file in `data/study/sessions/` and writes
  `data/processed/study_decisions.csv` (one row per rater × case, joined
  against the true diagnosis) and `data/processed/study_summary.csv`
  (per-rater accuracy, mean confidence, completion status). Rerun it any
  time; it's a read-only aggregation, safe to run while the study server is
  still live.
