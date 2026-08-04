# Deploying Lavish_bot for a real paper-trading run

This is the runbook for actually running the bot unattended (a week, or
indefinitely) instead of testing it interactively. Read
`lavish_core/discord_ingest/listener.py`'s module docstring before setting
any Discord env vars - that's a real ToS/account-risk decision, not a
config toggle.

**Never paste real secrets (Alpaca keys, Patreon tokens, Discord tokens)
into a chat with an AI assistant, including this one. They go directly
into `.env` on the server, typed or pasted there yourself.**

## 1. Get a server

A $5-6/mo VPS (1 vCPU, 1GB RAM, Ubuntu 22.04 or 24.04) is plenty - this
bot is a lightweight polling loop plus DuckDB, not a heavy workload.
DigitalOcean, Hetzner, and Linode are all fine; pick whichever's easiest
for you to sign up for. You need: the server's IP address, and SSH access
(a root password or an SSH key, whichever the provider set you up with).

## 2. Initial server setup

SSH in as root, then create a dedicated non-root user to run the bot
under (matches `deploy/lavish-bot.service`'s `User=lavish` - the process
only needs to read its own directory and make outbound HTTPS calls,
nothing that needs root):

```bash
ssh root@YOUR_SERVER_IP

adduser lavish
usermod -aG sudo lavish
su - lavish
```

## 3. Install system dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git tesseract-ocr
```

`tesseract-ocr` is the actual OCR engine `pytesseract` wraps - the pip
package alone isn't enough, `system_boot.py`'s pre-flight check verifies
this binary is on PATH before ever starting the bot.

## 4. Get the code and install dependencies

If the repo is private (likely, for a repo holding your trading logic),
a bare `https://` clone will just prompt for credentials and fail non-
interactively. Easiest path: generate a GitHub personal access token
(Settings -> Developer settings -> Fine-grained tokens, read-only,
scoped to just this repo) and use it in the clone URL - or set up an SSH
deploy key instead, if you'd rather not have a token in shell history.

```bash
sudo mkdir -p /opt/lavish-bot
sudo chown lavish:lavish /opt/lavish-bot
git clone -b claude/lavish-bot-audit-4m7e1o https://YOUR_TOKEN@github.com/lavish213/lavish.git /opt/lavish-bot
cd /opt/lavish-bot

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-deploy.txt
```

Use `requirements-deploy.txt`, not `requirements.txt` - the latter also
installs a large amount of unrelated ML/LLM/scraping code (torch,
transformers, langchain, easyocr, paddleocr, etc.) that the live bot
never touches. `requirements-deploy.txt` is the traced, verified-minimal
set - a normal VPS install with it takes a couple minutes, not twenty.

## 5. Configure `.env`

```bash
cp env.example .env
nano .env   # or vim, whatever you've got
```

Fill in, at minimum:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` - from a **paper** Alpaca account
  (app.alpaca.markets -> Paper Trading -> API Keys). Do not use live keys
  for this week.
- `ALPACA_BASE_URL` - leave as the paper endpoint (already the default in
  `env.example`)
- `TRADE_MODE=paper` - explicit, not left blank (blank defaults to `dry`,
  which parses alerts and logs what it would do but submits nothing - a
  legitimate first day of testing if you want to watch it think before
  it trades, but not "running paper" yet)
- Patreon: either `PATREON_ACCESS_TOKEN`, or the three
  `PATREON_CLIENT_ID`/`PATREON_CLIENT_SECRET`/`PATREON_REFRESH_TOKEN`
  values, from an account with active paid access to her tier
- `WHITELIST_TICKERS` - **review this against what she's actually called
  recently.** This session found NFLX and AMZN missing from the shipped
  default while testing real alerts - anything not on this list is
  silently skipped. Expand it before going live, not after noticing a
  missed alert.
- `DISCORD_WEBHOOK_URL` - strongly recommended even for paper mode. This
  is how circuit-breaker trips and recovered crash positions actually
  reach you - without it, those only show up in log files nobody's
  watching. Any Discord webhook URL works (Server Settings ->
  Integrations -> Webhooks), doesn't need to be in her server.
- Leave `DISCORD_USER_TOKEN`/`DISCORD_INGEST_ACCEPT_TOS_RISK` blank unless
  you've already made the call on that - the bot runs Patreon-only
  without them, same as it has all session.

## 6. Verify before installing the service

```bash
source venv/bin/activate
python3 system_boot.py
```

This has to print "Pre-flight check passed" and exit 0 before moving on.
If it doesn't, fix what it flags - don't proceed to the systemd install
with a known-broken config.

## 7. Install and start the systemd service

```bash
sudo cp deploy/lavish-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lavish-bot
```

## 8. Confirm it's actually alive

```bash
systemctl status lavish-bot        # should say "active (running)"
journalctl -u lavish-bot -f        # live tail, Ctrl-C to stop watching
tail -f /opt/lavish-bot/logs/lavish.log      # circuit breaker, broker, reconcile
tail -f /opt/lavish-bot/logs/run_ingest.log  # startup config, source status
```

You should see the startup config line (confirms `TRADE_MODE`, ticker
count, whether Discord is active) and, within `PATREON_POLL_SECONDS`
(default 20s), the first poll cycle.

## 9. What to check each day of the week

You don't need to watch it continuously - that defeats the point of
running it unattended - but check in daily:

- `systemctl status lavish-bot` - still "active (running)"? If it shows
  "failed," it crash-looped past the 5-crashes-in-10-minutes limit and
  needs a look (`journalctl -u lavish-bot -n 100` for what killed it).
- `tail -50 /opt/lavish-bot/logs/lavish.log` - any circuit breaker trips,
  broker errors, or reconciliation recoveries?
- Discord webhook channel - anything posted there needs a look; that's
  the "something needs your attention" channel by design.
- `python3 -m lavish_core.reporting.track_record --days 7` - running
  win rate, P&L, and slippage on whatever's actually traded so far.

## 10. End of the week - deciding if it's right

- `python3 -m lavish_core.reporting.track_record --days 7` for the full
  week's numbers - win rate, total P&L, avg slippage, excess return vs
  SPY.
- Check `open_entries_no_recorded_exit` in that report - anything sitting
  there needs manual review (a position that closed some way other than
  the bot's own bracket/exit monitor).
- Skim `logs/lavish.log` for the week for anything that repeated (a
  recurring warning is worth fixing before trusting it further, even if
  it never blocked a trade).
- Only after that: decide on `TRADE_MODE=live` - and if you do, flip
  `POSITION_SIZE_SCALE` to something like `0.1`-`0.25` first rather than
  full size, per the staged-rollout guardrail already built for exactly
  this.

## Updating the code later

```bash
cd /opt/lavish-bot
git pull
source venv/bin/activate
pip install -r requirements-deploy.txt   # in case dependencies changed
sudo systemctl restart lavish-bot
```
