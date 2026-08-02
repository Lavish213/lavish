# lavish_core/discord_ingest/listener.py
#
# Reads alert messages in near-real-time from a Discord server's channels
# (e.g. #stock-trades / #premium / #diamond) and feeds them into the same
# execution pipeline as the Patreon path (lavish_core.trading.trade_handler).
#
# =====================================================================
# READ THIS BEFORE SETTING DISCORD_USER_TOKEN
# =====================================================================
# There is no "bot" path for reading someone else's paid Discord server:
# a bot application has to be invited by the server owner, and that isn't
# going to happen for a server whose value proposition is paid alerts.
# The only way to read messages here is a *user account token* - logging
# in as yourself (or a throwaway account with its own paid membership) and
# having a script watch messages as if you were reading them yourself.
#
# Automating a Discord user account like this is against Discord's Terms
# of Service and can get that account warned or disabled - this isn't
# hypothetical, it's the same tradeoff every open-source project that does
# this (e.g. AdoNunes/DiscordAlertsTrader) documents explicitly. Nothing
# in this file runs unless you've made that call yourself:
#
#   DISCORD_USER_TOKEN=<your account's token>
#   DISCORD_INGEST_ACCEPT_TOS_RISK=1        (explicit opt-in, no default)
#   DISCORD_WATCH_CHANNEL_IDS=123,456,789   (channel IDs to watch)
#
# Requires the `discord.py-self` package (NOT `discord.py` - they occupy
# the same import name and cannot both be installed; discord.py-self is
# the community fork that supports logging in as a user account).
#   pip uninstall discord.py && pip install discord.py-self
#
# If you'd rather not take on that risk, don't set these - the bot will
# keep working off Patreon polling only, and you can manually paste alert
# text into a queue instead (a much smaller lift than the listener below,
# ask for it if you want that path instead).
# =====================================================================

from __future__ import annotations
import os, logging
from typing import Optional
from functools import partial

from lavish_core.logger_setup import get_logger
from lavish_core.trading.alert_handler import handle_alert_text as _handle_alert_text

log = get_logger("discord_ingest", log_dir="logs")

DISCORD_USER_TOKEN = os.getenv("DISCORD_USER_TOKEN")
ACCEPT_TOS_RISK = os.getenv("DISCORD_INGEST_ACCEPT_TOS_RISK", "0") == "1"
WATCH_CHANNEL_IDS = {
    int(c) for c in os.getenv("DISCORD_WATCH_CHANNEL_IDS", "").split(",") if c.strip().isdigit()
}
SIGNAL_SOURCE_NAME = os.getenv("DISCORD_SIGNAL_SOURCE_NAME", "discord")

# Kept as a module-level alias so existing callers/tests using
# listener.handle_alert_text(...) keep working; parsing/execution logic
# itself now lives in lavish_core.trading.alert_handler, shared with Patreon.
handle_alert_text = partial(_handle_alert_text, source=SIGNAL_SOURCE_NAME)


def run() -> None:
    if not DISCORD_USER_TOKEN or not ACCEPT_TOS_RISK:
        log.warning(
            "Discord listener not starting: requires both DISCORD_USER_TOKEN and "
            "DISCORD_INGEST_ACCEPT_TOS_RISK=1 to be set. See the module docstring "
            "in lavish_core/discord_ingest/listener.py before enabling this."
        )
        return
    if not WATCH_CHANNEL_IDS:
        log.warning("DISCORD_WATCH_CHANNEL_IDS is empty - nothing to listen to. Not starting.")
        return

    try:
        import discord
    except ImportError:
        log.error(
            "discord.py-self is not installed. This must NOT be installed alongside "
            "discord.py (same import name, they conflict). Run: "
            "pip uninstall discord.py && pip install discord.py-self"
        )
        return

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents, self_bot=True)

    @client.event
    async def on_ready():
        log.info("Discord listener connected as %s, watching channels: %s",
                  client.user, sorted(WATCH_CHANNEL_IDS))

    async def _process(message) -> None:
        if message.channel.id not in WATCH_CHANNEL_IDS:
            return
        if message.author.id == client.user.id:
            return

        image_path = None
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                import tempfile
                fd, image_path = tempfile.mkstemp(suffix=os.path.splitext(att.filename)[-1])
                os.close(fd)
                await att.save(image_path)
                break

        try:
            handle_alert_text(
                text=message.content,
                image_path=image_path,
                note=f"discord:{message.channel.id}:{message.id}",
            )
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    @client.event
    async def on_message(message):
        await _process(message)

    @client.event
    async def on_message_edit(before, after):
        # Alert callers sometimes correct themselves right after posting
        # ("Target $1,120 I meant.") - handle edits, not just new messages.
        await _process(after)

    log.info("Starting Discord listener (channels=%s)...", sorted(WATCH_CHANNEL_IDS))
    client.run(DISCORD_USER_TOKEN)


if __name__ == "__main__":
    run()
