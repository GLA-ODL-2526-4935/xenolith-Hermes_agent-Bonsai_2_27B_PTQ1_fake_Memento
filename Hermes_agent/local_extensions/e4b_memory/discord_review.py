"""Opt-in standalone owner bot. No model, chat agent, HTTP admin API or shell dispatch."""
import argparse
import asyncio
import io
import logging
import os
from pathlib import Path

from .remote_review import RemoteReview, handle_message
from .store import Store, file_lock


def build_client(review):
    import discord

    intents = discord.Intents.none()
    intents.dm_messages = True
    client = discord.Client(intents=intents, allowed_mentions=discord.AllowedMentions.none(),
                            max_messages=None)
    lock = asyncio.Lock()

    @client.event
    async def on_message(message):
        # Never handle guilds, group DMs, bot/webhook messages or display-name authorization.
        direct = message.guild is None and isinstance(message.channel, discord.DMChannel)

        async def send(text, attachment):
            kwargs = {"allowed_mentions": discord.AllowedMentions.none(), "suppress_embeds": True}
            if attachment is not None:
                kwargs["file"] = discord.File(io.BytesIO(attachment), filename="memory-proposal.md")
            await message.channel.send(text, **kwargs)

        if review.authorized(message.author.id, direct=direct, bot=message.author.bot,
                             webhook=bool(message.webhook_id)):
            async with lock:
                try:
                    await handle_message(review, message, direct=direct, send=send)
                except (discord.HTTPException, OSError):
                    # No content/token logging; failed previews never acquire usable receipts.
                    review.receipts.clear()
                    logging.getLogger(__name__).warning("Discord review delivery failed; inspect owner CLI")

    return client


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path,
                        help="Existing owner memory bank directory, not service root")
    parser.add_argument("--owner-id", required=True, help="Discord numeric user ID, not name")
    parser.add_argument("--allow-discord-export", required=True, action="store_true",
                        help="Consent to send requested private proposals/evidence to Discord")
    args = parser.parse_args(argv)
    if not args.bank.is_absolute() or not (args.bank / "memory.json").is_file():
        parser.error("Use the absolute path to an existing owner bank with memory.json")
    # health: allow PS-P06 -- standalone owner-only process, no profile multiplexing; intentionally outside agent secret scope
    token = os.environ.get("DISCORD_MEMORY_REVIEW_TOKEN", "").strip()
    if not token:
        parser.error("Set the dedicated secret DISCORD_MEMORY_REVIEW_TOKEN outside agent configuration")
    review = RemoteReview(Store(args.bank), args.owner_id)
    # Only one bot owns receipt state for this bank. Do not hold the memory transaction lock.
    with file_lock(args.bank / "discord-review.lock"):
        build_client(review).run(token, log_level=logging.WARNING)


if __name__ == "__main__":
    main()
