# Discord manual memory review

Opt-in, **not configured or connected**. A separate owner-only bot lets you
inspect and accept/reject the automatic drafts and same-layer conflict proposals.
It is not a chatting agent and does not call a model. Normal Hermes Discord chat
integration can be configured separately later; do not reuse that bot/token here.

## Owner setup later

Use the existing Hermes environment with its optional `discord` dependency group
(already pinned in the project). No dependency installation is performed here.
Create a dedicated Discord application/bot, keep its token private, and enable
developer mode in Discord to copy **your numeric user ID**, not your username.
The bot uses direct messages only: no Administrator role, server access, member
intent or privileged Message Content intent is needed. An app/server relationship
that permits you to DM the bot may be needed; actual Discord setup is untested.
If joining a private server is needed to establish that relationship, grant no
guild permissions; the adapter ignores every guild/group message.

Run from `Hermes_agent` in the owner/service environment, **not inside the
restricted model process**. Supply a dedicated secret environment variable
`DISCORD_MEMORY_REVIEW_TOKEN` through your secret manager or private launcher.
Do not put it in agent-visible configuration, prompts, tracked files or shared
chat credentials. Then explicitly start:

```bash
python -m local_extensions.e4b_memory.discord_review \
  --bank /absolute/path/to/existing/memory-bank \
  --owner-id YOUR_NUMERIC_DISCORD_USER_ID --allow-discord-export
```

`--bank` is the directory containing `memory.json`. For the separate memory
service it is `<service-root>/memory`, not the service root. For a local profile,
it is that profile's `e4b_memory` directory. The bank must already exist; this
command does not initialize a service, enable generation or create a profile.
Only run this with the bank owner's privileges. It deliberately does not expose
owner acceptance over the model/controller HTTP API or public network listener.

## DM commands

1. `!memory pending` lists up to 20 pending IDs, without private source excerpts.
2. `!memory show ID` sends a **readable Markdown proposal and evidence report** as an
   attachment plus exact accept/reject commands containing a random code.
3. Read the full attachment, then copy the chosen command:
   `!memory accept ID CODE` or `!memory reject ID CODE`.

For generation proposals, append selected one-based draft numbers, e.g.
`!memory accept ID CODE 1,3`. This accepts those drafts and explicitly rejects
the rest, closing the proposal. Full audit data remains in the Markdown report;
JSON remains authoritative storage. See [REVIEW_WORKFLOW.md](REVIEW_WORKFLOW.md)
for owner history, archive management and the manual-acceptance workflow switch.

Only the latest successfully delivered preview is actionable. Its code expires
in five minutes, is single-use, and is lost on restart. There is no accept-all or
natural-language approval. Request another preview if the proposal changes.
Busy work or changed draft sources/persistent comparison prevents acceptance.
Inspect owner CLI state after ambiguous network/disk failure; do not blindly
repeat decisions. A successful Discord send cannot prove you read the attachment.

Draft acceptance adds temporary L0–L3 records; conflict acceptance can keep,
rewrite or discard temporary content. Persistent records are never rewritten by
this bot; the existing current-conflict and idle promotion gates still apply.
This grants no tool/system permissions, evidence append window, general memory
editing, model credentials or shell execution. It is separate from permission
approvals for browser/download/system access, which are not implemented here.

## Privacy and remaining validation

- Asking for a preview exports its private source material and compared
  persistent layer to Discord. Text DMs are **not end-to-end encrypted**
  ([Discord's explanation](https://discord.com/blog/every-voice-and-video-call-on-discord-is-now-end-to-end-encrypted)). The explicit
  export flag acknowledges this. There is no automatic proposal/evidence push.
  Do not use it for sensitive memories you do not want on Discord.
- Numeric sender ID, direct-message channel, non-bot/non-webhook checks and a
  bound one-use receipt guard decisions. Another Discord user, a guild message,
  tool output or webpage cannot approve. Compromise of your Discord account or
  the owner bot process remains a serious risk; account MFA and OS isolation
  are still needed. Same-account filesystem access is not isolated by Python.
- Mentions and embeds are disabled; proposal content is a Markdown attachment with quoted source/audit data,
  not executable content or commands. Unverified text can still mislead a human.
- Only one bot per bank; no multi-user mapping, auto-start, automatic
  notifications, acceptance buttons or automatic acceptance is added. Close the
  owner bot to disable remote acceptance; local owner CLI remains available.
- Offline scripted transport/store checks do not establish Discord delivery,
  onboarding permissions, account isolation or Linux service isolation. Live
  owner-DM setup and end-to-end model workflow testing remain unchecked.

See [automatic review](AUTOMATIC_REVIEW.md) for generation, conflict and storage
limits. Remote review does not remove the two manual review stages.

Transport references: [Discord DM content intent exception](https://github.com/discord/discord-api-docs/blob/main/developers/events/gateway.mdx#message-content-intent)
and [discord.py API](https://discordpy.readthedocs.io/en/latest/api.html).

Offline verification: all 16 local-extension files passed (383 cases), including
29 remote-review cases and 39 existing draft/conflict-review cases. The final
remote-review file was rerun after a test-only refactor: 29 passed. Targeted Ruff
passed. The existing Hermes health engine, scoped to the four changed Python
files at the nested project root, found no blocking/advisory issues; the owner
bot's dedicated environment secret has an explicit standalone-process exception.
This is not a claim that every CI check ran. The optional Discord package is not
installed in the disposable test interpreter, so its actual client construction,
Discord transport and login remain live-validation work. No package was installed,
token supplied, connection opened or model/GPU/engine launched.
