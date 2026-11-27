#!/usr/bin/env bash
# Phone alerts for long jobs: Telegram bot (preferred) or WhatsApp through CallMeBot.
#
#   scripts/notify.sh "libero cache done"            # send one message
#   scripts/notify.sh --run "libero SFT" cmd args...  # run a command, then report ✅/❌, exit code and duration
#   ... | scripts/notify.sh --hook                    # Claude Code Notification hook (reads the hook JSON on stdin)
#   scripts/notify.sh --setup-telegram                # find your chat id after you messaged the bot, save it
#
# Credentials stay out of the repo, in ~/.config/surgicalvla-notify/env (chmod 600):
#   TELEGRAM_BOT_TOKEN=<token from @BotFather>
#   TELEGRAM_CHAT_ID=<filled in by --setup-telegram>
#   # optional fallback: CALLMEBOT_PHONE=+<number>  CALLMEBOT_APIKEY=<key>
# Without credentials every call is a silent no-op, and a failed send never fails the caller,
# so scripts can call this unconditionally on any machine. Messages start with the host name.
CONF=${NOTIFY_CONF:-$HOME/.config/surgicalvla-notify/env}
[ -f "$CONF" ] || CONF=$HOME/.config/notify-whatsapp/env  # earlier location
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"

send() {
    local text="[$(hostname -s)] $1"
    text=${text:0:3500}
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -s --max-time 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            --data-urlencode "chat_id=$TELEGRAM_CHAT_ID" --data-urlencode "text=$text" > /dev/null 2>&1 || true
    elif [ -n "${CALLMEBOT_PHONE:-}" ] && [ -n "${CALLMEBOT_APIKEY:-}" ]; then
        curl -s -G --max-time 20 "https://api.callmebot.com/whatsapp.php" \
            --data-urlencode "phone=$CALLMEBOT_PHONE" --data-urlencode "text=$text" \
            --data-urlencode "apikey=$CALLMEBOT_APIKEY" > /dev/null 2>&1 || true
    fi
}

case "${1:-}" in
    --run)
        label=$2; shift 2
        start=$(date +%s)
        "$@"; rc=$?
        mins=$(( ($(date +%s) - start) / 60 ))
        if [ "$rc" -eq 0 ]; then send "✅ $label done (${mins} min)"; else send "❌ $label failed (exit $rc, ${mins} min)"; fi
        exit "$rc" ;;
    --hook)
        msg=$(jq -r '"Claude Code [\(.cwd // "" | split("/") | last)]: \(.message // "needs your input")"' 2>/dev/null)
        send "🔔 ${msg:-Claude Code needs your input}"
        exit 0 ;;
    --setup-telegram)
        [ -n "${TELEGRAM_BOT_TOKEN:-}" ] || { echo "put TELEGRAM_BOT_TOKEN=<token> in $CONF first"; exit 1; }
        resp=$(curl -s --max-time 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates")
        [ "$(jq -r .ok <<< "$resp")" = true ] || { echo "token rejected by Telegram: $(jq -r .description <<< "$resp")"; exit 1; }
        id=$(jq -r '[.result[] | (.message // .channel_post // .my_chat_member) | .chat.id] | last // empty' <<< "$resp")
        [ -n "$id" ] || { echo "no message found: open your bot in Telegram, press Start / send 'hi', then run this again"; exit 1; }
        sed -i '/^TELEGRAM_CHAT_ID=/d' "$CONF" && echo "TELEGRAM_CHAT_ID=$id" >> "$CONF" && chmod 600 "$CONF"
        TELEGRAM_CHAT_ID=$id send "✅ SurgicalVLA alerts connected"
        echo "chat id saved to $CONF; a test message was sent"
        exit 0 ;;
    *)
        send "$*" ;;
esac
exit 0
