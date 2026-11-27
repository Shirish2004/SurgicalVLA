#!/usr/bin/env bash
# WhatsApp alerts through CallMeBot (https://www.callmebot.com/blog/free-api-whatsapp-messages/).
#
#   scripts/notify.sh "libero cache done"            # send one message
#   scripts/notify.sh --run "libero SFT" cmd args...  # run a command, then report ✅/❌, exit code and duration
#   ... | scripts/notify.sh --hook                    # Claude Code Notification hook (reads the hook JSON on stdin)
#
# Credentials stay out of the repo, in ~/.config/notify-whatsapp/env (chmod 600):
#   CALLMEBOT_PHONE=+<country code><number>
#   CALLMEBOT_APIKEY=<key CallMeBot sent you>
# Without them every call is a silent no-op, and a failed send never fails the caller, so
# scripts can call this unconditionally on any machine. Messages start with the host name.
CONF=${NOTIFY_WHATSAPP_CONF:-$HOME/.config/notify-whatsapp/env}
# shellcheck disable=SC1090
[ -f "$CONF" ] && . "$CONF"

send() {
    [ -n "${CALLMEBOT_PHONE:-}" ] && [ -n "${CALLMEBOT_APIKEY:-}" ] || return 0
    local text="[$(hostname -s)] $1"
    curl -s -G --max-time 20 "https://api.callmebot.com/whatsapp.php" \
        --data-urlencode "phone=$CALLMEBOT_PHONE" --data-urlencode "text=${text:0:900}" \
        --data-urlencode "apikey=$CALLMEBOT_APIKEY" > /dev/null 2>&1 || true
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
    *)
        send "$*" ;;
esac
exit 0
