#!/usr/bin/env python3
"""Telegram bridge: control this machine's SurgicalVLA jobs and talk to Claude Code from your phone.

    nohup python3 scripts/telegram_bridge.py > results/plan/telegram_bridge.log 2>&1 &

Only messages from TELEGRAM_CHAT_ID (your own chat, set by `scripts/notify.sh --setup-telegram`)
are acted on; anything from any other chat is ignored and logged. Credentials come from
~/.config/surgicalvla-notify/env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).

Commands (send them to your bot):
    /status         cache progress per benchmark, GPU, last plan lines
    /gpu            nvidia-smi summary
    /jobs           running labelling / training / upload processes
    /log <name>     last lines of results/plan/<name>.log (e.g. /log plan, /log libero_spatial_labels_w0)
    /new            start a fresh Claude conversation
    /help           this list
    any other text  a prompt for Claude Code (headless `claude -p`, run in the project directory)

Claude runs with the permissions in your Claude Code settings: actions that would need an
approval prompt are refused rather than asked (no one is at the keyboard). To let it edit files,
set CLAUDE_BRIDGE_PERMISSION_MODE=acceptEdits in the env file; its replies come back here. The
conversation is kept across messages (one session, /new resets it). This is a separate Claude
session: it cannot type into a session already open in a terminal.

Standard library only, so it runs with the system python3.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONF = Path(os.environ.get("NOTIFY_CONF", Path.home() / ".config" / "surgicalvla-notify" / "env"))
STATE = ROOT / "results" / "plan" / "telegram_bridge_state.json"  # update offset + Claude session id
MAX_MSG = 4000


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


CFG = load_env(CONF)
TOKEN = CFG.get("TELEGRAM_BOT_TOKEN", "")
CHAT = CFG.get("TELEGRAM_CHAT_ID", "")
PERMISSION_MODE = CFG.get("CLAUDE_BRIDGE_PERMISSION_MODE", "default")
CLAUDE_TIMEOUT = int(CFG.get("CLAUDE_BRIDGE_TIMEOUT_S", "1800"))
API = f"https://api.telegram.org/bot{TOKEN}"


def api(method: str, timeout: float = 70, **params) -> dict:
    data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(f"{API}/{method}", data=data, timeout=timeout) as r:
        return json.loads(r.read())


def send(text: str) -> None:
    text = text or "(empty)"
    for i in range(0, len(text), MAX_MSG):
        try:
            api("sendMessage", timeout=30, chat_id=CHAT, text=text[i : i + MAX_MSG])
        except Exception as exc:  # network hiccup: log, keep serving
            print(f"send failed: {exc}", flush=True)


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"offset": 0, "session": None}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, STATE)


def sh(cmd: str, timeout: int = 60) -> str:
    try:
        out = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return (out.stdout + out.stderr).strip()
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"


def cmd_status() -> str:
    parts = [sh("bash scripts/prepare_data.sh status 2>&1 | grep -v '^$'", timeout=180), cmd_gpu(),
             "last plan lines:\n" + sh("tail -n 5 results/plan/plan.log 2>/dev/null")]
    return "\n\n".join(p for p in parts if p)


def cmd_gpu() -> str:
    return sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader")


def cmd_jobs() -> str:
    out = sh("ps -eo pid,etime,args | grep -E 'make_pseudolabels|make_cot_traces|scripts/train.py|train_reinflow|upload-large-folder|prepare_data|train_parallel' "
             "| grep -v grep | cut -c1-160")
    return out or "no SurgicalVLA jobs running"


def cmd_log(name: str) -> str:
    name = Path(name.strip() or "plan").name.removesuffix(".log")  # file name only: no paths outside results/plan
    path = ROOT / "results" / "plan" / f"{name}.log"
    if not path.exists():
        logs = sorted(p.stem for p in (ROOT / "results" / "plan").glob("*.log"))
        return f"no {path.name}; available: {', '.join(logs[-30:])}"
    return sh(f"tail -n 25 {path} | cut -c1-300")


def ask_claude(prompt: str, state: dict) -> None:
    """Run one headless Claude turn (in a thread) and send the reply."""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--permission-mode", PERMISSION_MODE]
    if state.get("session"):
        cmd += ["--resume", state["session"]]
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT)
        try:
            res = json.loads(out.stdout)
            state["session"] = res.get("session_id") or state.get("session")
            save_state(state)
            reply = res.get("result") or out.stderr.strip() or "(no reply)"
        except json.JSONDecodeError:
            reply = (out.stdout + out.stderr).strip()[-3000:] or f"claude exited {out.returncode}"
    except subprocess.TimeoutExpired:
        reply = f"⏱ Claude did not finish within {CLAUDE_TIMEOUT // 60} min"
    except FileNotFoundError:
        reply = "claude CLI not found on PATH"
    send(reply)


HELP = __doc__.split("Commands (send them to your bot):")[1].split("Claude runs with")[0].strip()


def handle(text: str, state: dict) -> None:
    text = text.strip()
    head, _, rest = text.partition(" ")
    head = head.split("@")[0].lower()  # /status@MyBot -> /status
    if head in ("/help", "/start"):
        send("SurgicalVLA bridge commands:\n" + HELP)
    elif head == "/status":
        send(cmd_status())
    elif head == "/gpu":
        send(cmd_gpu())
    elif head == "/jobs":
        send(cmd_jobs())
    elif head == "/log":
        send(cmd_log(rest))
    elif head == "/new":
        state["session"] = None
        save_state(state)
        send("🆕 new Claude conversation")
    elif text:
        send("⏳ asking Claude…")
        threading.Thread(target=ask_claude, args=(text, state), daemon=True).start()


def main() -> None:
    if not TOKEN or not CHAT:
        sys.exit(f"TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in {CONF} (run scripts/notify.sh --setup-telegram)")
    state = load_state()
    print(f"bridge up for chat {CHAT}; permission mode {PERMISSION_MODE}", flush=True)
    send("🤖 SurgicalVLA bridge online. /help for commands.")
    while True:
        try:
            resp = api("getUpdates", offset=state["offset"], timeout=50)
        except Exception as exc:
            print(f"poll failed: {exc}", flush=True)
            time.sleep(10)
            continue
        for upd in resp.get("result", []):
            state["offset"] = upd["update_id"] + 1
            save_state(state)
            msg = upd.get("message") or {}
            chat = str((msg.get("chat") or {}).get("id", ""))
            if chat != str(CHAT):
                print(f"ignored update from chat {chat or '?'}", flush=True)
                continue
            if msg.get("text"):
                print(f"cmd: {msg['text'][:80]}", flush=True)
                try:
                    handle(msg["text"], state)
                except Exception as exc:
                    send(f"error: {exc}")


if __name__ == "__main__":
    main()
