#!/usr/bin/env bash
# crash_guard.sh — preflight, heartbeat, loss audit, and crash forensics.
#
# Why this exists:
#   2026-10-23 09:30 UTC the WSL2 VM disappeared mid-cron.  The host C: drive
#   had ~0.2 GB free, so the ext4.vhdx (155 GiB) could not grow.  A Kaggle
#   output pull hit ENOSPC and left a 0-byte checkpoint.  Inside WSL `df`
#   still showed 803 GB free, which is a lie: the vhdx is sparse and bounded
#   by free space on C:.
#
# This guard is intentionally conservative: it never launches anything, it
# only blocks unsafe launches and preserves evidence for the next autopsy.
#
# Usage:
#   bash scripts/crash_guard.sh --preflight     # exit 0 safe, 2 blocked
#   bash scripts/crash_guard.sh --sample        # append heartbeat (cron)
#   bash scripts/crash_guard.sh --boot-check    # report reboot + store boot id
#   bash scripts/crash_guard.sh --report [why]  # forensic bundle under logs/
#   bash scripts/crash_guard.sh --watch <pid>   # report when a job dies
#   bash scripts/crash_guard.sh --install-cron  # idempotent @reboot + */5 cron
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

REPORT_DIR="$ROOT/logs/crash_reports"
HEARTBEAT="$REPORT_DIR/heartbeat.log"
BOOTID_FILE="$REPORT_DIR/last_boot_id"
GUARD_LOG="$ROOT/logs/crash_guard.log"
VHDX="/mnt/c/Users/HP/AppData/Local/wsl/{10b42e19-fa91-4111-b7e9-6b10ddc0aad9}/ext4.vhdx"
CKPT_DIRS=(
  "$ROOT/resources/models/checkpoints"
  "$HOME/kaggle_ch2_output/checkpoints"
)

# Hard gates. A 2.2 GB Kaggle output pull needs headroom, so C: is not
# allowed to be "almost full" even though training runs remotely.
MIN_C_FREE_MB="${MIN_C_FREE_MB:-8192}"
MIN_ROOT_FREE_MB="${MIN_ROOT_FREE_MB:-4096}"
MIN_TMP_FREE_MB="${MIN_TMP_FREE_MB:-512}"
MIN_SWAP_FREE_MB="${MIN_SWAP_FREE_MB:-300}"
MIN_MEM_AVAIL_MB="${MIN_MEM_AVAIL_MB:-800}"

mkdir -p "$REPORT_DIR"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

df_free_mb() {
  local kb
  kb="$(df -P "$1" 2>/dev/null | awk 'NR==2 {print $4}')"
  [ -n "${kb:-}" ] || { echo 0; return; }
  echo $((kb / 1024))
}

swap_free_mb() { free -m | awk '/^Swap:/ {print $4}'; }
mem_avail_mb() { free -m | awk '/^Mem:/ {print $7}'; }
mem_used_pct() {
  free -m | awk '/^Mem:/ {if ($2 > 0) printf "%d", ($3 * 100) / $2; else print 0}'
}
vhdx_gb() {
  local b
  b="$(stat -c %s "$VHDX" 2>/dev/null || echo 0)"
  echo $((b / 1024 / 1024 / 1024))
}
top_proc() {
  ps -eo rss=,comm= --sort=-rss 2>/dev/null | awk 'NR==1 {printf "%s:%dMB", $2, $1/1024}'
}
train_count() {
  pgrep -fc 'train_paired|train_unpaired|train_dapi' 2>/dev/null || true
}
zero_byte_files() {
  local d
  for d in "${CKPT_DIRS[@]}"; do
    [ -d "$d" ] || continue
    find "$d" -type f -size 0 2>/dev/null
  done
}

snapshot() {
  printf 'boot=%s uptime_s=%s mem_avail_mb=%s mem_used_pct=%s swap_free_mb=%s root_free_mb=%s c_free_mb=%s tmp_free_mb=%s vhdx_gb=%s train_procs=%s top=%s' \
    "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null | cut -c1-8)" \
    "$(cut -d. -f1 /proc/uptime)" \
    "$(mem_avail_mb)" "$(mem_used_pct)" "$(swap_free_mb)" \
    "$(df_free_mb /)" "$(df_free_mb /mnt/c)" "$(df_free_mb /tmp)" \
    "$(vhdx_gb)" "$(train_count)" "$(top_proc)"
}

rotate_heartbeat() {
  [ -f "$HEARTBEAT" ] || return 0
  local lines
  lines="$(wc -l < "$HEARTBEAT" 2>/dev/null || echo 0)"
  if [ "$lines" -gt 4000 ]; then
    tail -n 2000 "$HEARTBEAT" > "$HEARTBEAT.tmp" && mv "$HEARTBEAT.tmp" "$HEARTBEAT"
  fi
}

preflight() {
  local fails=0 warns=0 c_free root_free tmp_free swap_free mem_avail
  c_free="$(df_free_mb /mnt/c)"
  root_free="$(df_free_mb /)"
  tmp_free="$(df_free_mb /tmp)"
  swap_free="$(swap_free_mb)"
  mem_avail="$(mem_avail_mb)"

  check() { # name value min unit
    local name="$1" value="$2" min="$3" unit="$4"
    if [ "$value" -lt "$min" ]; then
      printf 'FAIL  %-18s %s%s < %s%s\n' "$name" "$value" "$unit" "$min" "$unit"
      fails=$((fails + 1))
    else
      printf 'OK    %-18s %s%s >= %s%s\n' "$name" "$value" "$unit" "$min" "$unit"
    fi
  }

  echo "=== crash_guard preflight $(ts) ==="
  check "C: free (real cap)" "$c_free" "$MIN_C_FREE_MB" "MB"
  check "WSL root (virtual)" "$root_free" "$MIN_ROOT_FREE_MB" "MB"
  check "SwapFree" "$swap_free" "$MIN_SWAP_FREE_MB" "MB"
  check "MemAvailable" "$mem_avail" "$MIN_MEM_AVAIL_MB" "MB"
  if [ "$tmp_free" -lt "$MIN_TMP_FREE_MB" ]; then
    printf 'WARN  %-18s %sMB < %sMB (never pull outputs to /tmp)\n' \
      "/tmp free" "$tmp_free" "$MIN_TMP_FREE_MB"
    warns=$((warns + 1))
  else
    printf 'OK    %-18s %sMB\n' "/tmp free" "$tmp_free"
  fi
  printf 'INFO  %-18s %sGB (C: free is the real ceiling)\n' "ext4.vhdx" "$(vhdx_gb)"

  local zeros
  zeros="$(zero_byte_files)"
  if [ -n "$zeros" ]; then
    printf 'WARN  %-18s corrupt/truncated artifacts exist\n' "checkpoint audit"
    printf '%s\n' "$zeros" | sed 's/^/        /'
    warns=$((warns + 1))
  else
    printf 'OK    %-18s no 0-byte checkpoints\n' "checkpoint audit"
  fi

  if [ "$fails" -gt 0 ]; then
    echo "RESULT: BLOCKED ($fails hard gate(s), $warns warning(s))"
    echo "Do not launch training or pull Kaggle output until this passes."
    return 2
  fi
  echo "RESULT: SAFE ($warns warning(s))"
  return 0
}

report() {
  local why="${1:-manual}"
  local stamp file
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  file="$REPORT_DIR/crash_${stamp}.md"

  {
    echo "# Crash / state report — $(ts)"
    echo
    echo "- reason: \`$why\`"
    echo "- prev boot id: \`${PREV_BOOT_ID:-$(cat "$BOOTID_FILE" 2>/dev/null || echo unknown)}\`"
    echo "- current boot id: \`$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)\`"
    echo "- uptime: $(uptime -p 2>/dev/null || uptime)"
    echo "- snapshot: \`$(snapshot)\`"
    echo
    echo "## Boot history"
    echo '```'
    journalctl --list-boots 2>/dev/null | tail -5 || true
    echo '```'
    echo
    echo "## Disk / memory"
    echo '```'
    df -h / /mnt/c /tmp 2>/dev/null || true
    echo
    free -m 2>/dev/null || true
    echo
    printf 'ext4.vhdx: %s GB\n' "$(vhdx_gb)"
    echo '```'
    echo
    echo "## Previous boot tail (last 30 lines)"
    echo '```'
    journalctl -b -1 --no-pager -n 30 2>/dev/null || echo "no previous boot journal"
    echo '```'
    echo
    echo "## Previous boot warnings/errors"
    echo '```'
    journalctl -b -1 -p warning --no-pager 2>/dev/null | tail -40 || true
    echo "---"
    journalctl -b -1 --no-pager 2>/dev/null | grep -iE 'oom|killed process|no space|ext4.*error|corrupt|hung|blocked' | tail -20 || true
    echo '```'
    echo
    echo "## Kernel messages (current boot)"
    echo '```'
    dmesg 2>/dev/null | tail -25 || echo "dmesg unavailable"
    echo '```'
    echo
    echo "## Training / monitor processes"
    echo '```'
    pgrep -af 'train_paired|train_unpaired|train_dapi|kaggle' 2>/dev/null || echo "none"
    echo '```'
    echo
    echo "## Newest checkpoints"
    echo '```'
    find "${CKPT_DIRS[@]}" -type f -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' 2>/dev/null \
      | sort -r | head -25 || true
    echo '```'
    echo
    echo "## Zero-byte / truncated artifacts (loss audit)"
    echo '```'
    zero_byte_files || true
    [ -n "$(zero_byte_files)" ] || echo "none"
    echo '```'
    echo
    echo "## Heartbeat (last 15)"
    echo '```'
    tail -n 15 "$HEARTBEAT" 2>/dev/null || echo "none"
    echo '```'
    echo
    echo "## Git working tree"
    echo '```'
    git status --short 2>/dev/null || true
    echo '```'
    echo
    echo "## Likely cause (heuristic)"
    local c_free swap_free
    c_free="$(df_free_mb /mnt/c)"
    swap_free="$(swap_free_mb)"
    if [ "$c_free" -lt 1024 ]; then
      echo "- Windows C: has only **${c_free} MB** free. The WSL \`ext4.vhdx\`"
      echo "  ($(vhdx_gb) GB) cannot grow, so new block writes fail with ENOSPC"
      echo "  even though \`df\` inside WSL shows virtual free space. This matches"
      echo "  a 0-byte checkpoint and a host-side WSL termination."
    fi
    if [ "$swap_free" -lt 100 ]; then
      echo "- SwapFree is only **${swap_free} MB**; local training or a large"
      echo "  output pull can exhaust the 3.7 GB WSL commit and be killed."
    fi
    [ "$c_free" -ge 1024 ] && [ "$swap_free" -ge 100 ] && \
      echo "- No disk/swap red flag; use the previous-boot tail to classify."
    echo
    echo "## Recovery"
    echo '```bash'
    echo "bash scripts/crash_guard.sh --preflight"
    echo "bash scripts/status.sh"
    echo '```'
  } > "$file"

  echo "report: $file"
  touch "$ROOT/logs/CRASH_ALERT"
}

boot_check() {
  local cur prev
  cur="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"
  prev="$(cat "$BOOTID_FILE" 2>/dev/null || true)"
  export PREV_BOOT_ID="${prev:-unknown}"
  printf '%s\n' "$cur" > "$BOOTID_FILE"
  echo "[$(ts)] boot-check current=$cur previous=${prev:-none}" >> "$GUARD_LOG"
  if [ -n "$prev" ] && [ "$prev" != "$cur" ]; then
    report "boot changed ${prev:0:8} -> ${cur:0:8}"
  fi
}

sample() {
  rotate_heartbeat
  printf '%s %s\n' "$(ts)" "$(snapshot)" >> "$HEARTBEAT"
}

watch_pid() {
  local pid="${1:?usage: --watch <pid>}"
  echo "[$(ts)] watching PID $pid" >> "$GUARD_LOG"
  while kill -0 "$pid" 2>/dev/null; do
    sample
    sleep 60
  done
  report "watched PID $pid exited"
}

install_cron() {
  local cur
  cur="$(crontab -l 2>/dev/null || true)"
  if printf '%s\n' "$cur" | grep -q 'crash_guard.sh --sample'; then
    echo "cron: crash guard already installed"
    return 0
  fi
  {
    printf '%s\n' "$cur"
    echo "# >>> he-dapi crash guard (managed by scripts/crash_guard.sh)"
    echo "@reboot /bin/bash $ROOT/scripts/crash_guard.sh --boot-check >> $GUARD_LOG 2>&1"
    echo "*/5 * * * * /bin/bash $ROOT/scripts/crash_guard.sh --sample >> $GUARD_LOG 2>&1"
    echo "# <<< he-dapi crash guard"
  } | awk 'NF && !seen[$0]++' | crontab -
  echo "cron: installed @reboot boot-check + */5 heartbeat"
}

case "${1:---preflight}" in
  --preflight) preflight ;;
  --sample) sample ;;
  --boot-check) boot_check ;;
  --report) shift; report "${1:-manual}" ;;
  --watch) shift; watch_pid "${1:-}" ;;
  --install-cron) install_cron ;;
  -h|--help)
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    echo "unknown mode: $1 (try --help)" >&2
    exit 64
    ;;
esac
