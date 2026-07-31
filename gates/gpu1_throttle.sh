#!/usr/bin/env bash
# GPU1 duty-cycle watchdog (throwaway; tb_ prefix).
#
# GPU1 drives the display and can hang under sustained 100% utilization, so the lane running on it
# must leave the compositor gaps. This samples GPU1 every 5s and, after NEED consecutive samples at
# or above THRESH% (default 9 x 5s = 45s), SIGSTOPs THE CAMPAIGN'S OWN compute processes for HOLD
# seconds then SIGCONTs them — a 45/48 = ~94% duty cycle. Every event is appended to resources.log
# so the throttle's activity is measurable rather than assumed.
#
# SCOPE — the watchdog signals ONLY the campaign's own process tree. A pid holding a GPU1 compute
# context is eligible iff it, or any ancestor, is in the campaign's process group (arg 2, default:
# this watchdog's own PGID, which is the launcher's when the launcher starts it). Anything else on
# GPU1 is someone else's job — most likely the maintainer's — and is LOGGED ONCE AND NEVER
# SIGNALLED. Targeting by device alone was over-broad: it would have duty-cycled any CUDA process
# that happened to be on GPU1, which is not this campaign's to touch.
#
# Deliberately NOT nvidia-smi power/clock limits: those need root, change system state, and cap
# the clock rather than creating the idle gaps the compositor actually needs.
set -u
cd /home/vretiel/PycharmProjects/riskflow || exit 1
WF=${1:-artifacts/walk_forward/streaming_ladder}
LOG="$WF/resources.log"
# The campaign's process group. Default = our own, which is the launcher's group whenever the
# launcher spawns us; pass it explicitly when the watchdog is attached to an already-running
# campaign from a separate session.
CAMPAIGN_PGID=${2:-$(cut -d' ' -f3 <<<"$(sed 's/.*) //' /proc/$$/stat)")}
GPU=${GPU:-1}
THRESH=${THRESH:-98}          # % utilization that counts as "saturated"
NEED=${NEED:-9}               # consecutive 5s samples before intervening (9 = 45s)
HOLD=${HOLD:-3}               # seconds of SIGSTOP per intervention
HEARTBEAT=${HEARTBEAT:-12}    # log a sample summary every N samples (12 = 60s)

gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null; }
say()      { printf '%s WATCHDOG %s\n' "$(date -Is)" "$*" >> "$LOG"; }

# /proc/<pid>/stat, fields from `state` onward (comm can contain spaces and parens, so strip
# through the LAST ')'). Then $2 = ppid, $3 = pgrp.
proc_rest() { sed 's/.*) //' "/proc/$1/stat" 2>/dev/null; }

in_campaign() {                       # 0 = pid is inside the campaign's process tree
    local p=$1 hops=0 rest ppid pgrp
    while [ -n "$p" ] && [ "$p" -gt 1 ] && [ "$hops" -lt 32 ]; do
        rest=$(proc_rest "$p") || return 1
        [ -z "$rest" ] && return 1
        ppid=$(cut -d' ' -f2 <<<"$rest")
        pgrp=$(cut -d' ' -f3 <<<"$rest")
        [ "$pgrp" = "$CAMPAIGN_PGID" ] && return 0
        p=$ppid; hops=$((hops + 1))
    done
    return 1
}

eligible_pids() {                     # GPU1 compute pids that belong to this campaign
    local pid
    for pid in $(gpu_pids); do
        in_campaign "$pid" && printf '%s ' "$pid"
    done
}

FOREIGN_SEEN=" "
note_foreign() {                      # log a non-campaign GPU1 pid once, and never signal it
    local pid
    for pid in $(gpu_pids); do
        if ! in_campaign "$pid"; then
            case "$FOREIGN_SEEN" in
                *" $pid "*) ;;
                *) FOREIGN_SEEN="$FOREIGN_SEEN$pid "
                   say "FOREIGN pid=$pid on GPU$GPU (cmd: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-60)) — outside campaign PGID $CAMPAIGN_PGID, will NEVER be signalled" ;;
            esac
        fi
    done
}

say "armed on GPU$GPU: stop ${HOLD}s after ${NEED}x5s >= ${THRESH}% (~$((100 * NEED * 5 / (NEED * 5 + HOLD)))% duty), scope = campaign PGID $CAMPAIGN_PGID [members: $(pgrep -g "$CAMPAIGN_PGID" | tr '\n' ' ')]"
note_foreign

# Self-test: prove the STOP/CONT mechanism on the live lane once, so the throttle is known to work
# even if the utilization profile never crosses the trigger.
selftest=$(eligible_pids)
if [ -n "${selftest// /}" ]; then
    for pid in $selftest; do
        kill -STOP "$pid" 2>/dev/null && sleep 1 && kill -CONT "$pid" 2>/dev/null && \
            say "self-test OK: pid=$pid ELIGIBLE (ancestry reaches PGID $CAMPAIGN_PGID), SIGSTOP+SIGCONT 1s, lane resumed"
    done
else
    say "self-test SKIPPED: no CAMPAIGN compute process on GPU$GPU yet"
fi

hits=0; n=0; peak=0; sum=0
while true; do
    util=$(gpu_util)
    if [ -n "${util:-}" ]; then
        n=$((n + 1)); sum=$((sum + util)); [ "$util" -gt "$peak" ] && peak=$util
        if [ "$util" -ge "$THRESH" ]; then hits=$((hits + 1)); else hits=0; fi
    fi
    note_foreign
    if [ "$hits" -ge "$NEED" ]; then
        mine=$(eligible_pids)
        if [ -n "${mine// /}" ]; then
            for pid in $mine; do kill -STOP "$pid" 2>/dev/null; done
            say "THROTTLE util=${util}% sustained ${NEED}x5s -> SIGSTOP ${HOLD}s pids=[${mine% }]"
            sleep "$HOLD"
            for pid in $mine; do kill -CONT "$pid" 2>/dev/null; done
            say "THROTTLE -> SIGCONT pids=[${mine% }]"
        else
            say "SATURATED util=${util}% but no campaign process on GPU$GPU — standing down (not ours to throttle)"
        fi
        hits=0
    fi
    if [ "$n" -ge "$HEARTBEAT" ]; then
        say "gpu$GPU samples=$n mean=$((sum / n))% peak=${peak}% run=${hits}/${NEED} campaign_pids=[$(eligible_pids)]"
        n=0; peak=0; sum=0
    fi
    sleep 5
done
