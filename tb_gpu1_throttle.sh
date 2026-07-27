#!/usr/bin/env bash
# GPU1 duty-cycle watchdog (throwaway; tb_ prefix).
#
# GPU1 drives the display and can hang under sustained 100% utilization, so the lane running on it
# must leave the compositor gaps. This samples GPU1 every 5s and, after NEED consecutive samples at
# or above THRESH% (default 9 x 5s = 45s), SIGSTOPs the compute process for HOLD seconds then
# SIGCONTs it — a 45/48 = ~94% duty cycle. Every event is appended to resources.log so the
# throttle's activity is measurable rather than assumed.
#
# It targets by DEVICE, not by pid: `--query-compute-apps` on GPU1 finds whichever lane currently
# holds a context there, so the watchdog follows the campaign from arm A to arm B without being
# told. It exits when told to (kill it), not when a lane ends.
#
# Deliberately NOT nvidia-smi power/clock limits: those need root, change system state, and cap
# the clock rather than creating the idle gaps the compositor actually needs.
set -u
cd /home/vretiel/PycharmProjects/riskflow || exit 1
WF=${1:-artifacts/walk_forward/streaming_ladder}
LOG="$WF/resources.log"
GPU=${GPU:-1}
THRESH=${THRESH:-98}          # % utilization that counts as "saturated"
NEED=${NEED:-9}               # consecutive 5s samples before intervening (9 = 45s)
HOLD=${HOLD:-3}               # seconds of SIGSTOP per intervention
HEARTBEAT=${HEARTBEAT:-12}    # log a sample summary every N samples (12 = 60s)

gpu_util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null; }
gpu_pid()  { nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | head -1; }
say()      { printf '%s WATCHDOG %s\n' "$(date -Is)" "$*" >> "$LOG"; }

say "armed on GPU$GPU: stop ${HOLD}s after ${NEED}x5s >= ${THRESH}% (~$((100 * NEED * 5 / (NEED * 5 + HOLD)))% duty)"

# Self-test: prove the STOP/CONT mechanism on the live lane once, so the throttle is known to work
# even if the utilization profile never crosses the trigger.
pid=$(gpu_pid)
if [ -n "${pid:-}" ] && kill -STOP "$pid" 2>/dev/null; then
    sleep 1
    kill -CONT "$pid" 2>/dev/null
    say "self-test OK: SIGSTOP+SIGCONT pid=$pid (1s), lane resumed"
else
    say "self-test SKIPPED: no compute process on GPU$GPU yet"
fi

hits=0; n=0; peak=0; sum=0
while true; do
    util=$(gpu_util); pid=$(gpu_pid)
    if [ -n "${util:-}" ]; then
        n=$((n + 1)); sum=$((sum + util)); [ "$util" -gt "$peak" ] && peak=$util
        if [ "$util" -ge "$THRESH" ] && [ -n "${pid:-}" ]; then hits=$((hits + 1)); else hits=0; fi
    fi
    if [ "$hits" -ge "$NEED" ] && [ -n "${pid:-}" ]; then
        if kill -STOP "$pid" 2>/dev/null; then
            say "THROTTLE pid=$pid util=${util}% sustained ${NEED}x5s -> SIGSTOP ${HOLD}s"
            sleep "$HOLD"
            kill -CONT "$pid" 2>/dev/null
            say "THROTTLE pid=$pid -> SIGCONT"
        fi
        hits=0
    fi
    if [ "$n" -ge "$HEARTBEAT" ]; then
        say "gpu$GPU samples=$n mean=$((sum / n))% peak=${peak}% run=${hits}/${NEED} pid=${pid:-none}"
        n=0; peak=0; sum=0
    fi
    sleep 5
done
