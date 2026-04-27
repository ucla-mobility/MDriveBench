#!/bin/bash
# Sweep all v2xpnp scenes × {fcooper, attfuse, disco, cobevt} × {single, cooperative}.
#
# Saves one JSON-summary file per (scene, detector, mode) under $OUT_DIR.
# Skips combinations that already have a saved file → safe to interrupt and resume.
# Aggregates a final table across all completed runs.
#
# Usage (in tmux):
#   bash tools/sweep_v2xpnp_probe.sh
#
# Knobs (env vars):
#   PORT          CARLA port (default 1234)
#   GPU           CUDA_VISIBLE_DEVICES (default 5)
#   START_CARLA   if "1", launch own CARLA on $PORT + adapter $GPU and clean up on exit (default 0)
#   OUT_DIR       where to save per-run logs (default /tmp/v2xpnp_sweep)
#   DETECTORS     comma list (default fcooper,attfuse,disco,cobevt)
#   MODES         comma list (default single,cooperative)
#   SCENES        comma list of scene names; default = all in scenarioset/v2xpnp/
#   MAX_SCENES    cap on number of scenes (default 999)

set -u
PORT="${PORT:-1234}"
GPU="${GPU:-5}"
START_CARLA="${START_CARLA:-0}"
OUT_DIR="${OUT_DIR:-/tmp/v2xpnp_sweep}"
DETECTORS="${DETECTORS:-fcooper,attfuse,disco,cobevt}"
MODES="${MODES:-single,cooperative}"
MAX_SCENES="${MAX_SCENES:-999}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="/data/miniconda3/envs/colmdrivermarco2/bin/python"
CARLA_SH="$REPO_ROOT/carla912/CarlaUE4.sh"
SCENARIOSET_DIR="$REPO_ROOT/scenarioset/v2xpnp"

mkdir -p "$OUT_DIR/logs"

CARLA_PID=""

start_carla() {
    echo "[sweep] launching CARLA on port $PORT (graphicsadapter=$GPU)..."
    "$CARLA_SH" --world-port="$PORT" -graphicsadapter="$GPU" -RenderOffScreen \
        > "$OUT_DIR/carla.log" 2>&1 &
    CARLA_PID=$!
    # Wait up to 60 s for the port to open.
    for _ in $(seq 1 60); do
        if nc -z -w1 127.0.0.1 "$PORT" 2>/dev/null; then
            echo "[sweep] CARLA up (pid=$CARLA_PID)"
            sleep 3   # buffer for full server init
            return 0
        fi
        sleep 1
    done
    echo "[sweep] ERROR: CARLA did not open port $PORT within 60s; tail of carla.log:"
    tail -20 "$OUT_DIR/carla.log"
    return 1
}

stop_carla() {
    if [ -n "$CARLA_PID" ]; then
        echo "[sweep] stopping CARLA pid=$CARLA_PID..."
        kill "$CARLA_PID" 2>/dev/null
        # Also kill the actual CarlaUE4 binary in case the wrapper double-forked.
        pkill -f "CarlaUE4.*world-port=$PORT" 2>/dev/null
    fi
}

is_carla_alive() {
    nc -z -w1 127.0.0.1 "$PORT" 2>/dev/null
}

if [ "$START_CARLA" = "1" ]; then
    trap stop_carla EXIT INT TERM
    start_carla || exit 1
else
    if ! is_carla_alive; then
        echo "[sweep] ERROR: no CARLA listening on port $PORT (and START_CARLA=0)."
        echo "  Either start one yourself, or rerun with START_CARLA=1."
        exit 1
    fi
fi

if [ -n "${SCENES:-}" ]; then
    IFS=',' read -ra SCENE_LIST <<< "$SCENES"
else
    mapfile -t SCENE_LIST < <(ls -1 "$SCENARIOSET_DIR" | head -n "$MAX_SCENES")
fi
IFS=',' read -ra DETECTOR_LIST <<< "$DETECTORS"
IFS=',' read -ra MODE_LIST <<< "$MODES"

TOTAL=$(( ${#SCENE_LIST[@]} * ${#DETECTOR_LIST[@]} * ${#MODE_LIST[@]} ))
N=0
SKIPPED=0
FAILED=0
START_T=$(date +%s)

echo "[sweep] start  port=$PORT  GPU=$GPU  out=$OUT_DIR"
echo "[sweep] $TOTAL combinations: ${#SCENE_LIST[@]} scenes × ${#DETECTOR_LIST[@]} detectors × ${#MODE_LIST[@]} modes"

for scene in "${SCENE_LIST[@]}"; do
    for det in "${DETECTOR_LIST[@]}"; do
        for mode in "${MODE_LIST[@]}"; do
            N=$((N + 1))
            tag="${scene}__${det}__${mode}"
            log_path="$OUT_DIR/logs/${tag}.log"
            done_marker="$OUT_DIR/logs/${tag}.done"
            if [ -f "$done_marker" ]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            FLAG=""
            [ "$mode" = "cooperative" ] && FLAG="--cooperative"
            elapsed=$(( $(date +%s) - START_T ))
            echo "[sweep] ${N}/${TOTAL} (skip=$SKIPPED fail=$FAILED, ${elapsed}s) — $tag"
            CUDA_VISIBLE_DEVICES="$GPU" "$PYBIN" \
                "$REPO_ROOT/tools/detection_probe_v2xpnp.py" \
                --scenario-dir "$SCENARIOSET_DIR/$scene" \
                --detector "$det" \
                --port "$PORT" \
                --score-thresh 0.05 \
                $FLAG \
                > "$log_path" 2>&1
            rc=$?
            if [ $rc -eq 0 ] && grep -q "BEV IoU metrics" "$log_path"; then
                touch "$done_marker"
            else
                FAILED=$((FAILED + 1))
                echo "  -> FAILED rc=$rc, see $log_path"
                # If CARLA died, try to restart (only meaningful when START_CARLA=1).
                if ! is_carla_alive; then
                    echo "  -> CARLA appears dead; "
                    if [ "$START_CARLA" = "1" ]; then
                        echo "     restarting..."
                        stop_carla
                        sleep 3
                        start_carla || { echo "  -> restart failed, aborting"; exit 1; }
                    else
                        echo "     waiting up to 60s for it to come back..."
                        for _ in $(seq 1 60); do
                            sleep 1
                            is_carla_alive && break
                        done
                        is_carla_alive || { echo "  -> still dead, aborting"; exit 1; }
                    fi
                else
                    sleep 4
                fi
            fi
        done
    done
done

echo
echo "[sweep] done. total=$N completed=$((N-FAILED-SKIPPED)) skipped=$SKIPPED failed=$FAILED  out=$OUT_DIR"
echo "[sweep] aggregating..."
"$PYBIN" "$REPO_ROOT/tools/aggregate_v2xpnp_sweep.py" --logs "$OUT_DIR/logs"
