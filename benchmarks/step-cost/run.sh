#!/usr/bin/env bash
# Measure the fixed costs of a pipeline step over synthetic entries.
#
# Usage: benchmarks/step-cost/run.sh WORK OUT [SECTION...]
#
# WORK holds the datasets (several GB at the default size) and OUT the results.
# Sections run in the order given; the default runs all but profile:
#   build        write ENTRIES synthetic track files and convert them (base)
#   cold         speed-angvel cold at four worker settings, the build and
#                apply seconds of the process run, then cached in thread and
#                in process mode
#   pipes        speed-angvel cold in process mode, counting the bytes this
#                process sends to its workers and receives from them
#   request      the six-step recipe cold and cached, one process per step,
#                and the start-up and import times
#   settle       one `pipeline settle` process per cached step (after request)
#   wide         a 61-step recipe over two entries, cold, cached, per step
#   micro        per-entry operations timed alone
#   profile      cProfile and stack samples of the runs above (after them)
#   fingerprint  hashes of every dataset's outputs and index rows
# Every timed run appends to OUT/results.tsv and every counted run to
# OUT/pipes.tsv. summary.py OUT prints the tables.
#
# Environment: PY (default: the repository's .venv/bin/python), ENTRIES
# (default 160), WORKERS (default 8). Compare fingerprints only between runs
# over the same WORK path: the tracks index records each source's path.
set -u

if [ $# -lt 2 ]; then
    sed -n '2,26p' "$0"
    exit 2
fi
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
mkdir -p "$1" "$2"
WORK="$(cd "$1" && pwd)"
OUT="$(cd "$2" && pwd)"
shift 2
PY="${PY:-$repo/.venv/bin/python}"
ENTRIES="${ENTRIES:-160}"
WORKERS="${WORKERS:-8}"
RECIPES="$here/recipes"
if [ $# -eq 0 ]; then
    set -- build cold pipes request settle wide micro fingerprint
fi

timed() {
    local label="$1"
    shift
    "$PY" "$here/timed.py" "$OUT" "$label" -- "$@"
}

fresh() {
    rm -rf "${WORK:?}/$1"
    cp -R "$WORK/base" "$WORK/$1"
}

request_id() {
    "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["request_id"])' "$1"
}

section_build() {
    rm -rf "${WORK:?}/raw" "$WORK/base"
    "$PY" "$here/make_raw.py" "$WORK/raw" "$ENTRIES"
    "$PY" -m mosaic init "$WORK/base" > /dev/null
    "$PY" -m mosaic sources add -m "$WORK/base/dataset.yaml" --kind tracks \
        --path "$WORK/raw" --patterns '*.toml' --src-format synthetic > /dev/null
    "$PY" -m mosaic scan -m "$WORK/base/dataset.yaml" --kind tracks > /dev/null
    timed "convert-tracks cold, thread $WORKERS" "$PY" -m mosaic pipeline run \
        -m "$WORK/base/dataset.yaml" --recipe "@$RECIPES/convert.json" \
        --workers "$WORKERS" --parallel-mode thread
}

section_cold() {
    local name workers mode
    fresh speed_serial
    timed "speed-angvel cold, serial" "$PY" -m mosaic pipeline run \
        -m "$WORK/speed_serial/dataset.yaml" --recipe "@$RECIPES/speed.json" --workers 1
    for setting in "thread4 4 thread" "thread$WORKERS $WORKERS thread" \
        "process$WORKERS $WORKERS process"; do
        read -r name workers mode <<< "$setting"
        fresh "speed_$name"
        timed "speed-angvel cold, $name" "$PY" -m mosaic pipeline run \
            -m "$WORK/speed_$name/dataset.yaml" --recipe "@$RECIPES/speed.json" \
            --workers "$workers" --parallel-mode "$mode"
    done
    "$PY" "$here/builds.py" "$WORK/speed_process$WORKERS" \
        > "$OUT/speed-angvel cold, process$WORKERS.builds.txt"
    for attempt in 1 2; do
        timed "speed-angvel recipe cached, one process, $attempt" "$PY" -m mosaic \
            pipeline run -m "$WORK/speed_thread$WORKERS/dataset.yaml" \
            --recipe "@$RECIPES/speed.json" --workers "$WORKERS" --parallel-mode thread
    done
    for attempt in 1 2; do
        timed "speed-angvel recipe cached, process, $attempt" "$PY" -m mosaic \
            pipeline run -m "$WORK/speed_process$WORKERS/dataset.yaml" \
            --recipe "@$RECIPES/speed.json" --workers "$WORKERS" --parallel-mode process
    done
}

section_pipes() {
    local status
    fresh "pipes_process$WORKERS"
    "$PY" "$here/pipes.py" "$OUT" "speed-angvel cold, process$WORKERS" -- pipeline run \
        -m "$WORK/pipes_process$WORKERS/dataset.yaml" --recipe "@$RECIPES/speed.json" \
        --workers "$WORKERS" --parallel-mode process > /dev/null 2>&1
    status=$?
    tail -1 "$OUT/pipes.tsv"
    return "$status"
}

section_request() {
    local manifest="$WORK/six/dataset.yaml"
    fresh six
    timed "six steps cold, thread $WORKERS" "$PY" -m mosaic pipeline run -m "$manifest" \
        --recipe "@$RECIPES/six-steps.json" --workers "$WORKERS" --parallel-mode thread
    for attempt in 1 2; do
        timed "six steps cached, one process, $attempt" "$PY" -m mosaic pipeline run \
            -m "$manifest" --recipe "@$RECIPES/six-steps.json"
    done
    timed "six steps submit" "$PY" -m mosaic pipeline submit -m "$manifest" \
        --recipe "@$RECIPES/six-steps.json" --json
    local request
    request="$(request_id "$OUT/six steps submit.out")"
    for step in convert smooth speed heading kin agg; do
        timed "step process cached, six steps, $step" "$PY" -m mosaic run -m "$manifest" \
            --graph-request "$request" --step "$step" --json
    done
    timed "pipeline run --request, six cached steps" "$PY" -m mosaic pipeline run \
        -m "$manifest" --request "$request"
    for attempt in 1 2 3; do
        timed "mosaic --version, $attempt" "$PY" -m mosaic --version
    done
    for attempt in 1 2; do
        timed "mosaic run --help, $attempt" "$PY" -m mosaic run --help
        timed "import run and graph, $attempt" "$PY" -c \
            "import mosaic.core.pipeline.run, mosaic.core.pipeline.graph"
        timed "import feature library, $attempt" "$PY" -c \
            "import mosaic.behavior.feature_library"
        timed "import pandas numpy pyarrow pydantic, $attempt" "$PY" -c \
            "import pandas, numpy, pyarrow, pydantic"
    done
    "$PY" -X importtime -m mosaic --version 2> "$OUT/importtime-version.txt" > /dev/null
}

section_settle() {
    local manifest="$WORK/six/dataset.yaml"
    timed "settle submit" "$PY" -m mosaic pipeline submit -m "$manifest" \
        --recipe "@$RECIPES/six-steps.json" --json
    local request
    request="$(request_id "$OUT/settle submit.out")"
    for step in convert smooth speed heading kin agg; do
        timed "settle process cached, $step" "$PY" -m mosaic pipeline settle -m "$manifest" \
            --request "$request" --step "$step" --json
    done
}

section_wide() {
    local manifest="$WORK/wide/dataset.yaml"
    local entries=(--entry s000 --entry s001)
    fresh wide
    timed "wide cold, one process" "$PY" -m mosaic pipeline run -m "$manifest" \
        --recipe "@$RECIPES/wide.json" "${entries[@]}"
    for attempt in 1 2; do
        timed "wide cached, one process, $attempt" "$PY" -m mosaic pipeline run \
            -m "$manifest" --recipe "@$RECIPES/wide.json" "${entries[@]}"
    done
    timed "wide submit" "$PY" -m mosaic pipeline submit -m "$manifest" \
        --recipe "@$RECIPES/wide.json" "${entries[@]}" --json
    local request
    request="$(request_id "$OUT/wide submit.out")"
    for step in convert sp01 sp30 sp60; do
        timed "step process cached, wide, $step" "$PY" -m mosaic run -m "$manifest" \
            --graph-request "$request" --step "$step" --json
    done
}

section_micro() {
    "$PY" "$here/micro.py" "$WORK/base" "$WORK" | tee "$OUT/micro.txt"
}

section_profile() {
    local name request
    local -a samples
    fresh profile_serial
    "$PY" -m cProfile -o "$OUT/speed-cold-serial.prof" -m mosaic pipeline run \
        -m "$WORK/profile_serial/dataset.yaml" --recipe "@$RECIPES/speed.json" --workers 1 \
        > /dev/null 2>&1
    fresh profile_thread
    "$PY" "$here/stacksample.py" "$OUT/speed-cold-thread$WORKERS" -- pipeline run \
        -m "$WORK/profile_thread/dataset.yaml" --recipe "@$RECIPES/speed.json" \
        --workers "$WORKERS" --parallel-mode thread > /dev/null 2>&1
    "$PY" "$here/stacksample.py" "$OUT/speed-cached-thread$WORKERS" -- pipeline run \
        -m "$WORK/profile_thread/dataset.yaml" --recipe "@$RECIPES/speed.json" \
        --workers "$WORKERS" --parallel-mode thread > /dev/null 2>&1
    fresh profile_process
    "$PY" "$here/stacksample.py" "$OUT/speed-cold-process$WORKERS" -- pipeline run \
        -m "$WORK/profile_process/dataset.yaml" --recipe "@$RECIPES/speed.json" \
        --workers "$WORKERS" --parallel-mode process > /dev/null 2>&1
    samples=(speed-cold-thread$WORKERS speed-cached-thread$WORKERS speed-cold-process$WORKERS)
    if [ -f "$OUT/six steps submit.out" ]; then
        request="$(request_id "$OUT/six steps submit.out")"
        "$PY" -m cProfile -o "$OUT/step-process-speed.prof" -m mosaic run \
            -m "$WORK/six/dataset.yaml" --graph-request "$request" --step speed --json \
            > /dev/null 2>&1
        "$PY" -X importtime -m mosaic run -m "$WORK/six/dataset.yaml" \
            --graph-request "$request" --step speed --json \
            2> "$OUT/importtime-step-process.txt" > /dev/null
    fi
    if [ -f "$OUT/settle submit.out" ]; then
        request="$(request_id "$OUT/settle submit.out")"
        "$PY" "$here/stacksample.py" "$OUT/settle-heading" -- pipeline settle \
            -m "$WORK/six/dataset.yaml" --request "$request" --step heading --json \
            > /dev/null 2>&1
        samples+=(settle-heading)
    fi
    if [ -f "$WORK/wide/dataset.yaml" ]; then
        "$PY" "$here/stacksample.py" "$OUT/wide-cached" -- pipeline run \
            -m "$WORK/wide/dataset.yaml" --recipe "@$RECIPES/wide.json" \
            --entry s000 --entry s001 > /dev/null 2>&1
        samples+=(wide-cached)
    fi
    for name in "${samples[@]}"; do
        head -1 "$OUT/$name.stacks.txt"
    done
}

section_fingerprint() {
    local dataset
    for dataset in "$WORK"/*/; do
        dataset="${dataset%/}"
        [ -f "$dataset/dataset.yaml" ] || continue
        "$PY" "$here/fingerprint.py" "$dataset" > "$OUT/fingerprint-$(basename "$dataset").txt"
        echo "$(basename "$dataset"): $(tail -1 "$OUT/fingerprint-$(basename "$dataset").txt")"
    done
}

{
    echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host $(uname -srm), $("$PY" -c 'import os; print(os.cpu_count())') cpus"
    echo "processes on host $(ps -A | wc -l | tr -d ' ')"
    source="$("$PY" -c 'import pathlib, mosaic; print(pathlib.Path(mosaic.__file__).parent)')"
    echo "mosaic $(git -C "$source" rev-parse --short HEAD 2> /dev/null) at $source"
    echo "entries $ENTRIES, workers $WORKERS, sections $*"
} | tee -a "$OUT/host.txt"

status=0
for section in "$@"; do
    echo "== $section"
    "section_$section" || status=1
done
"$PY" "$here/summary.py" "$OUT" "$ENTRIES" "$WORKERS"
exit "$status"
