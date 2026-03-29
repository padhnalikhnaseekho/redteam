#!/usr/bin/env bash
# ==============================================================================
# Full Red Team Benchmark: All attacks x All defenses x All models
#
# Usage:
#   chmod +x scripts/run_full_benchmark.sh
#   ./scripts/run_full_benchmark.sh
#
# Optional:
#   ./scripts/run_full_benchmark.sh --skip-multi-agent   # Skip D4 (saves API cost)
#   ./scripts/run_full_benchmark.sh --models groq-llama   # Single model only
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Timestamp for results directory
TIMESTAMP=$(date +"%m%d_%H%M")
RESULTS_DIR="results/results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Parse args
SKIP_MULTI_AGENT=false
MODELS=("groq-llama" "mistral-large")

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-multi-agent)
            SKIP_MULTI_AGENT=true
            shift
            ;;
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

# Defense configurations
DEFENSES=("input_filter" "output_validator" "guardrails" "human_in_loop")
if [ "$SKIP_MULTI_AGENT" = false ]; then
    DEFENSES+=("multi_agent")
fi

ALL_DEFENSES="${DEFENSES[*]}"

# Counters
TOTAL_RUNS=0
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Calculate total runs: (no_defense + each_defense + all_combined) x models
RUNS_PER_MODEL=$(( 1 + ${#DEFENSES[@]} + 1 ))
TOTAL_RUNS=$(( RUNS_PER_MODEL * ${#MODELS[@]} ))

echo "============================================================"
echo "  CommodityRedTeam Full Benchmark"
echo "============================================================"
echo "  Timestamp:    $TIMESTAMP"
echo "  Results dir:  $RESULTS_DIR"
echo "  Models:       ${MODELS[*]}"
echo "  Defenses:     ${DEFENSES[*]}"
echo "  Total runs:   $TOTAL_RUNS"
echo "  Skip D4:      $SKIP_MULTI_AGENT"
echo "============================================================"
echo ""

run_attack() {
    local model="$1"
    local label="$2"
    local output_file="$3"
    shift 3
    local defense_args=("$@")

    local run_num=$((COMPLETED + FAILED + 1))
    echo "[$run_num/$TOTAL_RUNS] $model | $label"

    local cmd="python scripts/run_attacks.py --model $model --output $output_file"
    if [ ${#defense_args[@]} -gt 0 ]; then
        cmd="$cmd --defense ${defense_args[*]}"
    fi

    if eval "$cmd" > "${output_file%.json}.log" 2>&1; then
        COMPLETED=$((COMPLETED + 1))
        echo "         OK -> $output_file"
    else
        FAILED=$((FAILED + 1))
        echo "         FAILED (see ${output_file%.json}.log)"
    fi
    echo ""
}

# ── Run benchmark ────────────────────────────────────────────────

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's/-/_/g')

    echo "============================================================"
    echo "  Model: $MODEL"
    echo "============================================================"

    # 1. No defense
    run_attack "$MODEL" "no_defense" \
        "$RESULTS_DIR/${MODEL_SHORT}_no_defense.json"

    # 2. Each defense individually
    for DEFENSE in "${DEFENSES[@]}"; do
        run_attack "$MODEL" "$DEFENSE" \
            "$RESULTS_DIR/${MODEL_SHORT}_${DEFENSE}.json" \
            "$DEFENSE"
    done

    # 3. All defenses combined
    run_attack "$MODEL" "all_combined" \
        "$RESULTS_DIR/${MODEL_SHORT}_all_combined.json" \
        ${ALL_DEFENSES}

done

# ── Merge all results into a single CSV ──────────────────────────

echo "============================================================"
echo "  Generating combined report"
echo "============================================================"

python3 -c "
import csv, json, glob, os

results_dir = '$RESULTS_DIR'
all_rows = []
fieldnames = None

for jf in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    with open(jf) as f:
        data = json.load(f)

    meta = data.get('metadata', {})
    model = meta.get('model', 'unknown')
    defense = meta.get('defense', 'none')

    for r in data.get('results', []):
        r['model'] = model
        r['defense'] = defense
        r['source_file'] = os.path.basename(jf)
        if fieldnames is None:
            fieldnames = list(r.keys())
        all_rows.append(r)

if all_rows:
    # Combined CSV
    csv_path = os.path.join(results_dir, 'all_results_combined.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in all_rows:
            row = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
            writer.writerow(row)
    print(f'  Combined CSV: {csv_path} ({len(all_rows)} rows)')

    # Summary CSV (grouped by model x defense)
    from collections import defaultdict
    groups = defaultdict(lambda: {'total': 0, 'success': 0, 'detected': 0, 'impact': 0.0})
    for r in all_rows:
        key = (r['model'], r['defense'])
        groups[key]['total'] += 1
        groups[key]['success'] += 1 if r.get('success') else 0
        groups[key]['detected'] += 1 if r.get('detected') else 0
        groups[key]['impact'] += r.get('financial_impact', 0) if r.get('success') else 0

    summary_path = os.path.join(results_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'defense', 'total_attacks', 'successful', 'asr_pct', 'detected', 'detection_rate_pct', 'total_impact_usd'])
        for (model, defense), s in sorted(groups.items()):
            asr = round(100 * s['success'] / s['total'], 1) if s['total'] else 0
            dr = round(100 * s['detected'] / s['total'], 1) if s['total'] else 0
            writer.writerow([model, defense, s['total'], s['success'], asr, s['detected'], dr, int(s['impact'])])
    print(f'  Summary CSV:  {summary_path}')

    # Print summary table
    print()
    print(f'  {\"Model\":<20} {\"Defense\":<25} {\"ASR\":>6} {\"Detected\":>10} {\"Impact ($)\":>14}')
    print(f'  {\"-\"*20} {\"-\"*25} {\"-\"*6} {\"-\"*10} {\"-\"*14}')
    for (model, defense), s in sorted(groups.items()):
        asr = f\"{100*s['success']/s['total']:.0f}%\" if s['total'] else '0%'
        dr = f\"{100*s['detected']/s['total']:.0f}%\" if s['total'] else '0%'
        print(f'  {model:<20} {defense:<25} {asr:>6} {dr:>10} {\"\${:,.0f}\".format(s[\"impact\"]):>14}')
else:
    print('  No results found.')
"

# ── Final summary ────────────────────────────────────────────────

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  Benchmark Complete"
echo "============================================================"
echo "  Results dir:  $RESULTS_DIR"
echo "  Completed:    $COMPLETED / $TOTAL_RUNS"
echo "  Failed:       $FAILED / $TOTAL_RUNS"
echo "  Duration:     ${MINUTES}m ${SECONDS}s"
echo ""
echo "  Files:"
ls -1 "$RESULTS_DIR"/*.csv "$RESULTS_DIR"/*.json 2>/dev/null | sed 's/^/    /'
echo "============================================================"
