#!/bin/bash
# Compare paper-S2 (dh-target) vs paper-S3 (mispricing-only) results.
# Run any time: ./compare_paper.sh
cd "$(dirname "$0")"

for tag in s2 s3; do
    LOG="window_log.paper-${tag}.csv"
    if [[ ! -f "$LOG" ]]; then
        echo "── paper-${tag}: no windows logged yet ──"
        continue
    fi
    echo "── paper-${tag} ($(wc -l < "$LOG" | tr -d ' ') lines, last entry: $(tail -1 "$LOG" | cut -d',' -f1)) ──"
    awk -F',' 'NR>1 {
        n++
        pnl += $18
        if ($18 < worst) {worst = $18; worst_t = $1}
        if ($18 > 0.01) wins++
        if ($18 < -0.01) losses++
        wagered += $13
    }
    END {
        printf "  %d windows, %d wins / %d losses (%.0f%% win rate)\n", n, wins, losses, (wins/n)*100
        printf "  Total PnL: $%+.2f  |  Wagered: $%.2f  |  ROI: %+.2f%%\n", pnl, wagered, (pnl/wagered)*100
        printf "  Worst: $%+.2f at %s\n", worst, worst_t
    }' "$LOG"
    echo ""
done
