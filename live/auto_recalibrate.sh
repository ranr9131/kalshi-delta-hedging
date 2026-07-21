#!/bin/bash
# Daily recalibration runner.  Called by kalshi-recal.timer at 06:00 UTC.
# - Refits calibration_v2.json from accumulated live data
# - Uses --merge so existing entries with more data are preserved
# - Appends output to recal.log
# - The sniper_v2 fair-price model hot-reloads on file mtime change,
#   so no service restart needed.
set -e

cd /home/ec2-user/kalshi-delta-hedging/live

LOG=/home/ec2-user/kalshi-delta-hedging/live/recal.log
echo "================================================================" >> "$LOG"
echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ')  daily recalibration starting" >> "$LOG"
echo "================================================================" >> "$LOG"

/usr/bin/python3.11 /home/ec2-user/kalshi-delta-hedging/live/fit_calibration_live.py \
    --merge \
    --snipes /home/ec2-user/kalshi-delta-hedging/live/snipes_v2.csv \
    --out    /home/ec2-user/kalshi-delta-hedging/live/calibration_v2.json \
    >> "$LOG" 2>&1

echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ')  done" >> "$LOG"
echo "" >> "$LOG"
