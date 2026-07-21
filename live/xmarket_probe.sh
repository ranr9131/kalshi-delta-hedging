#!/bin/bash
# IOC fillability probe for cross-venue WNBA locks (the 7/13 persistence-audit gate).
# Fires up to 3 REAL kalshi-only IOC attempts (5 contracts) on live-game locks >=5c,
# then stops. Worst case per attempt ~$3.50 if filled unhedged; a fill is the
# bullish outcome (quotes are real, not phantoms). 48h timeout. Run manually:
#   cd ~/kalshi-delta-hedging/live && nohup ./xmarket_probe.sh > xmarket_probe.log 2>&1 &
cd /home/ec2-user/kalshi-delta-hedging/live
N0=$(grep -c kalshi-only xmarket_trades.csv 2>/dev/null || echo 0)
END=$((SECONDS + 48*3600))
echo "$(date -u +%FT%TZ) probe loop start, baseline kalshi-only rows: $N0"
while [ $SECONDS -lt $END ]; do
  N=$(grep -c kalshi-only xmarket_trades.csv 2>/dev/null || echo 0)
  if [ $((N - N0)) -ge 3 ]; then
    echo "$(date -u +%FT%TZ) 3 probes fired — stopping"
    break
  fi
  python3.11 -u xmarket_arb.py --live --yes --kalshi-only --contracts 5 --min-edge 0.05 --once
  sleep 10
done
echo "$(date -u +%FT%TZ) probe loop exit"
