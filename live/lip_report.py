"""
LIP farmer dashboard. Reads lip_state.json + CSVs; no API calls needed
(pass --live to also pull balance and program paid_out status).

  python3 lip_report.py
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import lip_config as cfg


def load_state():
    try:
        with open(cfg.STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def accrual_by_day():
    out = defaultdict(float)
    if not os.path.exists(cfg.ACCRUAL_CSV):
        return out
    with open(cfg.ACCRUAL_CSV) as f:
        for row in csv.DictReader(f):
            out[row["ts"][:10]] += float(row["accrued_dollars"] or 0)
    return out


def main():
    state = load_state()
    held = state.get("held", {})
    mode = "PAPER" if state.get("paper", True) else "LIVE"
    print(f"=== LIP farmer report ({mode})  updated {state.get('updated', '?')} ===\n")

    if held:
        rows = sorted(held.items(), key=lambda kv: -kv[1]["est_per_day"])
        total_day = sum(v["est_per_day"] for _, v in rows)
        total_cap = sum(v["capital"] for _, v in rows)
        total_loss = sum(v["worst_loss"] for _, v in rows)
        print(f"{'ticker':42s} {'share':>6s} {'$/day':>7s} {'quote':>9s} "
              f"{'size':>6s} {'cap$':>7s} {'risk$':>6s}  ends")
        for t, v in rows:
            quote = f"{v['yes_price'] or '-'}/{v['no_price'] or '-'}"
            print(f"{t:42s} {v['share']:6.1%} {v['est_per_day']:7.2f} {quote:>9s} "
                  f"{v['size']:6d} {v['capital']:7.0f} {v['worst_loss']:6.0f}  "
                  f"{v['end_date'][:10]}")
        print(f"\n  {len(rows)} markets | est ${total_day:.2f}/day | "
              f"${total_cap:.0f} capital | ${total_loss:.0f} worst-case loss")
    else:
        print("  no markets held")

    print(f"\n  accrued estimate total: ${state.get('accrual_total', 0):.2f}")
    days = accrual_by_day()
    for day in sorted(days)[-7:]:
        print(f"    {day}: ${days[day]:.2f}")

    positions = {t: p for t, p in (state.get("positions") or {}).items()
                 if p.get("yes") or p.get("no")}
    if positions:
        print("\n  open positions from fills:")
        for t, p in positions.items():
            print(f"    {t}: yes={p.get('yes', 0)} no={p.get('no', 0)} "
                  f"cost=${p.get('cost', 0):.2f}")

    cooldowns = state.get("cooldowns") or {}
    now = datetime.now(timezone.utc).timestamp()
    active_cd = {t: u for t, u in cooldowns.items() if u > now}
    if active_cd:
        print(f"\n  cooldowns: {', '.join(active_cd)}")

    if "--live" in sys.argv:
        from dotenv import dotenv_values
        import kalshi_auth, lip_api
        env = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
        pk = kalshi_auth.load_private_key(env["KALSHI_PRIVATE_KEY"])
        kid = env["KALSHI_API_KEY_ID"]
        print(f"\n  balance: ${lip_api.get_balance(pk, kid):.2f}")


if __name__ == "__main__":
    main()
