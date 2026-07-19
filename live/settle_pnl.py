import csv, requests, os, sys
sys.path.insert(0,".")
from dotenv import dotenv_values
import kalshi_auth
env=dotenv_values(".env")
kid=env.get("KALSHI_API_KEY_ID") or env.get("LEO_KEY_ID")
pem=env.get("KALSHI_PRIVATE_KEY") or env.get("LEO_PRIVATE_KEY")
pk=kalshi_auth.load_private_key(pem)
BASE="https://api.elections.kalshi.com"
def result_of(t):
    p="/trade-api/v2/markets/"+t
    h=kalshi_auth.make_auth_headers(pk,kid,"GET",p)
    m=requests.get(BASE+p,headers=h,timeout=15).json().get("market",{})
    return m.get("result"), m.get("status")
fills=list(csv.DictReader(open("mm_shadow_fills.csv")))
cache={}; settled=0; pnl=0.0; openpos=0; rows=[]
for f in fills:
    t=f["ticker"]
    if t not in cache: cache[t]=result_of(t)
    res,status=cache[t]
    price=float(f["fill_price_c"]); side=f["side"]
    if res in ("yes","no"):
        win=100.0 if res=="yes" else 0.0
        p=(win-price) if side=="buy_yes" else (price-win)
        settled+=1; pnl+=p
        rows.append((t[:24],side,price,res,round(p,1)))
    else:
        openpos+=1
print("fills=%d  settled=%d  still-open=%d"%(len(fills),settled,openpos))
print("%-24s %-8s %4s %4s %7s"%("ticker","side","px","res","pnl_c"))
for r in rows: print("%-24s %-8s %4.0f %4s %7.1f"%r)
if settled:
    print("\nREAL settled P&L: %+.1fc over %d fills = %+.2fc/fill avg"%(pnl,settled,pnl/settled))
    print("at 10 contracts/fill: %+.2f dollars on settled fills so far"%(pnl/100*10))
