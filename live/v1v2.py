import csv, sys
def f(x):
    try: return float(x)
    except: return None
res={r["ticker"]:r["result"] for r in csv.DictReader(open("settlements.csv"))}
def fee(p): p=p/100.0; return 7.0*p*(1-p)
def net(r):
    fill=f(r["fill_cents_est"]) or f(r["limit_cents"]); rr=res.get(r["ticker"])
    if fill is None or rr not in ("yes","no"): return None
    yes=rr=="yes"
    g=((100.0 if yes else 0.0)-fill) if r["side"]=="yes" else ((100.0 if not yes else 0.0)-fill)
    return g-fee(fill)
def stats(path,since=None):
    rows=list(csv.DictReader(open(path)))
    if since: rows=[r for r in rows if r["ts_iso"]>=since]
    v=[net(r) for r in rows]; v=[x for x in v if x is not None]
    if not v: return "no settled"
    return "n=%4d  win=%4.1f%%  NET=%+.2fc/contract  total=%+.2f$/contract-snipe"%(
        len(v),100*sum(1 for x in v if x>0)/len(v),sum(v)/len(v),sum(v)/100)
for label,path in [("V1 (snipes.csv)","snipes.csv"),("V2 (snipes_v2.csv)","snipes_v2.csv")]:
    print("%-20s ALL : %s"%(label,stats(path)))
    print("%-20s 6/07: %s"%(label,stats(path,"2026-06-07")))
