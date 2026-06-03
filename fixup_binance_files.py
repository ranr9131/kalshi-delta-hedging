"""Re-aggregate existing binance_30s_*.json files from 1s ms-keyed data into proper 30s s-keyed buckets."""
import os, json, glob, sys

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")

STEP_SECS = 30


def reaggregate(path):
    with open(path) as f:
        d = json.load(f)
    # Detect: is this ms-keyed 1s data?
    keys = sorted(int(k) for k in d.keys())
    if not keys:
        return False
    if len(str(keys[0])) < 12:
        return False  # already seconds-keyed
    buckets = {}
    for k_str, candle in d.items():
        ts_ms = int(k_str)
        ts_s = ts_ms // 1000
        bucket_start = (ts_s // STEP_SECS) * STEP_SECS
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        v = float(candle["volume"])
        if bucket_start not in buckets:
            buckets[bucket_start] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        else:
            b = buckets[bucket_start]
            b["high"] = max(b["high"], h)
            b["low"] = min(b["low"], l)
            b["close"] = c
            b["volume"] += v
    out = {str(k): v for k, v in sorted(buckets.items())}
    with open(path, "w") as f:
        json.dump(out, f)
    return True


files = sorted(glob.glob(os.path.join(CACHE_DIR, "binance_30s_*.json")))
print(f"Found {len(files)} files")
fixed = 0
for path in files:
    if reaggregate(path):
        fixed += 1
print(f"Fixed: {fixed}")
print(f"Already-correct: {len(files) - fixed}")

# Verify one fixed file
if files:
    with open(files[0]) as f:
        d = json.load(f)
    keys = sorted(int(k) for k in d.keys())
    print(f"\nSample verification of {os.path.basename(files[0])}:")
    print(f"  bucket count: {len(keys)}")
    print(f"  key digits: {len(str(keys[0]))}")
    print(f"  first key: {keys[0]}")
    print(f"  diff key 0->1: {keys[1]-keys[0]}s")
