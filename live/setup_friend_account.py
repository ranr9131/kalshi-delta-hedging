"""
Add a second Kalshi account to live/.env safely.

Reads the new account's KEY_ID and PRIVATE_KEY from stdin (so the secret
never lives on local disk and flows directly via encrypted SSH).

Usage (over ssh):
  ssh ec2-user@HOST 'python3.11 /home/ec2-user/.../setup_friend_account.py FRIEND' \
    <<'EOF'
  KEY_ID=...
  PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----
  ...
  -----END RSA PRIVATE KEY-----
  EOF

What it does:
  1. Parses stdin for KEY_ID= and PRIVATE_KEY= (multi-line PEM supported)
  2. Reads current /home/ec2-user/.../live/.env
  3. Adds: <LABEL>_KEY_ID, <LABEL>_PRIVATE_KEY, plus
     LEO_KEY_ID/LEO_PRIVATE_KEY (aliases of existing KALSHI_*)
     and ACCOUNTS=LEO,<LABEL>
  4. Backs up old .env, writes new atomically, chmod 600
  5. Validates BOTH accounts by calling get_balance()
  6. If either fails, restores backup and exits non-zero

Won't touch any running sniper services — that's a separate step.
"""
from __future__ import annotations
import os, sys, shutil, time
sys.path.insert(0, "/home/ec2-user/kalshi-delta-hedging/live")
from dotenv import dotenv_values
import kalshi_auth, kalshi_trade


ENV_PATH = "/home/ec2-user/kalshi-delta-hedging/live/.env"


def parse_stdin() -> tuple[str, str]:
    """Read KEY_ID= and PRIVATE_KEY= from stdin.  PRIVATE_KEY may span
    multiple lines (PEM) — collect everything from that line up to and
    including the END marker."""
    raw = sys.stdin.read()
    key_id = None
    pem_lines = []
    in_pem = False
    for line in raw.splitlines():
        if line.startswith("KEY_ID="):
            key_id = line.split("=", 1)[1].strip()
        elif line.startswith("PRIVATE_KEY="):
            in_pem = True
            first = line.split("=", 1)[1]
            # Strip surrounding quotes if user added them
            first = first.lstrip('"').lstrip("'")
            pem_lines.append(first)
        elif in_pem:
            pem_lines.append(line)
            if "END RSA PRIVATE KEY" in line:
                # Trim trailing quote if present on the last line
                pem_lines[-1] = pem_lines[-1].rstrip('"').rstrip("'")
                in_pem = False
    if not key_id or not pem_lines:
        print("ERROR: missing KEY_ID or PRIVATE_KEY in stdin", file=sys.stderr)
        sys.exit(2)
    return key_id, "\n".join(pem_lines)


def serialize_env(d: dict) -> str:
    """Render dict to .env format.  Multi-line values get double-quoted."""
    out = []
    for k, v in d.items():
        if v is None:
            continue
        sv = str(v)
        if "\n" in sv:
            out.append(f'{k}="{sv}"')
        elif " " in sv or "#" in sv:
            out.append(f'{k}="{sv}"')
        else:
            out.append(f"{k}={sv}")
    return "\n".join(out) + "\n"


def main():
    label = sys.argv[1].upper() if len(sys.argv) > 1 else "FRIEND"
    print(f"adding account: {label}")

    new_key_id, new_pem = parse_stdin()
    print(f"  parsed KEY_ID:  {new_key_id}")
    print(f"  parsed PEM:     {len(new_pem.splitlines())} lines, "
          f"starts {new_pem[:40]!r}")

    # Validate the new key BEFORE touching .env
    try:
        priv = kalshi_auth.load_private_key(new_pem)
    except Exception as e:
        print(f"ERROR: PEM didn't load: {e}", file=sys.stderr)
        sys.exit(3)
    try:
        bal = kalshi_trade.get_balance(priv, new_key_id)
        if bal is None:
            print(f"ERROR: get_balance returned None — key probably wrong",
                  file=sys.stderr)
            sys.exit(4)
        print(f"  ✓ new account balance reads: ${bal:.2f}")
    except Exception as e:
        print(f"ERROR: balance fetch threw: {e}", file=sys.stderr)
        sys.exit(5)

    # Read existing .env
    existing = dotenv_values(ENV_PATH)
    if not existing.get("KALSHI_API_KEY_ID") or not existing.get("KALSHI_PRIVATE_KEY"):
        print(f"ERROR: existing .env missing KALSHI_API_KEY_ID/PRIVATE_KEY",
              file=sys.stderr)
        sys.exit(6)
    print(f"  existing KALSHI_API_KEY_ID present, length ok")

    # Validate the EXISTING key too (sanity check)
    try:
        prev_priv = kalshi_auth.load_private_key(existing["KALSHI_PRIVATE_KEY"])
        prev_bal = kalshi_trade.get_balance(prev_priv, existing["KALSHI_API_KEY_ID"])
        print(f"  ✓ existing account (LEO) balance reads: ${prev_bal:.2f}")
    except Exception as e:
        print(f"ERROR: existing key failed to load: {e}", file=sys.stderr)
        sys.exit(7)

    # Build new .env contents
    new = dict(existing)
    new["LEO_KEY_ID"]      = existing["KALSHI_API_KEY_ID"]
    new["LEO_PRIVATE_KEY"] = existing["KALSHI_PRIVATE_KEY"]
    new[f"{label}_KEY_ID"]      = new_key_id
    new[f"{label}_PRIVATE_KEY"] = new_pem
    new["ACCOUNTS"] = f"LEO,{label}"

    # Backup + atomic write
    backup = ENV_PATH + f".bak.{int(time.time())}"
    shutil.copyfile(ENV_PATH, backup)
    tmp = ENV_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(serialize_env(new))
    os.chmod(tmp, 0o600)
    os.replace(tmp, ENV_PATH)
    print(f"  ✓ wrote new .env ({len(new)} vars)  backup → {os.path.basename(backup)}")

    # Re-validate by reading back and trying both keys
    fresh = dotenv_values(ENV_PATH)
    for tag in ("LEO", label):
        try:
            p = kalshi_auth.load_private_key(fresh[f"{tag}_PRIVATE_KEY"])
            b = kalshi_trade.get_balance(p, fresh[f"{tag}_KEY_ID"])
            print(f"  ✓ re-read {tag}: balance ${b:.2f}")
        except Exception as e:
            print(f"  ✗ re-read {tag} FAILED: {e}", file=sys.stderr)
            print(f"  restoring backup", file=sys.stderr)
            shutil.copyfile(backup, ENV_PATH)
            sys.exit(8)

    total = sum(kalshi_trade.get_balance(
                  kalshi_auth.load_private_key(fresh[f"{t}_PRIVATE_KEY"]),
                  fresh[f"{t}_KEY_ID"]) or 0
                for t in ("LEO", label))
    leo_bal = kalshi_trade.get_balance(
        kalshi_auth.load_private_key(fresh["LEO_PRIVATE_KEY"]),
        fresh["LEO_KEY_ID"]) or 0
    other_bal = total - leo_bal
    print()
    print("=" * 50)
    print(f"  Total capital across both accounts: ${total:.2f}")
    print(f"    LEO:    ${leo_bal:.2f}  ({leo_bal/total*100:.1f}%)")
    print(f"    {label}: ${other_bal:.2f}  ({other_bal/total*100:.1f}%)")
    print("=" * 50)
    print()
    print("Setup complete. Next steps:")
    print("  sudo systemctl stop kalshi-sniper           # halt V1")
    print("  sudo systemctl disable kalshi-sniper        # don't auto-start")
    print("  sudo systemctl enable --now kalshi-sniper-multi  # go live")


if __name__ == "__main__":
    main()
