"""
Kalshi API authentication — RSA-SHA256-PSS signing.
Ported from nba-odds-visualizer/server.js kalshiSign().
"""

import base64
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding


def load_private_key(pem_string: str):
    """Load RSA private key from PEM string (PKCS1 or PKCS8 format)."""
    return serialization.load_pem_private_key(
        pem_string.strip().encode("utf-8"),
        password=None,
    )


def make_auth_headers(private_key, api_key_id: str, method: str, path: str) -> dict:
    """
    Build the three Kalshi auth headers for one request.

    method: uppercase HTTP verb, e.g. "GET", "POST", "DELETE"
    path:   path only, no query string, e.g. "/trade-api/v2/portfolio/orders"

    Signing: RSA-SHA256-PSS over (timestamp_ms + method + path).
    salt_length = SHA256 digest size (32) — must match Node.js RSA_PSS_SALTLEN_DIGEST.
    """
    timestamp_ms = str(int(time.time() * 1000))
    message = (timestamp_ms + method + path).encode("utf-8")

    signature = private_key.sign(
        message,
        asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256().digest_size,
        ),
        hashes.SHA256(),
    )

    return {
        "KALSHI-ACCESS-KEY":       api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        "Content-Type":            "application/json",
    }


if __name__ == "__main__":
    # Quick sanity check — prints headers without hitting the API
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dotenv import dotenv_values
    env = dotenv_values(os.path.join(os.path.dirname(__file__), ".env"))
    key = load_private_key(env["KALSHI_PRIVATE_KEY"])
    headers = make_auth_headers(key, env["KALSHI_API_KEY_ID"], "GET", "/trade-api/v2/markets")
    print("Auth headers generated successfully:")
    for k, v in headers.items():
        print(f"  {k}: {v[:40]}..." if len(v) > 40 else f"  {k}: {v}")
