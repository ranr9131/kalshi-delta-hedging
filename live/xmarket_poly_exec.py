"""
Polymarket CLOB execution leg for the cross-venue arb system.

Credentials are read from .env ON YOUR BOX — the raw wallet key never leaves it
and is never passed to anyone. Paper mode needs no creds at all.

Required .env vars for LIVE Polymarket trading (you add these yourself):
  POLY_PRIVATE_KEY = 0x...      # your Polygon wallet private key (controls funds!)
  POLY_FUNDER      = 0x...      # your Polymarket deposit/proxy address
  POLY_SIG_TYPE    = 1          # 0=EOA(MetaMask), 1=email/magic proxy, 2=Gnosis safe
One-time on-chain setup also required: USDC on Polygon + CLOB allowances
(run `--check` after funding; see set_allowances()).

Marketable buys use FAK (Fill-And-Kill): fills what it can at/under the limit
price immediately, cancels the rest — no resting order. Exactly what a hedge
leg wants.
"""
from __future__ import annotations
import os
from typing import Optional
from dotenv import dotenv_values

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon


class PolyExec:
    def __init__(self, paper: bool = True, env_path: Optional[str] = None):
        self.paper = paper
        self.client = None
        self._ready = False
        if paper:
            return
        env = dotenv_values(env_path or os.path.join(os.path.dirname(__file__), ".env"))
        pk = env.get("POLY_PRIVATE_KEY")
        if not pk:
            raise RuntimeError("POLY_PRIVATE_KEY not in .env — required for live Poly trading")
        funder = env.get("POLY_FUNDER")
        sig_type = int(env.get("POLY_SIG_TYPE", "1"))
        from py_clob_client.client import ClobClient
        self.client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID,
                                 funder=funder, signature_type=sig_type)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())
        self._ready = True

    # ── account / setup ────────────────────────────────────────────────────
    def usdc_balance(self) -> Optional[float]:
        if self.paper:
            return None
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        try:
            r = self.client.get_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
            # balances are in micro-USDC (6 decimals)
            return float(r.get("balance", 0)) / 1e6
        except Exception as e:
            print("usdc_balance err:", e)
            return None

    def set_allowances(self):
        """One-time: approve the CLOB exchange to move your USDC. Run once after
        funding (this DOES send on-chain txs — you run it deliberately)."""
        if self.paper:
            print("[paper] would set USDC allowances"); return
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        self.client.update_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        print("USDC allowance update submitted.")

    def tick_size(self, token_id: str) -> float:
        try:
            import requests
            b = requests.get(f"{HOST}/book", params={"token_id": token_id}, timeout=6).json()
            return float(b.get("tick_size", 0.01))
        except Exception:
            return 0.01

    # ── execution ──────────────────────────────────────────────────────────
    def buy(self, token_id: str, size: float, max_price: float) -> dict:
        """Marketable FAK buy: up to `size` shares at <= max_price.
        Returns {filled_size, avg_price, ok, raw}. Paper mode simulates a full
        fill at max_price."""
        if self.paper:
            return {"filled_size": size, "avg_price": max_price, "ok": True, "paper": True}
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY
        tick = self.tick_size(token_id)
        price = round(round(max_price / tick) * tick, 4)
        price = min(0.999, max(tick, price))
        try:
            signed = self.client.create_order(
                OrderArgs(token_id=token_id, price=price, size=size, side=BUY))
            resp = self.client.post_order(signed, OrderType.FAK)
            filled = float(resp.get("takingAmount", resp.get("size_matched", 0)) or 0)
            return {"filled_size": filled, "avg_price": price,
                    "ok": resp.get("success", True), "raw": resp}
        except Exception as e:
            return {"filled_size": 0.0, "avg_price": None, "ok": False, "error": str(e)}


def make(paper: bool, env_path: Optional[str] = None) -> "PolyExec":
    return PolyExec(paper=paper, env_path=env_path)
