#!/bin/zsh
cd "$(dirname "$0")"
export SSL_CERT_FILE="./venv/lib/python3.11/site-packages/certifi/cacert.pem"
exec ./venv/bin/python plot_balance.py
