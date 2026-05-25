#!/usr/bin/env bash
set -euo pipefail

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q -e .
