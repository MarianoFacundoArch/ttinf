#!/bin/bash
# ============================================================
# EC2 setup script for BTC predictor
# Run this ONCE after SSH into the instance:
#   bash scripts/ec2_setup.sh
# ============================================================
set -e

echo "=== 1. System packages ==="
sudo apt update -y
sudo apt install -y python3-pip python3-venv python3-dev git htop tmux

echo "=== 2. Python environment ==="
python3 -m venv ~/venv
source ~/venv/bin/activate

pip install --upgrade pip
pip install \
    pandas>=2.2 \
    pyarrow>=15.0 \
    numpy>=1.26 \
    lightgbm>=4.0 \
    scikit-learn>=1.4 \
    requests \
    websockets>=12.0 \
    aiohttp>=3.9

echo "=== 3. Verify installs ==="
python -c "
import pandas, pyarrow, numpy, lightgbm, sklearn, requests
print(f'  pandas:   {pandas.__version__}')
print(f'  pyarrow:  {pyarrow.__version__}')
print(f'  numpy:    {numpy.__version__}')
print(f'  lightgbm: {lightgbm.__version__}')
print(f'  sklearn:  {sklearn.__version__}')
print('  All OK!')
"

echo "=== 4. Set up TARDIS API key ==="
# Add to bashrc so it persists
if ! grep -q TARDIS_API_KEY ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# Tardis API key' >> ~/.bashrc
    echo 'export TARDIS_API_KEY="TD.QAY0WvGO2P2swjRN.ZFCCjFCULixtErg.clGX1LmSrXnszy3.MFJSL2Ek8KEorHD.AtnL1h7voQzDDer.rckG"' >> ~/.bashrc
    echo '# Auto-activate venv' >> ~/.bashrc
    echo 'source ~/venv/bin/activate' >> ~/.bashrc
fi

echo "=== 5. Check disk space ==="
df -h /

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Run: source ~/.bashrc"
echo "  Then: cd ~/predictxgboost"
echo "        python -m src.data.download_all --start 2025-11-11 --end 2026-03-11 --workers 8"
echo "============================================================"
