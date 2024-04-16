#!/usr/bin/env bash
set -euo pipefail

# Python repo
sudo add-apt-repository -y ppa:deadsnakes/ppa

# Update repositories
sudo apt update

# Install runtime dependencies
sudo apt install -y --no-install-recommends \
  python3.12 \
  python3.12-venv

# Install dev dependencies
sudo apt install -y --no-install-recommends \
  direnv \
  make

# Make direnv load automatically for bash
echo 'eval "$(direnv hook bash)"' >>~/.bashrc

# Install jupyter-core
sudo apt install -y --no-install-recommends \
  jupyter-core

# pip install --upgrade --force-reinstall --no-cache-dir jupyter

echo 'alias jupyter-notebook="~/.local/bin/jupyter-notebook --no-browser"' >>~/.bashrc
echo 'export PATH=$PATH:~/.local/bin' >>~/.bashrc
