#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

echo "Install Causal-Conv1d..."
cd /workspaces/Polaris/third_party/causal-conv1d
pip install . --no-build-isolation -v

echo "Install Mamba..."
cd /workspaces/Polaris/third_party/mamba
pip install . --no-build-isolation -v

echo "Setup complete!"