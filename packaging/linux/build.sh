#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --onefile --name my-ai my_ai_package/main.py

echo ""
echo "Build complete! Executable at: dist/my-ai"
echo ""
echo "To run, first set environment variables:"
echo "  export RUNPOD_API_KEY='your_key'"
echo "  export VLLM_API_KEY='your_key'"
echo "  ./dist/my-ai"
