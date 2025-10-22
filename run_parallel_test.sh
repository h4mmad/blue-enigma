#!/bin/bash

# Script to run the parallel vs sequential timing test

echo "=========================================="
echo "Installing async dependencies..."
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found. Install dependencies globally or create venv first."
fi

# Install required async libraries
pip install aiohttp nest-asyncio --quiet

echo ""
echo "=========================================="
echo "Running parallel timing test..."
echo "=========================================="
echo ""

python test_parallel_timing.py

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
