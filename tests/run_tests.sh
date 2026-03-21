#!/bin/bash
# Run the hosted-tinker test suite against a running server.
#
# Usage:
#   # Against local server:
#   bash tests/run_tests.sh
#
#   # Against remote server:
#   TINKER_BASE_URL=http://10.0.0.1:8000 bash tests/run_tests.sh
#
#   # Run specific test file:
#   bash tests/run_tests.sh tests/test_forward_backward.py
#
#   # Run with verbose output:
#   bash tests/run_tests.sh -v
#
# Environment variables:
#   TINKER_BASE_URL  - Server URL (default: http://localhost:8000)
#   TINKER_API_KEY   - API key (default: tml-dummy)
#   TINKER_MODEL     - Model name (default: Qwen/Qwen3-30B-A3B)
#   TINKER_LORA_RANK - LoRA rank (default: 32)

set -euo pipefail
cd "$(dirname "$0")/.."

export TINKER_BASE_URL="${TINKER_BASE_URL:-http://localhost:8000}"
export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export TINKER_MODEL="${TINKER_MODEL:-Qwen/Qwen3-30B-A3B}"
export TINKER_LORA_RANK="${TINKER_LORA_RANK:-32}"

echo "============================================"
echo "Hosted-Tinker Test Suite"
echo "  Server: ${TINKER_BASE_URL}"
echo "  Model:  ${TINKER_MODEL}"
echo "  Rank:   ${TINKER_LORA_RANK}"
echo "============================================"

# Check server is reachable
if ! curl -sf "${TINKER_BASE_URL}/api/v1/healthz" > /dev/null 2>&1; then
    echo "ERROR: Server not reachable at ${TINKER_BASE_URL}"
    echo "Start the server first, e.g.:"
    echo "  python -m hosted_tinker.api --base-model ${TINKER_MODEL} --backend jax --backend-config '{...}'"
    exit 1
fi
echo "Server is healthy."
echo

# Install test dependencies if needed
pip install -q pytest pytest-asyncio tinker 2>/dev/null || true

# Run tests
if [ $# -eq 0 ]; then
    # Run all tests with short summary
    python -m pytest tests/ -x --tb=short -q "$@"
else
    python -m pytest "$@" --tb=short
fi
