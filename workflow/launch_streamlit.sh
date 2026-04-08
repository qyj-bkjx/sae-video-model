#!/bin/bash
###############################################################################
# SGLang Video Generation Streamlit App Launcher
###############################################################################

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to script directory
cd "$SCRIPT_DIR"

# Default port
PORT=${PORT:-8500}

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SGLang Video Generation Streamlit App${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Starting Streamlit server on port $PORT...${NC}"
echo ""

# Launch Streamlit
streamlit run app.py \
    --server.port "$PORT" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --browser.gatherUsageStats false
