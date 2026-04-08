#!/bin/bash
###############################################################################
# SGLang Server Launcher
###############################################################################

export RING_DEGREE=2
export ULYSSES_DEGREE=4
export TP_SIZE=1
export PORT=30000

# 模型路径（用于判断模型类型）。请通过环境变量 MOVA_MODEL_PATH 设置，例如：
#   export MOVA_MODEL_PATH="/path/to/your/MOVA-360p-hf"
# 若不设置，需在下方 DEFAULT_MODEL_PATH 中填写你的模型目录。
DEFAULT_MODEL_PATH="${MOVA_MODEL_PATH:-}"
if [ -z "$DEFAULT_MODEL_PATH" ]; then
    echo "Error: MOVA_MODEL_PATH is not set. Please set it to your MOVA checkpoint directory (e.g. .../MOVA-360p-hf)." >&2
    exit 1
fi
MODEL_PATH="$DEFAULT_MODEL_PATH"

# 根据模型路径判断模型类型
if [[ "$MODEL_PATH" == *"360p"* ]]; then
    MODEL_KEY="mova-360p"
elif [[ "$MODEL_PATH" == *"720p"* ]]; then
    MODEL_KEY="mova-720p"
else
    MODEL_KEY="mova-360p"  # 默认
    echo "Warning: Could not determine model type from path, defaulting to mova-360p" >&2
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 切换到脚本目录，确保相对路径正确
cd "$SCRIPT_DIR"

# 统一输出目录：如果环境变量 SG_OUTPUT_DIR 未设置，使用相对路径 outputs/sglang_output
if [ -z "$SG_OUTPUT_DIR" ]; then
    # 使用相对路径，相对于脚本目录
    SG_OUTPUT_DIR="${SCRIPT_DIR}/outputs/sglang_output"
fi

# 确保目录存在
mkdir -p "$SG_OUTPUT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SGLang Server Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Model: ${MODEL_KEY}"
echo -e "  Port: ${PORT}"
echo -e "  GPUs: $((RING_DEGREE*ULYSSES_DEGREE*TP_SIZE))"
echo -e "  Output Directory: ${SG_OUTPUT_DIR}"
echo ""

# 启动服务器（后台运行）
echo -e "${GREEN}Starting SGLang server...${NC}"
sglang serve \
  --host 0.0.0.0 \
  --port ${PORT} \
  --model-path "${MODEL_PATH}" \
  --adjust_frames 'false' \
  --num-gpus $((RING_DEGREE*ULYSSES_DEGREE*TP_SIZE)) \
  --ring-degree ${RING_DEGREE} \
  --ulysses-degree ${ULYSSES_DEGREE} \
  --tp ${TP_SIZE} \
  --enable-torch-compile \
  --save-output \
  --output-path "$SG_OUTPUT_DIR" &

SERVER_PID=$!

# 等待服务器启动
echo -e "${YELLOW}Waiting for server to start...${NC}"
for i in {1..30}; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1 || \
       curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1 || \
       netstat -tuln 2>/dev/null | grep -q ":${PORT}" || \
       ss -tuln 2>/dev/null | grep -q ":${PORT}"; then
        echo -e "${GREEN}✓ Server is ready!${NC}"
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

# 显示访问信息
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Server Access Information:${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Local URL:"
echo -e "  ${GREEN}http://localhost:${PORT}${NC}"
echo -e "  ${GREEN}http://127.0.0.1:${PORT}${NC}"
echo ""
echo -e "${YELLOW}Server PID: ${SERVER_PID}${NC}"
echo -e "${YELLOW}To stop the server, run: kill ${SERVER_PID}${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"

# 保持脚本运行（等待服务器进程）
wait $SERVER_PID