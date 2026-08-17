#!/bin/bash
# Deploy fde-rag stack to spark2 via rsync + docker compose
set -euo pipefail

source ~/.fde-cluster.env
SPARK2=${FDE_SPARK2_IP:-192.168.100.11}
REMOTE_DIR=/opt/fde/fde-rag-stack
SECRETS=/home/spark-acn/.fde-secrets.env  # needs NEO4J_PASSWORD and SPARK_VLLM_API_KEY

if [[ ! -f "$SECRETS" ]]; then
  echo "ERROR: $SECRETS not found. Create it with:"
  echo "  NEO4J_PASSWORD=<password>"
  echo "  SPARK_VLLM_API_KEY=local-dummy"
  exit 1
fi

echo "==> Syncing stack to spark2:$REMOTE_DIR"
ssh spark-acn@"$SPARK2" "mkdir -p $REMOTE_DIR"
rsync -av --delete \
  --exclude='rag/go.sum' \
  "$(dirname "$0")/" \
  "spark-acn@${SPARK2}:${REMOTE_DIR}/"

echo "==> Copying secrets"
scp "$SECRETS" "spark-acn@${SPARK2}:${REMOTE_DIR}/.env"

echo "==> Building and starting stack on spark2"
ssh spark-acn@"$SPARK2" "
  cd $REMOTE_DIR
  docker compose build --pull
  docker compose up -d --remove-orphans
  docker compose ps
"

echo "==> Done. MCP endpoint: http://${SPARK2}:7900/mcp"
