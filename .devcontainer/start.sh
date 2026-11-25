#!/bin/bash
# Codespaces 每次启动容器时跑：先把服务拉起来，端口转发后浏览器会自动打开。
# 已经在跑就不重复启动（重复启动只会抢端口、报错刷屏）。
set -e

cd "$(dirname "$0")/.."

if curl -s -m 2 http://127.0.0.1:8770/api/health >/dev/null 2>&1; then
  echo "[preproc] 服务已在运行"
  exit 0
fi

echo "[preproc] 启动预处理工作台..."
nohup python run.py --host 0.0.0.0 --port 8770 > /tmp/preproc.log 2>&1 &
sleep 2
cat /tmp/preproc.log || true
