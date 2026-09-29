#!/usr/bin/env bash
# 阶段0：数据落盘 + 环境自检（Java/database 已由前置步骤就位）
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/data
uv run python - <<'EOF'
from oak.config import load_config
from oak.data.queries import load_queries, partition_train, stratified_test_subset
cfg = load_config()
for split in ("train", "validation"):
    qs = load_queries(split, cfg)
    print(split, len(qs), "queries")
groups = partition_train(cfg.train_per_round, cfg.rounds, cfg.seed, cfg)
test = stratified_test_subset(cfg.test_size, cfg.seed, cfg)
print("round groups:", groups)
print("test subset:", test[:10], "... total", len(test))
EOF
echo "setup_env done"
