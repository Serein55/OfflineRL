#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor
fetch() {
  local name="$1" url="$2" revision="$3" patch="${4:-}"
  if [ ! -d "vendor/$name/.git" ]; then
    git clone --filter=blob:none --no-checkout "$url" "vendor/$name"
    git -C "vendor/$name" checkout "$revision"
  fi
  test "$(git -C "vendor/$name" rev-parse HEAD)" = "$revision"
  if [ -n "$patch" ]; then
    if git -C "vendor/$name" apply --reverse --check "../../$patch" 2>/dev/null; then
      return
    fi
    git -C "vendor/$name" apply --check "../../$patch"
    git -C "vendor/$name" apply "../../$patch"
  fi
}
fetch lerobot https://github.com/huggingface/lerobot.git ce3b9f627e55223d6d1c449d348c6b351b35d082 artifacts/lerobot.patch
fetch LIBERO https://github.com/Lifelong-Robot-Learning/LIBERO.git 8f1084e3132a39270c3a13ebe37270a43ece2a01 artifacts/libero.patch
fetch reinboT https://github.com/COST-97/reinboT.git 81ecdd0e6662e0070dcb55cc7001f4170f2a43e6
