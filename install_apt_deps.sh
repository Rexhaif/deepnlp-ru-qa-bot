#!/usr/bin/env sh
echo "=> Installing dev packages from apt..."
apt -q --yes update && apt install --yes -q aria2 zstd
