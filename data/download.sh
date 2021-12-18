#!/usr/bin/env bash
set -euo pipefail
echo "=> Downloading train.txt..."
aria2c -x4 -s4 -j4 https://deepnlp-ru-qa-bot.s3.eu-central-1.amazonaws.com/train.txt.zst
echo "=> Decompressing..."
zstd -vv --rm -d train.txt.zst
echo "=> Downloading qa_data.jsonl..."
aria2c -x4 -s4 -j4 https://deepnlp-ru-qa-bot.s3.eu-central-1.amazonaws.com/qa_data.jsonl.zst
echo "=> Decompressing..."
zstd -vv --rm -d qa_data.jsonl.zst
