# deepnlp-ru-qa-bot
QA-bot project for deepnlp course

## Features:
* colorful cli logging via `rich`
* re-implemented from scratch gpt2-like transformer model
* prompt-based question-answer generation
* multi-gpu/distributed/tpu-based training via huggingface/accelerate
* gradient accumulation
* fp16 training
* WandB logging [https://wandb.ai/dslarionov/deepnlp-ru-qa-bot/overview?workspace=user-dslarionov](https://wandb.ai/dslarionov/deepnlp-ru-qa-bot/overview?workspace=user-dslarionov)

## Reproduction instructions
* install apt requirements via `install_apt_deps.sh`
* download data into ./data dir via `download.sh`
* preprocess dataset via `python preprocess.py --data-folder ../data/ --output output.txt`
* train huggingface wordpiece tokenizer `python train_tokenizer.py --input ../data/output.txt --output ../data/tokenizer.json --vocab-size=60000 --min-frequency=2`
* configure huggingface/accelerate `accelerate config`, then test it `accelerate test`
* lauch training `accelerate launch train_model.py --config=./config.toml`
