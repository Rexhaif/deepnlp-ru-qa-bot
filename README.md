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
* explore generative capabilities!
```bash
❯ python interactive_cli.py --config="./config.toml"
Loading pretrained model...
<All keys matched successfully>
Model loaded
Welcome at interactive Q&A cli for generative model 🦾🦾
🧑‍💻 Type your questions(and categories, if you know) below👇 or type $EXIT to exit
Type your question: вредны ли вареничьки?
Type category (like 'Красота и Здоровье') or press Enter if you don't know: 
generation:   9%|████████▉                                                                                      | 6/64 [00:00<00:01, 44.74it/s]
🔮 Answer:  да, это не очень. 
Type your question: а что не вредно?
Type category (like 'Красота и Здоровье') or press Enter if you don't know: 
generation:   9%|████████▉                                                                                      | 6/64 [00:00<00:01, 42.70it/s]
🔮 Answer:  не вредно, но полезно. 
Type your question$EXIT 
Have a good day!😀
```
## Pretrained models
* generative transformer, hidden size = 512, num layers = 12, num heads = 8, max length = 128, trained for 25k steps with 256 batch size
  * [model.pth](https://www.icloud.com/iclouddrive/0051Wm21rx9_j7RnkpnBWhDYg#model) - should be placed in data/model.pth
  * [tokenizer.json](https://www.icloud.com/iclouddrive/09fbU5nBp_wlS6kWwpzhdLeWQ#tokenizer) - should be placed in data/tokenizer.json 
