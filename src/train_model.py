from model import GenerativeTransformer, CrossEntropyForLM
from data import QAGenDataset
from tokenizers import Tokenizer
import numpy as np
import wandb
import toml

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import accelerate as ac
import argparse as ap
from tqdm.auto import tqdm

import torch
from torch import optim
from torch import nn
import transformers as tr

import math
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

optim_map = {
    'Adam': optim.Adam,
    'AdamW': tr.AdamW,
    'RAdam': optim.RAdam
}

lr_sched_map = {
    'cosine': tr.get_cosine_schedule_with_warmup,
    'linear': tr.get_linear_schedule_with_warmup,
    'const-warmup': tr.get_constant_schedule_with_warmup,
    'const': tr.get_constant_schedule
}

def load_data(config: ap.Namespace, verbose=True):
    if verbose:
        logger.info(f"Loading training data from {config.data_file}")
    with open(config.data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    train, valid = train_test_split(lines, test_size=config.test_size, shuffle=True, random_state=config.seed)
    if verbose:
        logger.info(f"Read {len(lines)} examples => {len(train)} train, {len(valid)} valid")
    
    tokenizer = Tokenizer.from_file(config.tokenizer)
    train_ds = QAGenDataset(train, tokenizer, max_length=config.max_length)
    valid_ds = QAGenDataset(valid, tokenizer, max_length=config.max_length)
    
    return train_ds, valid_ds, tokenizer


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog='train_model.py')
    parser.add_argument("-c", "--config", type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = toml.load(f)
        config = ap.Namespace(**config)
    
    accelerator = ac.Accelerator()
    if accelerator.is_local_main_process:
        logger.info(f"Started training with config: {config}")
    
    train_ds, valid_ds, tokenizer = load_data(config, verbose=accelerator.is_local_main_process)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.device_batch_size, num_workers=config.dataloader_num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=config.device_batch_size, num_workers=config.dataloader_num_workers)
    
    # setting model/optimizer
    model = GenerativeTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        max_length=config.max_length,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        embd_pd=config.embd_pd,
        attn_pd=config.attn_pd,
        resid_pd=config.resid_pd
    )
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = optim_map[config.optimizer]
    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )
    
    if accelerator.is_local_main_process:
        num_params = sum(x.numel() for x in model.parameters())
        logger.info(f"Training model with {(num_params / 1e6):.2f}M parameters")
    
    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )
    
    # setting lr scheduler
    num_warmup_steps = round(config.n_training_steps * config.warmup_ratio)
    lr_sched_args = {
        'optimizer': optimizer
    }
    if config.lr_scheduler != 'const':
        lr_sched_args['num_warmup_steps'] = num_warmup_steps

    if config.lr_scheduler in {'cosine', 'linear'}:
        lr_sched_args['num_training_steps'] = config.n_training_steps

    scheduler = lr_sched_map[config.lr_scheduler](**lr_sched_args)
    
    num_train_epochs = config.n_training_steps // config.eval_frequency
    total_batch_size = config.device_batch_size * config.gradient_accumulation_steps * accelerator.num_processes

    if accelerator.is_local_main_process:
        logger.info(f"Num technical epochs: {num_train_epochs}")
        logger.info(f"Total Batch Size: {total_batch_size}")
        logger.info(f"Starting training")
        wandb_logger = wandb.init(
            project="deepnlp-ru-qa-bot",
            entity="dslarionov",
            job_type='train',
            config=config.__dict__
        )
        wandb.watch(model)
    else:
        wandb_logger = None
        
    progress_bar = tqdm(range(config.n_training_steps), desc="Training", disable=(not accelerator.is_local_main_process))
    
    loss_fn = CrossEntropyForLM()
    step_i = 0
    accelerator.wait_for_everyone()
    for epoch in range(num_train_epochs):
        model.train()
        for i, batch in enumerate(train_dl):
            input_ids, mask = batch
            labels = input_ids.clone()
            labels = torch.where(mask, input_ids, torch.tensor(-100, dtype=torch.long, device=accelerator.device))
            outputs = model(input_ids, mask)
            loss = loss_fn(outputs, labels)
            loss /= config.gradient_accumulation_steps
            accelerator.backward(loss)
            if (i % config.gradient_accumulation_steps == 0) or (i == len(train_dl) - 1):
                accelerator.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                if not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                    
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': loss.item()
                })
                step_i += 1
                if accelerator.is_local_main_process:
                    wandb_logger.log({'train/loss': loss.item()}, step=step_i)

            if step_i >= config.n_training_steps or (step_i % config.eval_frequency == 0):
                break # go to validation

        model.eval()
        losses = []
        for step, batch in tqdm(
            enumerate(valid_dl),
            desc='Validation',
            total=len(valid_dl),
            disable=(not accelerator.is_local_main_process)
        ):
            with torch.no_grad():
                input_ids, mask = batch
                labels = input_ids.clone()
                labels = torch.where(mask, input_ids, torch.tensor(-100, dtype=torch.long, device=accelerator.device))
                outputs = model(input_ids, mask)
                loss = loss_fn(accelerator.gather(outputs), accelerator.gather(labels))
                losses.append(loss.item())
        try:
            perplexity = math.exp(np.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        
        if accelerator.is_local_main_process:
            wandb_logger.log({'eval/ppl': perplexity, 'eval/loss': np.mean(losses)}, step=step_i)
            logger.info(f"Epoch: {epoch}, Perplexity: {perplexity:.4f}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), config.model_save_path)
            wandb.save(config.model_save_path)
            logger.info(f"Saved model")
            
    if accelerator.is_local_main_process:
        logger.info(f"Training finished")
    

        
    
    