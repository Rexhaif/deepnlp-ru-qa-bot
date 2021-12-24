import argparse as ap
from rich import print
from tokenizers import Tokenizer
from model import GenerativeTransformer
from generation import GreedyGenerationWrapper, GenerationWrapper
import toml
import torch

def run_interactive(gw: GenerationWrapper):
    print("Welcome at interactive Q&A cli for generative model 🦾🦾")
    print("🧑‍💻 Type your questions(and categories, if you know) below👇 or type $EXIT to exit")
    while True:
        x = input("Type your question: ")
        if x == "$EXIT":
            break
        else:
            cat = input("Type category (like 'Красота и Здоровье') or press Enter if you don't know: ")
            if len(cat) != 0:
                input_str = f"<START> {x} <CAT> {cat} <RESP>"
            else:
                input_str = f"<START> {x} <RESP>"
                
            response = gw.generate(input_str, 64)
            response = response.split("<RESP>")[-1].split("<END>")[0]
            print(f"🔮 Answer: {response}")
            

if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="interactive_cli.py")
    parser.add_argument("--config", type=str, required=True, help="Path to config.toml")
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = toml.load(f)
        
    config = ap.Namespace(**config)
    
    tokenizer = Tokenizer.from_file(config.tokenizer)
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
    print("Loading pretrained model...")
    state_dict = torch.load(config.model_save_path, map_location="cpu")
    print(model.load_state_dict(state_dict))
    print("Model loaded")
    
    eos_token_id = tokenizer.encode("<END>").ids[0]
    wrapper = GreedyGenerationWrapper(model, tokenizer, eos_token_id, True, True)
    
    run_interactive(wrapper)
    print(f"Have a good day!😀")