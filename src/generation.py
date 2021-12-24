from tokenizers import Tokenizer
from tqdm.auto import tqdm
import torch


class GenerationWrapper:
    
    def generate(self, prompt: str, max_length: int) -> str:
        raise NotImplementedError()
        
        
class GreedyGenerationWrapper(GenerationWrapper):
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        tokenizer: Tokenizer, 
        eos_token_id: int, 
        verbose: bool = False, 
        return_special_tokens: bool = False
    ):
        super(GenerationWrapper, self).__init__()
        self.model = model
        self.model = self.model.eval()
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.verbose = verbose
        self.return_special_tokens = return_special_tokens
        
    def generate(self, prompt: str, max_length: int) -> str:
        ids = self.tokenizer.encode(prompt).ids
        with torch.no_grad():
            for _ in tqdm(range(max_length), desc="generation", disable=(not self.verbose)):
                logits = self.model(input_ids=torch.tensor(ids, dtype=torch.long).unsqueeze(0))
                logits = logits[0, -1] # take last token logits
                probs  = torch.softmax(logits, -1)
                idx    = torch.argmax(probs, -1).item()
                ids += [idx]
                if idx == self.eos_token_id:
                    break
                    
        result = self.tokenizer.decode(ids, skip_special_tokens=(not self.return_special_tokens))
        return result
