import argparse as ap
import logging

from rich.logging import RichHandler
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.decoders import WordPiece as WordPieceDecoder

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


def train_tokenizer(args: ap.Namespace):
    input_file: str = args.input
    output: str = args.output
    vocab_size: int = args.vocab_size
    min_frequency: int = args.min_frequency

    tokenizer = Tokenizer(WordPiece(unk_token="<UNK>"))
    trainer = WordPieceTrainer(
        show_progress=True,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        continuing_subword_prefix="##",
        special_tokens=["<PAD>", "<UNK>", "<START>", "<END>", "<CAT>", "<RESP>"],
    )
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.decoder = WordPieceDecoder()


    logger.info("Begin train tokenzier")
    tokenizer.train([input_file], trainer=trainer)
    logger.info(f"Training done, saving to {output}")
    tokenizer.save(output, pretty=True)


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="train_tokenizer.py")
    parser.add_argument("--input", type=str, help="Path to .txt filw with texts")
    parser.add_argument("--output", type=str, help="Output path for tokenizer file")
    parser.add_argument(
        "--vocab-size", type=int, default=30000, help="Size of tokenzier vocab"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=0,
        help="Minimal numer of token occurencies",
    )
    args: ap.Namespace = parser.parse_args()

    logger.info(f"Called train_tokenizer.py with arguments: {args}")
    train_tokenizer(args)
