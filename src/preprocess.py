import argparse as ap
import json
import logging

import pandas as pd
from rich.logging import RichHandler
from rich.progress import track

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)


def preprocess_for_seq2seq(args: ap.Namespace):
    data_folder: str = args.data_folder
    output: str = args.output

    log.info("Reading qa_data.jsonl..")
    with open(f"{data_folder}/qa_data.jsonl", "r", encoding="utf-8") as f:
        qa_data = f.readlines()
    log.info(f"Read {len(qa_data)} lines")

    with open(f"{data_folder}/{output}", "w", encoding="utf-8") as f:
        # Stage 1: read jsonl objects
        log.info("Stage 1: generating training data from qa_data.jsonl objects")
        st1_counter: int = 0
        for line in track(qa_data, description="Running through qa_data.jsonl"):
            obj = json.loads(line)
            question: str = obj["question"]
            category: str = obj["category"]
            for response in obj["responses"]:
                st1_counter += 1
                out_line: str = f"{question}\t{category}\t{response}"
                f.write(f"{out_line}\n")
        log.info(f"Generated {st1_counter} examples")

        # Stage 2: read .txt file
        log.info("Stage 2: concatenating with training examples from train.txt")
        with open(f"{data_folder}/train.txt", "r", encoding="utf-8") as train_file:
            st2_counter: int = 0
            lines = train_file.readlines()
            for line in track(lines, description="Running through train.txt"):
                question, response = line.split("\t")
                st2_counter += 1
                f.write(f"{question}\t{response}")

        log.info(f"Concatenated with {st2_counter} examples")
        log.info(f"Total examples count = {st1_counter + st2_counter}")

    log.info(f"Written to {data_folder}/{output}")


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="preprocess.py")
    parser.add_argument("--data-folder", type=str, help="Path to data folder")
    parser.add_argument(
        "--output",
        type=str,
        default="seq2seq_output.txt",
        help="Name of output file within data fodler",
    )
    args: ap.Namespace = parser.parse_args()
    log.info(f"Executing preprocess.py with arguments: {args}")
    preprocess_for_seq2seq(args)
    log.info("Done")
