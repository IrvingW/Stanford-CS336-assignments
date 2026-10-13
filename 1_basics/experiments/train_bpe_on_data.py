# /usr/bin/env python3

from cs336_basics.train_bpe import train_bpe
import argparse
import pickle
from pathlib import Path
import logging

logging.basicConfig(format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def main(data_path: str, vocab_size: int, num_workers: int, output_dir: str, special_tokens: list[str] | None):
    logging.info(f"Training BPE on {data_path} with vocab_size={vocab_size}, num_workers={num_workers}, special_tokens={special_tokens}")
    
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens or [],
        num_workers=num_workers,
    )

    logging.info(f"Training BPE completed. Vocabulary size: {len(vocab)}")
    logging.info(f"Number of merges: {len(merges)}")

    # save vocab and merges to the directory of this script located
    logging.info(f"Prepare to save vocab and merges to {Path.absolute(Path(output_dir))}")

    vocab_file = Path(output_dir) / "vocab.pkl"
    merges_file = Path(output_dir) / "merges.pkl"
    with vocab_file.open("wb") as f:
        pickle.dump(vocab, f)
    with merges_file.open("wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=False, default=10000)
    parser.add_argument("--num_workers", type=int, required=False, default=4)
    parser.add_argument("--special_tokens", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, required=False, default=".")
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        vocab_size=args.vocab_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        special_tokens=args.special_tokens,
    )
