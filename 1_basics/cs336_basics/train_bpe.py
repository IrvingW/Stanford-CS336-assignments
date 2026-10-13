import os
import regex as re
from collections import Counter
import multiprocessing
from typing import BinaryIO
import logging

logging.basicConfig(format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

Token = bytes
Pretoken = tuple[Token, ...]
Pair = tuple[Token, Token]
PretokenCounts = Counter[Pretoken]
PairCounts = Counter[tuple[Token, Token]]

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Create the initial byte-level vocabulary.

    The vocabulary must contain all 256 single-byte tokens and each unique
    special token. Token IDs should be deterministic so later merges can be
    appended consistently.
    """
    # 0~255 char
    vocab_dict = {i: bytes([i]) for i in range(256)}
    # special tokens start from the index of 256
    new_idx = 256
    existing = set(vocab_dict.values())
    for special in special_tokens:
        sp_bytes = special.encode('utf-8')
        if sp_bytes not in existing:
            vocab_dict[new_idx] = sp_bytes
            existing.add(sp_bytes)
            new_idx += 1
            
    return vocab_dict


def _pre_tokenize(
    input_path: str | os.PathLike,
    start: int,
    end: int | None,
    special_tokens: list[str],
) -> PretokenCounts:
    """Read the corpus and count its pre-tokenized byte sequences.

    Special tokens must act as hard boundaries and must not contribute to the
    pre-token counts. Ordinary text should be split with the GPT-2-style regex
    from the assignment, encoded as UTF-8, and represented as tuples of
    single-byte tokens.
    """
    # read the file
    with open(input_path, "rb") as f:
        if end is not None:
            f.seek(start)
            input_text = f.read(end - start).decode("utf-8")
        else:
            # no split, read all
            input_text = f.read().decode("utf-8")
    
    # split using special tokens like <EOF>
    if len(special_tokens) > 0:
        chunks = re.split('|'.join(map(re.escape,special_tokens)), input_text)
    else:
        chunks = [input_text]
    
    result :PretokenCounts = Counter()
    for chunk in chunks:
        for word in re.findall(GPT2_SPLIT_PATTERN, chunk):
            pre_token = tuple(bytes([b]) for b in word.encode("utf-8"))
            result[pre_token] += 1
            
    return result


def _count_pairs(pretoken_counts: PretokenCounts) -> PairCounts:
    """Count adjacent token pairs across all pre-tokens.

    Each pair's contribution must be weighted by the frequency of the
    pre-token that contains it. Pairs must never cross pre-token boundaries.
    """
    pair_counts = Counter()
    for pre_token, freq in pretoken_counts.items():
        for i in range(len(pre_token) - 1):
            pair_counts[pre_token[i], pre_token[i + 1 ]] += freq
            
    return pair_counts
        


def _select_best_pair(pair_counts: PairCounts) -> Pair | None:
    """Select the pair to merge next.

    Choose the pair with the highest frequency. When frequencies are tied,
    choose the lexicographically greatest pair, as required by the assignment.
    Return ``None`` when no mergeable pair remains.
    """
    # all pre-tokens are already single token
    if not pair_counts:
        return None
   
    # comment: Easy way to write 
    # max_freq = max(pair_counts.values())
    # candidates = [k for k, v in pair_counts.items() if v == max_freq]
    # select = max(candidates)
    # return select
    return max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _merge_pair(seq: Pretoken, pair: Pair) -> Pretoken:
    """Merge every non-overlapping occurrence of one pair in one pre-token.

    This helper operates on a single pre-token only. It should preserve the
    order of all unaffected tokens and return a new token tuple.
    """
    result = []
    new_token = pair[0] + pair[1]
    idx = 0
    while idx < len(seq):
        if idx < len(seq) - 1 and (seq[idx], seq[idx + 1]) == pair:
            result.append(new_token)
            idx += 2
        else:
            result.append(seq[idx])
            idx += 1
    return tuple(result)


def _apply_merge(pretoken_counts: PretokenCounts, merge_pair: Pair, pair_counts: PairCounts) -> tuple[PretokenCounts, PairCounts]:
    """Apply one selected merge to the complete pre-token frequency table.

    Preserve each pre-token's corpus frequency. If multiple old pre-tokens
    become the same token tuple after merging, combine their frequencies.
    """
    effected_pretoken = Counter()
    for pre_token, freq in pretoken_counts.items():
        if any((pre_token[i], pre_token[i+1]) == merge_pair for i in range(len(pre_token) - 1)):
            effected_pretoken[pre_token] = freq

    merged_pretoken = Counter()
    for seq, freq in effected_pretoken.items():
        del pretoken_counts[seq]
        for pair in [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]:
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
        new_seq = _merge_pair(seq, merge_pair)
        pretoken_counts[new_seq] += freq
        merged_pretoken[new_seq] += freq

    new_pair_counts = _count_pairs(merged_pretoken)            
    pair_counts.update(new_pair_counts)
    
    return pretoken_counts, pair_counts


def _parallel_pretokenize(
    input_path: str | os.PathLike,
    num_workers: int,
    special_tokens: list[str],
) -> PretokenCounts:
    if num_workers <= 1: 
        pre_token_counts: PretokenCounts = _pre_tokenize(input_path, 0, None, special_tokens)
    else:
        pre_token_counts: PretokenCounts = Counter()
        with open(input_path, "rb") as f:
            boundaries = _find_chunk_boundaries(f, num_workers, b"<|endoftext|>")
        logging.info(f"Pretokenize with {num_workers} processes.")
        with multiprocessing.Pool(num_workers) as pool:
            results = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                results.append(pool.apply_async(_pre_tokenize, args=(input_path, start, end, special_tokens)))
            for result in results:
                pre_token_counts.update(result.get())
   
    logging.info(f"Pretokenize completed. Total pre-tokens: {len(pre_token_counts)}")
    return pre_token_counts    

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return its vocabulary and merges.

    Orchestrate vocabulary initialization, pre-token counting, pair counting,
    deterministic pair selection, and repeated merging until ``vocab_size`` is
    reached or no mergeable pair remains.

    Args:
        input_path: Path to the text corpus used for BPE training.
        vocab_size: Maximum final vocabulary size, including byte tokens,
            special tokens, and tokens created by merges.
        special_tokens: Strings that are added to the vocabulary and treated
            as hard boundaries during training.

    Returns:
        A pair containing the token-ID-to-bytes vocabulary and the merge pairs
        in creation order.
    """
    assert vocab_size >= 256 + len(special_tokens)
   
    # initialize vocab dict 
    vocab_dict :dict[int, bytes] = _initialize_vocab(special_tokens=special_tokens)
    num_merges = vocab_size - len(vocab_dict)
    new_token_id = len(vocab_dict)
    
    # parallelize pre-tokenization
    pre_token_counts: PretokenCounts = _parallel_pretokenize(input_path=input_path, num_workers=num_workers, special_tokens=special_tokens)
   
    pair_counts: PairCounts = _count_pairs(pre_token_counts)
    logging.info(f"Count pairs completed. Total pairs: {len(pair_counts)}")
   
    logging.info(f"Start training BPE with maximum {num_merges} merges.")
    merges = []
    while len(merges) < num_merges:
       select_pair: Pair|None = _select_best_pair(pair_counts)
       if not select_pair:
           # could not do merge any more
           break
       vocab_dict[new_token_id] = select_pair[0] + select_pair[1]
       merges.append(select_pair)
       new_token_id += 1
       # update counts map
       pre_token_counts, pair_counts = _apply_merge(pretoken_counts=pre_token_counts, merge_pair=select_pair, pair_counts=pair_counts)
       
       if len(merges) % 1000 == 0:
           logging.info(f"Merged {len(merges)} pairs, total pre-token count: {len(pre_token_counts)}, total pairs count: {len(pair_counts)}")
       
       
    return vocab_dict, merges
