import regex as re

from collections import Counter
from multiprocessing import Pool
from typing import Iterator, Iterable\

from .pretokenization_example import find_chunk_boundaries


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_by_special_token_and_pre_tokenize(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[bytes, int]:
    special_token_patterns = '|'.join(re.escape(t) for t in special_tokens)
    counts = Counter()

    with open(input_path, 'rb') as f:
        f.seek(start)
        current_chunk = f.read(end - start).decode("utf-8", errors="ignore")
        spans = re.split(special_token_patterns, current_chunk)
        for span in spans:
            if not span:
                continue
            for match in re.finditer(PAT, span):
                token = match.group(0)
                counts[token.encode('utf-8')] += 1
    return counts

def merge_pair_in_tuple_token(old_token: tuple[bytes], pair: tuple[bytes]) -> tuple[bytes]:
    new_token_parts = []
    i = 0
    while i < len(old_token):
        if i < len(old_token) - 1 and (old_token[i], old_token[i + 1]) == pair:
            new_token = old_token[i] + old_token[i + 1]
            new_token_parts.append(new_token)
            i += 2
        else:
            new_token_parts.append(old_token[i])
            i += 1
    return tuple(new_token_parts)

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str] = None,
        num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    chunk_separator = '<|endoftext|>'

    if special_tokens is None:
        special_tokens = [chunk_separator]
    if chunk_separator not in special_tokens:
        special_tokens.append(chunk_separator)

    vocab = {i: bytes([i]) for i in range(256)}
    cur_id = len(vocab)
    merges = []

    for special_token in special_tokens:
        vocab[cur_id] = special_token.encode('utf-8')
        cur_id += 1

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, chunk_separator.encode('utf-8'))

    process_args = [(input_path, start, end, special_tokens)
                    for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(num_processes) as pool:
        chunk_counts = pool.starmap(split_by_special_token_and_pre_tokenize, process_args)
    total_counts = Counter()
    for chunk_counter in chunk_counts:
        total_counts.update(chunk_counter)

    pair_counts = Counter()
    word_freqs = Counter()
    for byte_token, count in total_counts.items():
        word_tuple = tuple(byte_token[i: i + 1] for i in range(len(byte_token)))
        word_freqs[word_tuple] = count
    for word, count in word_freqs.items():
        for i in range(len(word) - 1):
            cur_pair = (word[i], word[i + 1])
            pair_counts[cur_pair] += count

    while len(vocab) < vocab_size and pair_counts:
        top_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(top_pair)
        vocab[cur_id] = top_pair[0] + top_pair[1]
        cur_id += 1

        for word, freq in list(word_freqs.items()):
            for i in range(len(word) - 1):
                cur_pair = (word[i], word[i + 1])
                pair_counts[cur_pair] -= freq
                if pair_counts[cur_pair] == 0:
                    del pair_counts[cur_pair]
            new_word = merge_pair_in_tuple_token(word, top_pair)
            word_freqs[word] -= freq
            if word_freqs[word] == 0:
                del word_freqs[word]
            word_freqs[new_word] += freq
            for i in range(len(new_word) - 1):
                cur_pair = (new_word[i], new_word[i + 1])
                pair_counts[cur_pair] += freq

    return vocab, merges

class Tokenizer():
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        raise NotImplementedError()

    def encode(self, text: str) -> list[int]:
        ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ...

    def decode(self, ids: list[int]) -> str:
        ...


if __name__ == '__main__':
    print(train_bpe('../tests/fixtures/tiny_text.txt', 263))
