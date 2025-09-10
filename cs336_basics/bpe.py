import regex as re

from collections import Counter
from multiprocessing import Pool
from typing import Iterator, Iterable\

from .pretokenization_example import find_chunk_boundaries


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def process_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
    keep_special_tokens: bool = False,
) -> list[bytes]:
    with open(input_path, 'rb') as f:
        f.seek(start)
        current_chunk = f.read(end - start).decode("utf-8", errors="ignore")

    spans = split_chunk_by_special_tokens(current_chunk, special_tokens)
    return pre_tokenize(spans, special_tokens, keep_special_tokens)

def split_chunk_by_special_tokens(current_chunk: str, special_tokens: list[str]) -> list[str]:
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_token_patterns = '(' + '|'.join(re.escape(t) for t in special_tokens) + ')'
    spans = re.split(special_token_patterns, current_chunk)
    return spans

def pre_tokenize(spans: list[str], special_tokens: list[str], keep_special_tokens: bool = False) -> list[bytes]:
    pre_tokens = []
    for span in spans:
        if not span:
            continue
        if span in special_tokens:
            if keep_special_tokens:
                pre_tokens.append(span.encode("utf-8"))
        else:
            for match in re.finditer(PAT, span):
                token = match.group(0)
                pre_tokens.append(token.encode("utf-8"))
    return pre_tokens

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
        pre_tokenized_chunks = pool.starmap(process_chunk, process_args)
    total_counts = Counter()
    for chunk_pre_tokens in pre_tokenized_chunks:
        total_counts.update(Counter(chunk_pre_tokens))

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
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = {token_pair: i for i, token_pair in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens else ['<|endoftext|>']
        self.special_tokens_set = set([st.encode('utf-8') for st in self.special_tokens])

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        raise NotImplementedError()

    def encode(self, text: str) -> list[int]:
        spans = split_chunk_by_special_tokens(text, self.special_tokens)
        pre_tokens = pre_tokenize(spans, self.special_tokens, keep_special_tokens=True)
        token_ids = []
        for pre_token in pre_tokens:
            if pre_token in self.special_tokens_set:
                token_ids.append(self.reverse_vocab[pre_token])
                continue
            cur_tokens = [pre_token[i: i + 1] for i in range(len(pre_token))]

            while True:
                best_pair_info = {'rank': float('inf'), 'pair': None, 'idx': -1}

                for i in range(len(cur_tokens) - 1):
                    pair = (cur_tokens[i], cur_tokens[i + 1])
                    rank = self.merges_dict.get(pair, None)
                    if rank is not None and rank < best_pair_info['rank']:
                        best_pair_info['rank'] = rank
                        best_pair_info['pair'] = pair
                        best_pair_info['idx'] = i

                if best_pair_info['pair'] is None:
                    break

                best_idx = best_pair_info['idx']
                best_pair = best_pair_info['pair']
                cur_tokens = cur_tokens[:best_idx] + [best_pair[0] + best_pair[1]] + cur_tokens[best_idx + 2:]

            for token in cur_tokens:
                token_ids.append(self.reverse_vocab[token])
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        boundary_token = '<|endoftext|>'
        buffer = ''

        for chunk in iterable:
            buffered_chunk = buffer + chunk
            parts = buffered_chunk.split(boundary_token)

            for i, part in enumerate(parts[:-1]):
                text_to_encode = part + boundary_token
                yield from self.encode(text_to_encode)

            buffer = parts[-1]

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])

        all_bytes = b''.join(byte_chunks)
        return all_bytes.decode('utf-8', errors='replace')


if __name__ == '__main__':
    print(train_bpe('../tests/fixtures/tiny_text.txt', 263))
