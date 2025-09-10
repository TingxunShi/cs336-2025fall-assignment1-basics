## BPE Tokenizer

### Understanding Unicode

1. `chr(0)` returns `\x00` (`'\0'` in C).
2. If we invoke `__repr__()` method, it will escape all the potential characters. For example, `chr(0).__repr__()` shows `'\x00'`.
3. In the print version the character will not be escaped, so `chr(0)` will not be displayed.

### Unicode Encodings

1. UTF-8 encoding is widely used and is the most concise. Take the example given in the note "hello! こんにちは!", if we encode it using UTF-8, its length is only 23. However, its utf-16 encoding string has 28 characters, and utf-32 counterpart has 56 characters.
2. This program fails when the input contains non-ascii characters, like '你好'. It breaks the unicode series into pieces and decode them separately, leading to `UnicodeDecodeError`
3. `b'\xC0\xAF'`. This is an "overlong encoding." The byte `\xC0` indicates the start of a 2-byte sequence, but the sequence `\xC0\xAF` would decode to the character / (U+002F), which already has a shorter, canonical 1-byte representation (`b'\x2F'`). The UTF-8 standard explicitly forbids overlong encodings to ensure that every character has a unique byte representation.

