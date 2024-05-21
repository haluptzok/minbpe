"""
todo list:
0> Why does tokenization help - what is the logprob of the tokenized text compared to original text just using bigrams?
1> Understand the code
2> Understand the test code - how to invoke it
3> Add test.py to test the code
4> Add test case to test.py to the count of tokens for egg, Egg, EGG example from Karpathy on all tokenizers
5> Make capital_space_out.py work with no changes - just new names
6> Add test capital_space_out.py to test the code and test.py
6> Make tokenization create tuples of (token, capitalization, space) instead of just token - check it works.
6> Make space and capitalization indepenent of each other - both or neither can be invoked.
7> Make capital_space_out.py work with new token indexes, 1M * capital_index
8> Make capital_space_out.py work with new token indexes, 10M * space_index
9> Merge tokens preserving the Capitalization and Space information

Tokenization papers to compare with:
https://medium.com/@bradneysmith/tokenization-llms-from-scratch-1-cedc9f72de4e
https://arxiv.org/html/2404.17808v1
https://arxiv.org/abs/2402.18376

https://tiktokenizer.vercel.app/?model=gpt-4o
<|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|im_start|>user<|im_sep|>egg egg egg
Egg Egg Egg
EGG EGG EGG
<|im_end|><|im_start|>assistant<|im_sep|>
200264, 17360, 200266, 3575, 553, 261, 10297, 29186, 200265, 200264, 1428, 200266, 72126, 16102, 16102, 198, 109379, 52711, 52711, 198, 21389, 38, 457, 43469, 457, 43469, 198, 200265, 200264, 173781, 200266
egg egg egg
Egg Egg Egg
EGG EGG EGG
72126, 16102, 16102, 198, 109379, 52711, 52711, 198, 21389, 38, 457, 43469, 457, 43469, 198

https://tiktokenizer.vercel.app/?model=meta-llama%2FMeta-Llama-3-8B
<|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|im_start|>user<|im_sep|>egg egg egg
Egg Egg Egg
EGG EGG EGG
<|im_end|><|im_start|>assistant<|im_sep|>
27, 91, 318, 5011, 91, 29, 9125, 27, 91, 318, 55875, 91, 29, 2675, 527, 264, 11190, 18328, 27, 91, 318, 6345, 91, 1822, 91, 318, 5011, 91, 29, 882, 27, 91, 318, 55875, 91, 29, 29468, 19151, 19151, 198, 36, 14736, 42313, 42313, 198, 9560, 38, 469, 23050, 469, 23050, 198, 27, 91, 318, 6345, 91, 1822, 91, 318, 5011, 91, 29, 78191, 27, 91, 318, 55875, 91, 29
egg egg egg
Egg Egg Egg
EGG EGG EGG
29468, 19151, 19151, 198, 36, 14736, 42313, 42313, 198, 9560, 38, 469, 23050, 469, 23050, 198

https://tiktokenizer.vercel.app/?model=google%2Fgemma-7b
egg egg egg
Egg Egg Egg
EGG EGG EGG
2, 53892, 13514, 13514, 108, 55834, 36223, 36223, 108, 235291, 14798, 189148, 189148, 108

Minimal (byte-level) Byte Pair Encoding tokenizer that strips out capitalization and spaces.

Text can be represented as a single stream of lowercase and upper case characters mixed.
Or text can be represted as two streams, one stream of all charcters mapped to lower-case, and another stream specfying if the character was originally upper-case.
The allows egg/Egg/EGG to be represented as the same token, sharing the same embedding.
The extra stream contains tokens that represent 3 states: 
    +0 million for all lower-case
    +1 million for all all-caps (including a single upper cased letter)
    +2 million for capitalized meaning just the first letter upper-case
    +3 million is a mish-mash of upper and lower case - we can only store 1 type of mish-mash in the decode table, and skip the other types of mish-mash.

The transformer using this tokenizer needs to input and output the extra stream of capitalization tokens.
On the input side there is an embedding for the 3 states which is added in just like the positional encoding.
On the output side the extra stream of tokens requires a separate softmax layer to predict the 3 states of the output token.
This potentially reduces the number of tokens in the vocabulary by a factor of 3, for the price of 3 extra embeddings and a separate softmax layer.
So for the same sized embedding layer, the model can have a much larger vocabulary effectively, and have better generalization to less common words.
For example uncommon EGG and common egg would share the same embedding, and the model would learn to predict the correct capitalization based on the context.
When merging 2 tokens, if both are lower-case, the result is lower-case.
When merging 2 tokens, if both are all-caps, the result is all-caps.
When merging 2 tokens, if the left one is capitalized (or all caps single letter) and the right one is lower-case, the result is capitalized.
When merging 2 tokens, if the nice case doesn't occur, the result is mish-mash, we store the 1 most common type of mish-mash in the table, and skip the other types of mish-mash.
For mish-mash when decoding, the decoding table will specify the left and right token types.
So when merging tokens - the all-lower, all-upper, and capitalized tokens are merged as expected and specified by the extra stream of tokens.
But if the merging tokens aren't of one of those types - we pick the most common type of all the tokens we could merge, and put their types in the decoding table.
And we assign the merged tokens "mish-mash" type, and all the other matches of different mish-mash don't get merged and are skipped.
In this way, the entry in the table is:
i phone -> iphone + lower tag
I PHONE -> iphone + upper tag
I phone -> iphone + cap tag
i Phone -> iphone + mish-mash tag, specify in the decode table (lower tag, cap tag), taking most common type of all the possible mish-mash merges.
i pHone -> skipped in merging, we can only store in the table 1 type of mish-mash.  This could get merged in a later round of merging.
Also could only create the mish-mash when it's unique, as an option, the mish-mash could be skipped, and the tokens not merged, unless it's unique.
Because the mish-mash embedding may not be adequate to represent all the possible meanings for different combinations of capitalization.

Failure case - you need iPhone != iphone != iPHONE != Iphone

Text can be represented as a single stream of characters with spaces interspersed.
Or text can be represented as two streams, one stream of all characters, and another stream specifying if a space occurs before or after character.
This allows 'egg'/' egg'/'egg '/' egg ' to be represented as the same token, sharing the same embedding.
The extra stream contains tokens that represent 3 states:
    0 for no space before or after
    1 for space before
    2 for space after
    3 for space before and after
    ... one could imagine even more states for multiple spaces, tabs, returns, punctuation before or after.
The transformer using this tokenizer needs to input and output the extra stream of space tokens.
On the input side there is an embedding for the 4 states which is added in just like the positional encoding.
On the output side the extra stream of tokens requires a separate softmax layer to predict the 4 states of the space output token.
This potentially reduces the number of tokens in the vocabulary by a factor of 2, for the price of 4 extra embeddings and a separate softmax layer.
So for the same sized embedding layer, the model can have a much larger vocabulary effectively, and have better generalization to less common words.
For example uncommon 'egg' and common ' egg' would share the same embedding, and the model would learn to predict the correct capitalization based on the context.
Decoding space tokens is more complex than capitalization tokens, because the space token can be before or after the character, or both.

There are a few different cases to consider, we need to test round trip all the following cases:
# no space
'a'
'ab'
'abc'
# single space before, after, both
' a'
'a '
' a '
'a a'
' a a'
'a a '
' a a '
# double space before, after, both
'a  a'
'  a  a'
'a  a  '
'  a  a  '
'a  a  a'
# triple space before, after, both
# quadruple space before, after, both
# quintuple space before, after, both
# single, double, triple quadruple quintiple a aa aaa aaaa aaaaa
# with all variations of capitalization
# generate all possible combinations of space and capitalization, to not miss any edge cases

If there are 2 or more spaces between tokens, we eat the first left-most space.
When decoding if there is a token indicating add one to the left, we only do it if there is no space to the left of the token already

This representation is more complex, but it is more efficient than standard BPE tokenization.
It's benefits would be most pronounced in Wester Languages that use capitalization and spaces, but those are significant markets worth targeting.
The goal is better embeddings of the tokens by folding all the variations of capitalization and spaces into a single token.
Along with effectively getting a longer context window by getting more merging done in the same sized embedding table.

"""

import regex as re
from .base import Tokenizer, get_stats, merge

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class CapitalSpaceOutTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # map every upper case character to lower-case + 1 million

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        cs_merges = {} # (int, int) -> (int, int, int)  # (token, capitalization, space)
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
