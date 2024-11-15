"""
todo list:
Done: 0> Why does tokenization help - what is the logprob of the tokenized text compared to original text just using bigrams?
    If the text was completely uniform, then tokenization wouldn't help at all.  But if the text has some structure, then tokenization should help.
1> Understand the code
Done: 2> Understand the test code - how to invoke it
Done: 3> Add test.py to test the code
Done: 4> Add test case to test.py to the count of tokens for egg, Egg, EGG example from Karpathy on all tokenizers
Done: 5> Make capital_space_out.py work with no changes - just new names
Done: 6> Add test capital_space_out.py to test the code and test.py
Done: 6> Make tokenization create tokens = token + 1M * capitalization + 10M * space.  Instead of just token - check it works.
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
    +1 million for capitalized meaning just the first letter upper-case (including a single upper cased letter)
    +2 million for all all-caps
    +3 million is a mish-mash of upper and lower case - we can only store 1 type of mish-mash in the decode table, so skip the other types of mish-mash when merging
"""
LETTER_OFFSETS     = 1_000_000
LOWERCASE_OFFSET   = 0          # 1 or more lower case letters
CAPITALIZED_OFFSET = 1_000_000  # 1 capital letter followed by 0 or more lower case letters
ALLCAPS_OFFSET     = 2_000_000  # 2 or more capital letters
MISHMASH_OFFSET    = 3_000_000  # 2 or more mix of capital and lower case letters in non-standard way
SPACEOUT_OFFSET    = 40_000_000 # 1 space to the left of the token
"""
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

"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions

def get_stats(ids, stats_raw=None, stats_merged=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    # stats_raw is counts of consecutive pairs ignoring SPACEOUT_OFFSET
    # stats_merged is counts of consecutive pairs modulo SPACEOUT_OFFSET and LETTER_OFFSETS
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        # ignore the SPACE_OFFSET for character matching on counts
        item_0 = pair[0] % SPACEOUT_OFFSET
        item_1 = pair[1] % SPACEOUT_OFFSET
        pair_m = (item_0, item_1)
        stats_raw[pair_m] = stats_raw.get(pair_m, 0) + 1
        # ignore the LETTER_OFFSETS for character matching on stats_m
        item_0 = pair[0] % LETTER_OFFSETS
        item_1 = pair[1] % LETTER_OFFSETS
        pair_m = (item_0, item_1)
        stats_merged[pair_m] = stats_merged.get(pair_m, 0) + 1

    return

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        print(f"{model_file=}", model_file)
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe CapitalSpaceOut v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            f.write(f"{len(self.merges)}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

            # enumerate k,v from vocab_cnt_ids - it's not contiguous with special tokens occurring anywhere
            f.write(f"{len(self.vocab_cnt_ids)}\n")
            for k, v in self.vocab_cnt_ids.items():
                # The first 256 are contiguous and special cased.
                if k >= 256:
                    f.write(f"{k} {v}\n")

            # enumerate k,v from recursive_vocab - it's not contiguous with special tokens occurring anywhere
            f.write(f"{len(self.recursive_vocab)}\n")
            for k, v in self.recursive_vocab.items():
                # The first 256 are contiguous and special cased.
                if k >= 256:
                    # print("save recursive_vocab", k, v, self.vocab_cnt_ids[k])
                    if self.vocab_cnt_ids[k] == 1: # special tokens (and 0-255) all map to just 1 token
                        f.write(f"{k} {v[0]}\n")
                    else:
                        f.write(f"{k} {v[0]} {v[1]}\n") # the merged tokens

        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe CapitalSpaceOut v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            # read the merges
            num_merges = int(f.readline().strip())
            for _ in range(num_merges):
                line = f.readline().strip()
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

            vocab_cnt_ids = {idx: 1 for idx in range(256)} # idx -> count of ids
            num_vocab_cnt_ids = int(f.readline().strip())
            for i in range(256, num_vocab_cnt_ids):
                line = f.readline().strip()
                key, idx = map(int, line.split())
                vocab_cnt_ids[key] = idx

            recursive_vocab = {idx: [idx] for idx in range(256)} # idx -> list of ids
            num_recursive_vocab = int(f.readline().strip())
            for i in range(256, num_recursive_vocab):
                line = f.readline().strip()
                line_split = line.split()
                # print("load recursive_vocab", i, line, line_split, line_split[0], vocab_cnt_ids[int(line_split[0])])
                if vocab_cnt_ids[int(line_split[0])] == 1:
                    key, idx1 = map(int, line_split)
                    recursive_vocab[key] = (idx1,)
                else:
                    key, idx1, idx2 = map(int, line_split)
                    recursive_vocab[key] = (idx1, idx2)

        self.special_tokens = special_tokens
        self.merges = merges
        self.vocab = self._build_vocab()
        self.recursive_vocab = recursive_vocab # used in decode() - recursive decoding to the individual tokens
        self.vocab_cnt_ids = vocab_cnt_ids # used in decode() - recursive decoding to the individual tokens

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

        # iteratively merge the most common pairs to create new tokens
        self.merges = {} # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        self.recursive_vocab = {idx: [idx] for idx in range(256)} # idx -> list of ids
        self.vocab_cnt_ids = {idx: 1 for idx in range(256)} # idx -> count of ids

    def merge(self, ids, pair, idx):
        """
        In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        base_pair = (pair[0] % LETTER_OFFSETS, pair[1] % LETTER_OFFSETS)
        newids = []
        i = 0
        while i < len(ids) - 1:
            # if the base_pair matches for valid LETTER_OFFSETS combos, replace it.
            ids_i0 = ids[i] % SPACEOUT_OFFSET # ignore the SPACEOUT_OFFSET for character matching
            space_offset0 = ids[i] - ids_i0 # But if there is a SPACEOUT_OFFSET on the left, we need to preserve it.
            # ids_i1 = ids[i+1] % SPACEOUT_OFFSET - can't be SPACEOUT_OFFSET on the right, so we don't need to check/preserve it.
            if ids_i0 == (base_pair[0] + LOWERCASE_OFFSET) and ids[i+1] == (base_pair[1] + LOWERCASE_OFFSET):
                newids.append(idx + LOWERCASE_OFFSET + space_offset0)
                i += 2
            elif ids_i0 == (base_pair[0] + CAPITALIZED_OFFSET) and ids[i+1] == (base_pair[1] + LOWERCASE_OFFSET):
                newids.append(idx + CAPITALIZED_OFFSET + space_offset0)
                i += 2
            elif ids_i0 == (base_pair[0] + CAPITALIZED_OFFSET) and ids[i+1] == (base_pair[1] + CAPITALIZED_OFFSET) and self.vocab_cnt_ids[base_pair[0]] == 1 and self.vocab_cnt_ids[base_pair[1]] == 1:
                newids.append(idx + ALLCAPS_OFFSET + space_offset0)
                i += 2
            elif ids_i0 == (base_pair[0] + CAPITALIZED_OFFSET) and ids[i+1] == (base_pair[1] + ALLCAPS_OFFSET) and self.vocab_cnt_ids[base_pair[0]] == 1:
                newids.append(idx + ALLCAPS_OFFSET + space_offset0)
                i += 2
            elif ids_i0 == (base_pair[0] + ALLCAPS_OFFSET) and ids[i+1] == (base_pair[1] + CAPITALIZED_OFFSET) and self.vocab_cnt_ids[base_pair[1]] == 1:
                newids.append(idx + ALLCAPS_OFFSET + space_offset0)
                i += 2
            elif ids_i0 == (base_pair[0] + ALLCAPS_OFFSET) and ids[i+1] == (base_pair[1] + ALLCAPS_OFFSET):
                newids.append(idx + ALLCAPS_OFFSET + space_offset0)
                i += 2
            elif ids_i0 == pair[0] and ids[i+1] == pair[1]:
                # find which is the most common mish-mash and store it in the table for decoding
                newids.append(idx + MISHMASH_OFFSET + space_offset0)
                i += 2
            else:
                newids.append(ids[i])
                i += 1

        if i == len(ids) - 1:
            newids.append(ids[-1])

        return newids

    def train(self, text, vocab_size, verbose=True):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        # map every upper case character to lower-case + CAPITALIZED_OFFSET
        ids = []
        for chunk in text_chunks:
            # print(f"{chunk=}")
            ids_piece = []
            i = 0
            # 1st space is a special case, remove it if the next character is not a space
            space_out = False
            if len(chunk) > 1 and ord(chunk[0]) == 32 and ord(chunk[1]) != 32:
                space_out = True
                # skip the first space in the output
                i = 1

            while i < len(chunk):
                c = chunk[i]
                if 65 <= ord(c) <= 90:
                    # map every upper case character to lower-case + CAPITALIZED_OFFSET
                    new_c = [ord(c) + CAPITALIZED_OFFSET + (97-65)]
                else:
                    new_c = list(c.encode("utf-8"))
                ids_piece += new_c
                i += 1
            if space_out == True:
                # add a space_out to the first character
                ids_piece[0] += SPACEOUT_OFFSET

            ids.append(ids_piece)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        recursive_vocab = {idx: [idx] for idx in range(256)} # idx -> list of ids
        vocab_cnt_ids = {idx: 1 for idx in range(256)} # idx -> count of ids
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode() - 1 shot decoding to the whole string
        self.recursive_vocab = recursive_vocab # used in decode() - recursive decoding to the individual tokens
        self.vocab_cnt_ids = vocab_cnt_ids # used in decode() - recursive decoding to the individual tokens
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats_raw = {}
            stats_merged = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats_raw, stats_merged)
            # find the pair of codepoints with the highest count, ignoring LETTER_OFFSETS
            pair = max(stats_merged, key=stats_merged.get)
            # find the pair_raw of codepoints with the highest count for the pair, respecting LETTER_OFFSETS
            # find the most common non-standard/invalid merge rules for the mish-mash case
            # invalid_combos lists all the invalid combinations, some of which are valid if their count of tokens is 1.
            invalid_combos = [# (LOWERCASE_OFFSET, LOWERCASE_OFFSET, 0, 0), # valid
                              (LOWERCASE_OFFSET, CAPITALIZED_OFFSET, 0, 0),
                              (LOWERCASE_OFFSET, ALLCAPS_OFFSET, 0, 0),
                              (LOWERCASE_OFFSET, MISHMASH_OFFSET, 0, 0),
                              # (CAPITALIZED_OFFSET, LOWERCASE_OFFSET, 0, 0), # valid
                              (CAPITALIZED_OFFSET, CAPITALIZED_OFFSET, 1, 1), # if first token or second is more than 1 letter, this is invalid
                              (CAPITALIZED_OFFSET, ALLCAPS_OFFSET, 1, 0),     # if first token is more than 1 letter, this is invalid
                              (CAPITALIZED_OFFSET, MISHMASH_OFFSET, 0, 0),
                              (ALLCAPS_OFFSET, LOWERCASE_OFFSET, 1, 0),       # if first token is more than 1 letter, this is invalid
                              (ALLCAPS_OFFSET, CAPITALIZED_OFFSET, 0, 1),     # if second token is more than 1 letter, this is invalid
                              # (ALLCAPS_OFFSET, ALLCAPS_OFFSET, 0, 0),       # valid
                              (ALLCAPS_OFFSET, MISHMASH_OFFSET, 0, 0),
                              (MISHMASH_OFFSET, LOWERCASE_OFFSET, 0, 0),
                              (MISHMASH_OFFSET, CAPITALIZED_OFFSET, 0, 0),
                              (MISHMASH_OFFSET, ALLCAPS_OFFSET, 0, 0),
                              (MISHMASH_OFFSET, MISHMASH_OFFSET, 0, 0)]

            best_invalid_combo = (LOWERCASE_OFFSET, CAPITALIZED_OFFSET) # Make iPhone work by default if nothing else is found
            best_invalid_count = 0
            # which of the invalid combos has the highest count?  Use it for the mish-mash case
            for invalid_combo in invalid_combos:
                if invalid_combo[2] >= self.vocab_cnt_ids[pair[0] % LETTER_OFFSETS]:
                    continue # valid combo, skip it
                if invalid_combo[3] >= self.vocab_cnt_ids[pair[1] % LETTER_OFFSETS]:
                    continue # valid combo, skip it
                invalid_count = stats_raw.get((pair[0] + invalid_combo[0], pair[1] + invalid_combo[1]), 0)
                if invalid_count > best_invalid_count:
                    best_invalid_combo = invalid_combo
                    best_invalid_count = invalid_count

            pair_raw = (pair[0] + best_invalid_combo[0], pair[1] + best_invalid_combo[1])
            # mint a new token: assign it the next available id
            idx = 256 + i
            print(f"{idx=} {pair=} {pair_raw=}")
            # save the merge
            merges[pair] = idx
            recursive_vocab[idx] = pair_raw # MISHMASH_OFFSET is supported by recording the pair that was merged here
            vocab[idx] = vocab[pair[0] % LETTER_OFFSETS] + vocab[pair[1] % LETTER_OFFSETS]
            vocab_cnt_ids[idx] = vocab_cnt_ids[pair[0] % LETTER_OFFSETS] + vocab_cnt_ids[pair[1] % LETTER_OFFSETS]
            # replace all occurrences of pair in ids with idx
            ids = [self.merge(chunk_ids, pair_raw, idx) for chunk_ids in ids]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats_merged[pair]} occurrences")

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        for k, v in special_tokens.items():
            # print(f"register_special_tokens {k=} {v=}")
            self.vocab[v] = k.encode("utf-8")
            self.vocab_cnt_ids[v] = 1
            self.recursive_vocab[v] = (v,)

    def decode_recursive(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            idx_char = idx % SPACEOUT_OFFSET
            idx_space = idx - idx_char # equals SPACEOUT_OFFSET or 0
            idx_base = idx_char % LETTER_OFFSETS
            idx_cap = idx_char - idx_base
            if self.vocab_cnt_ids[idx_base] == 1:
                    if idx_space == SPACEOUT_OFFSET:
                        part_bytes.append(b" ")
                    if idx_base >= 97 and idx_base <= 122 and idx_cap != LOWERCASE_OFFSET:
                        # make CAPITALIZED and ALLCAPS into an upper case letter
                        part_bytes.append(self.vocab[idx_base - (97 - 65)])
                    else:
                        part_bytes.append(self.vocab[idx_base])
            elif idx_base in self.vocab:
                # It's a merged token - recurse on it - pass down the case markings properly
                # To support mish-mash, we store the mish-mash character markings in the recursive_vocab table
                # So zero out the caps markings when copying out of the table when not mish-mash.
                if idx_cap == LOWERCASE_OFFSET:
                    # LOWERCASE_OFFSET -> LOWERCASE_OFFSET, LOWERCASE_OFFSET
                    recurse_ids = [self.recursive_vocab[idx_base][0] % LETTER_OFFSETS, self.recursive_vocab[idx_base][1] % LETTER_OFFSETS]
                    recurse_ids[0] += LOWERCASE_OFFSET
                    if idx_space == SPACEOUT_OFFSET:
                        recurse_ids[0] += SPACEOUT_OFFSET
                    recurse_ids[1] += LOWERCASE_OFFSET
                    part_bytes += self.decode_recursive(recurse_ids)
                elif idx_cap == CAPITALIZED_OFFSET:
                    # CAPITALIZED_OFFSET -> CAPITALIZED_OFFSET, LOWERCASE_OFFSET
                    recurse_ids = [self.recursive_vocab[idx_base][0] % LETTER_OFFSETS, self.recursive_vocab[idx_base][1] % LETTER_OFFSETS]
                    recurse_ids[0] += CAPITALIZED_OFFSET
                    if idx_space == SPACEOUT_OFFSET:
                        recurse_ids[0] += SPACEOUT_OFFSET
                    recurse_ids[1] += LOWERCASE_OFFSET
                    part_bytes += self.decode_recursive(recurse_ids)
                elif idx_cap == ALLCAPS_OFFSET:
                    # ALLCAPS_OFFSET -> ALLCAPS_OFFSET, ALLCAPS_OFFSET
                    recurse_ids = [self.recursive_vocab[idx_base][0] % LETTER_OFFSETS, self.recursive_vocab[idx_base][1] % LETTER_OFFSETS]
                    recurse_ids[0] += ALLCAPS_OFFSET
                    if idx_space == SPACEOUT_OFFSET:
                        recurse_ids[0] += SPACEOUT_OFFSET
                    recurse_ids[1] += ALLCAPS_OFFSET
                    part_bytes += self.decode_recursive(recurse_ids)
                else:
                    # MISHMASH_OFFSET -> Table lookup, Table lookup - whatever was merged together in the table is preserved.
                    assert idx_cap == MISHMASH_OFFSET
                    recurse_ids = [self.recursive_vocab[idx_base][0], self.recursive_vocab[idx_base][1]]
                    if idx_space == SPACEOUT_OFFSET:
                        recurse_ids[0] += SPACEOUT_OFFSET
                    part_bytes += self.decode_recursive(recurse_ids)
            else:
                raise ValueError(f"decode_recursive invalid token id: {idx} {idx_base} {idx_cap} {self.vocab_cnt_ids[idx_base]}")
        return part_bytes

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            idx_base = idx % LETTER_OFFSETS
            if idx_base in self.vocab:
                # part_bytes.append(self.vocab[idx])
                part_bytes += self.decode_recursive([idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, ids):
        # convert the list of text bytes to the token ids
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats_raw = {}
            stats_merged = {}
            get_stats(ids, stats_raw, stats_merged)
            c_ids = len(ids)
            # A pair might not result in a merge if it's not a compatible LETTER_OFFSETS pair
            # so skip it and keep looking for a pair that will merge, until nothing left to merge
            while c_ids == len(ids): # keep trying until a merge is done
                if len(stats_merged) == 0:  # Are there any pairs left to merge?
                    break
                base_pair = min(stats_merged, key=lambda p: self.merges.get(p, float("inf")))
                # print(f"{pair=} {stats_merged[pair]=} {self.merges.get(pair, float('inf'))=}")
                # subtle: if there are no more merges available, the key will
                # result in an inf for every single pair, and the min will be
                # just the first pair in the list, arbitrarily
                # we can detect this terminating case by a membership check
                if base_pair not in self.merges:
                    break # nothing else can be merged anymore
                # otherwise let's merge the base_pair (lowest merge index)
                idx = self.merges[base_pair]
                pair = self.recursive_vocab[idx] # get the mish-mash pair
                ids = self.merge(ids, pair, idx)
                del stats_merged[base_pair] # don't consider this pair again

            if c_ids == len(ids):  # if no merges were done, we're finished
                break
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        # all chunks of text are encoded separately, then results are joined
        for chunk in text_chunks:
            # print(f"{chunk=}")
            ids_chunk = []
            i = 0
            # 1st space is a special case, remove it if the next character is not a space
            space_out = False
            if len(chunk) > 1 and ord(chunk[0]) == 32 and ord(chunk[1]) != 32:
                space_out = True
                # skip the first space in the output
                i = 1

            while i < len(chunk):
                c = chunk[i]
                if 65 <= ord(c) <= 90:
                    # map every upper case character to lower-case + CAPITALIZED_OFFSET
                    new_c = [ord(c) + CAPITALIZED_OFFSET + (97-65)]
                else:
                    new_c = list(c.encode("utf-8"))

                ids_chunk += new_c
                i += 1
            chunk_ids = self._encode_chunk(ids_chunk)
            if space_out == True:
                # add a space_out to the first character
                chunk_ids[0] += SPACEOUT_OFFSET
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
