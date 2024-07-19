"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time
from minbpe import BasicTokenizer, RegexTokenizer, CapitalSpaceOutTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/egg.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

# for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer, CapitalSpaceOutTokenizer], ["basic", "regex", "CapitalSpaceOut"]):
for TokenizerClass, name in zip([RegexTokenizer, CapitalSpaceOutTokenizer], ["regexTrain", "CapitalSpaceOutTrain"]):
    t0 = time.time()
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text

    tokenizer_ids = tokenizer.encode(text, 'all')
    print(name, f"{len(tokenizer_ids)=}")

    t1 = time.time()
    print(name, f"Training took {t1 - t0:.2f} seconds")