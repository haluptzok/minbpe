"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""
import os
import time
import tiktoken
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer, CapitalSpaceOutTokenizer

egg_test_string = '''
egg egg egg
Egg Egg Egg
EGG EGG EGG
'''
text = egg_test_string
tokenizer = GPT4Tokenizer()
enc = tiktoken.get_encoding("cl100k_base")
tiktoken_ids = enc.encode(egg_test_string)
tiktoken_str = enc.decode(tiktoken_ids)
gpt4_tokenizer_ids = tokenizer.encode(egg_test_string)
gpt4_tokenizer_str = tokenizer.decode(gpt4_tokenizer_ids)
assert gpt4_tokenizer_ids == tiktoken_ids
assert gpt4_tokenizer_str == tiktoken_str
assert gpt4_tokenizer_str == egg_test_string
print("GPT4 Tokenizer on egg_test_string:")
print(egg_test_string)
print(gpt4_tokenizer_ids)

# open some text and train a vocab of 512 tokens
# filename = "taylorswift" # fast test - 15 seconds
filename = "egg" # fast test - 15 seconds
# filename = "input"     # slow test - 115 seconds
text = open("tests/" + filename + ".txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

for TokenizerClass, name in zip([CapitalSpaceOutTokenizer], ["CapitalSpaceOutTokenizer"]):
    print(f"Tokenizer {name} training on {filename}.txt:")
    t0 = time.time()
    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
    t1 = time.time()
    print(f"Training {name} took {t1 - t0:.2f} seconds")
    t0 = time.time()
    # test we can tokenize and detokenize the training set exactly
    tokenizer_ids = tokenizer.encode(egg_test_string)
    tokenizer_str = tokenizer.decode(tokenizer_ids)
    assert tokenizer_str == egg_test_string
    # test we can tokenize and detokenize the egg_test_string exactly
    tokenizer_ids = tokenizer.encode(egg_test_string)
    tokenizer_str = tokenizer.decode(tokenizer_ids)
    assert tokenizer_str == egg_test_string
    print(f"Tokenizer {name} on {filename}.txt:")
    print(tokenizer_ids)
    print(tokenizer_str)
    t1 = time.time()
    print(f"Testing {name} took {t1 - t0:.2f} seconds")
