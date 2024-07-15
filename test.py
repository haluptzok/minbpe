"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""
import os
import time
import tiktoken
from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer, CapitalSpaceOutTokenizer

short_test_string1 = '?'
short_test_string2 = 'a'
short_test_string3 = 'A <|endoftext|> egg <|fim_prefix|> Egg <|fim_middle|> EGG <|fim_suffix|> eGG <|endofprompt|> iPhone'

egg_test_string = '''
egg egg egg
Egg Egg Egg
EGG EGG EGG
eGG eGG eGG
egG egG egG
EGg EGg EGg
EgG EgG EgG
eGg eGg eGg
'''

iphone_test_string = '''
iphone iphone iphone
Iphone Iphone Iphone
IPHONE IPHONE IPHONE
iPhone iPhone iPhone
IPhone IPhone IPhone
iPHONE iPHONE iPHONE
iPhoNe IPhoNe IPhoNe
iphONE iphONE iphONE
iPhoNE IPhoNE IPhoNE
'''

text = egg_test_string
tokenizer = GPT4Tokenizer()
enc = tiktoken.get_encoding("cl100k_base")

for text in [egg_test_string, iphone_test_string]:
    tiktoken_ids = enc.encode(text)
    tiktoken_str = enc.decode(tiktoken_ids)
    gpt4_tokenizer_ids = tokenizer.encode(text)
    gpt4_tokenizer_str = tokenizer.decode(gpt4_tokenizer_ids)
    assert gpt4_tokenizer_ids == tiktoken_ids
    assert gpt4_tokenizer_str == tiktoken_str
    assert gpt4_tokenizer_str == text
    print("GPT4 Tokenizer on egg_test_string:")
    print(text)
    print(gpt4_tokenizer_ids)
    for token in gpt4_tokenizer_ids:
        print(f'{token:14d}', tokenizer.decode([token]))


# open some text and train a vocab of 512 tokens
# filename = "taylorswift" # fast test - 15 seconds
filename = "egg" # fast test - 20 seconds
# filename = "input"     # slow test - 115 seconds
text = open("tests/" + filename + ".txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

# for TokenizerClass, name in zip([CapitalSpaceOutTokenizer], ["CapitalSpaceOutTokenizer"]):
for TokenizerClass, name in zip([RegexTokenizer, CapitalSpaceOutTokenizer], ["RegExTokenizer", "CapitalSpaceOutTokenizer"]):
# for TokenizerClass, name in zip([RegexTokenizer], ["RegExTokenizer"]):
    print(f"Tokenizer {name} training on {filename}.txt:")
    t0 = time.time()
    # construct the Tokenizer object and kick off verbose training
    tokenizer_train = TokenizerClass()
    vocab_size = 512
    tokenizer_train.train(text,vocab_size)
    tokenizer_train.register_special_tokens(special_tokens)

    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer_train.save(prefix)
    t1 = time.time()
    print(f"Training {name} took {t1 - t0:.2f} seconds")
    t0 = time.time()
    # test we can load and tokenize/detokenize the test strings exactly
    tokenizer = TokenizerClass()
    tokenizer.load(prefix + ".model")
    tokenizer.save(prefix + ".model_comp")
    for text in [egg_test_string, iphone_test_string, short_test_string1, short_test_string2, short_test_string3]:
        print(f"Tokenizer {name} trained on {filename}.txt tested on:\n", text)
        tokenizer_ids = tokenizer.encode(text, "none")
        print(tokenizer_ids)
        for token in tokenizer_ids:
            print(f'{token:14d}', tokenizer.vocab[token % 1_000_000], tokenizer.recursive_vocab[token % 1_000_000] if hasattr(tokenizer, "recursive_vocab") else 0, tokenizer.decode([token]))
        tokenizer_str = tokenizer.decode(tokenizer_ids)
        print("Test String and token Lengths:", len(text), len(tokenizer_ids))
        assert tokenizer_str == text
        tokenizer_ids = tokenizer.encode(text, 'all')
        print(tokenizer_ids)
        for token in tokenizer_ids:
            print(f'{token:14d}', tokenizer.vocab[token % 1_000_000], tokenizer.recursive_vocab[token % 1_000_000] if hasattr(tokenizer, "recursive_vocab") else 0, tokenizer.decode([token]))
        tokenizer_str = tokenizer.decode(tokenizer_ids)
        print("Test String and token Lengths:", len(text), len(tokenizer_ids))
        assert tokenizer_str == text



    # test we can tokenize and detokenize the training sets exactly
    for training_set in ["egg", "input", "taylorswift"]:
        print(f"Tokenizer {name} trained on {filename}.txt tested on {training_set}.txt::")
        text = open("tests/" + training_set + ".txt", "r", encoding="utf-8").read()
        tokenizer_ids = tokenizer.encode(text)
        tokenizer_str = tokenizer.decode(tokenizer_ids)
        print("Test String and token Lengths:", len(text), len(tokenizer_ids))
        assert tokenizer_str == text
    t1 = time.time()
    print(f"Testing {name} took {t1 - t0:.2f} seconds")
    # print(tokenizer.vocab)
    for i in range(256, vocab_size):
        if hasattr(tokenizer, "recursive_vocab"):
            if tokenizer.recursive_vocab[i][0] != 226:
                print(i, tokenizer.vocab[i], tokenizer.recursive_vocab[i], tokenizer.decode([i]), tokenizer.decode([i + 1_000_000]), tokenizer.decode([i + 2_000_000]), tokenizer.decode([i + 3_000_000]))
            else:
                print(i, tokenizer.vocab[i], tokenizer.recursive_vocab[i], tokenizer.decode([i]).encode("utf-8"), tokenizer.decode([i + 1_000_000]).encode("utf-8"), tokenizer.decode([i + 2_000_000]).encode("utf-8"), tokenizer.decode([i + 3_000_000]).encode("utf-8"))
        else:
            print(i, tokenizer.vocab[i], tokenizer.decode([i]))
