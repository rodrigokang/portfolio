"""
Inspect the tokenisation pipeline of a pretrained language model.
"""

import os
import warnings

# Suppress warnings from external libraries.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer


def inspect_tokenization(text, tokenizer):
    """
    Inspect the tokens and token IDs produced for an input text.

    Input
    -----
    text : str
        Text to tokenize.
    tokenizer : PreTrainedTokenizer
        Pretrained tokenizer.

    Output
    ------
    token_ids : list
        Token IDs produced by the tokenizer.
    """
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print("Input text:")
    print(text)

    print("\nTokens:")
    print(tokens)

    print("\nToken IDs:")
    print(token_ids)

    print("\nToken-ID pairs:")
    for token, token_id in zip(tokens, token_ids):
        print(f"{token!r:20} -> {token_id}")

    print("\nDecoded text:")
    print(tokenizer.decode(token_ids))

    return token_ids


def inspect_tokenizer(tokenizer):
    """
    Inspect basic properties of a pretrained tokenizer.

    Input
    -----
    tokenizer : PreTrainedTokenizer
        Pretrained tokenizer.

    Output
    ------
    None
    """
    print("Vocabulary size:")
    print(len(tokenizer))

    print("\nSpecial tokens:")
    print(tokenizer.special_tokens_map)


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM-135M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "Tokenization transforms text into tokens."

    inspect_tokenization(text, tokenizer)

    print("\n" + "=" * 50 + "\n")

    inspect_tokenizer(tokenizer)