"""
Inspect static and contextualised token embeddings.
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings("ignore")

import torch
from huggingface_hub import logging as hf_logging
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

disable_progress_bars()
hf_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


def inspect_embeddings(text, tokenizer, model):
    """
    Inspect token IDs, static embeddings, and contextualised representations.

    Input
    -----
    text : str
        Text to process.
    tokenizer : PreTrainedTokenizer
        Pretrained tokenizer.
    model : PreTrainedModel
        Pretrained language model.

    Output
    ------
    None
    """
    inputs = tokenizer(text, return_tensors="pt")
    token_ids = inputs["input_ids"]

    embedding_layer = model.get_input_embeddings()

    with torch.no_grad():
        static_embeddings = embedding_layer(token_ids)

        output = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    contextualised_embeddings = output.hidden_states[-1]

    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])

    print("Tokens:")
    print(tokens)

    print("\nToken IDs:")
    print(token_ids[0].tolist())

    print("\nEmbedding matrix shape:")
    print(tuple(embedding_layer.weight.shape))

    print("\nStatic embeddings shape:")
    print(tuple(static_embeddings.shape))

    print("\nContextualised representations shape:")
    print(tuple(contextualised_embeddings.shape))


def compare_repeated_token(text, token, tokenizer, model):
    """
    Compare static and contextualised representations of a repeated token.

    Input
    -----
    text : str
        Text containing repeated occurrences of a token.
    token : str
        Token to compare.
    tokenizer : PreTrainedTokenizer
        Pretrained tokenizer.
    model : PreTrainedModel
        Pretrained language model.

    Output
    ------
    None
    """
    inputs = tokenizer(text, return_tensors="pt")
    token_ids = inputs["input_ids"]

    embedding_layer = model.get_input_embeddings()

    with torch.no_grad():
        static_embeddings = embedding_layer(token_ids)

        output = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    contextualised_embeddings = output.hidden_states[-1]
    tokens = tokenizer.convert_ids_to_tokens(token_ids[0])

    positions = [
        i for i, current_token in enumerate(tokens)
        if current_token == token
    ]

    if len(positions) < 2:
        print(f"Token {token!r} does not occur at least twice.")
        return

    i, j = positions[:2]

    static_equal = torch.equal(
        static_embeddings[0, i],
        static_embeddings[0, j],
    )

    contextualised_equal = torch.equal(
        contextualised_embeddings[0, i],
        contextualised_embeddings[0, j],
    )

    print("\nRepeated-token comparison:")
    print(f"Token: {token!r}")
    print(f"Positions: {i}, {j}")
    print(f"Static embeddings equal: {static_equal}")
    print(
        "Contextualised representations equal: "
        f"{contextualised_equal}"
    )


if __name__ == "__main__":
    model_name = "HuggingFaceTB/SmolLM-135M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    text = "The model transforms tokens into embeddings."
    inspect_embeddings(text, tokenizer, model)

    repeated_text = "The model learns and the model predicts."
    compare_repeated_token(
        repeated_text,
        "Ġmodel",
        tokenizer,
        model,
    )