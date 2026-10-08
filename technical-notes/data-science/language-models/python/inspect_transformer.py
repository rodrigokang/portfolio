"""
Inspection of the architecture and tensor dimensions of a pretrained
Transformer language model.
"""

import os
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM
from transformers.utils import logging

logging.set_verbosity_error()
logging.disable_progress_bar()


def main():
    model_name = "HuggingFaceTB/SmolLM-135M"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
    )
    model.eval()

    config = model.config
    layer = model.model.layers[0]

    print("Model:")
    print(model_name)

    print("\nArchitecture:")
    print(f"Vocabulary size:       {config.vocab_size}")
    print(f"Model dimension:       {config.hidden_size}")
    print(f"Decoder layers:        {config.num_hidden_layers}")
    print(f"Attention heads:       {config.num_attention_heads}")
    print(f"Key-value heads:       {config.num_key_value_heads}")
    print(f"Feedforward dimension: {config.intermediate_size}")

    print("\nEmbedding layer:")
    print(model.get_input_embeddings())

    print("\nLanguage modelling head:")
    print(model.get_output_embeddings())

    print("\nFirst Transformer block:")
    print(layer)

    print("\nProjection weight shapes:")
    print(f"W_Q: {tuple(layer.self_attn.q_proj.weight.shape)}")
    print(f"W_K: {tuple(layer.self_attn.k_proj.weight.shape)}")
    print(f"W_V: {tuple(layer.self_attn.v_proj.weight.shape)}")
    print(f"W_O: {tuple(layer.self_attn.o_proj.weight.shape)}")


if __name__ == "__main__":
    main()