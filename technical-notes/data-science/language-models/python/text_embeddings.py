"""Generate and compare text embeddings."""

import os
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

warnings.filterwarnings("ignore")

import torch
from huggingface_hub import logging as hf_logging
from huggingface_hub.utils import disable_progress_bars
from sentence_transformers import SentenceTransformer

disable_progress_bars()
hf_logging.set_verbosity_error()


def cosine_similarity(x, y):
    """
    Description
    -----------
    Compute the cosine similarity between two vectors.

    Input
    -----
    x, y : torch.Tensor
        Input vectors.

    Output
    ------
    similarity : float
        Cosine similarity between the vectors.
    """
    similarity = torch.dot(x, y) / (
        torch.linalg.vector_norm(x)
        * torch.linalg.vector_norm(y)
    )

    return similarity.item()


def compare_embeddings(sentences, model):
    """
    Description
    -----------
    Generate text embeddings and compare their cosine similarities.

    Input
    -----
    sentences : list
        Sentences to encode.
    model : SentenceTransformer
        Pretrained text embedding model.

    Output
    ------
    None
    """
    embeddings = model.encode(
        sentences,
        convert_to_tensor=True,
    )

    print("Embedding matrix shape:")
    print(tuple(embeddings.shape))

    print("\nSentences:")
    for i, sentence in enumerate(sentences):
        print(f"{i}: {sentence}")

    print("\nCosine similarities:")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = cosine_similarity(
                embeddings[i],
                embeddings[j],
            )

            print(f"{i} <-> {j}: {similarity:.4f}")


if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"

    model = SentenceTransformer(model_name)

    sentences = [
        "The cat is sleeping on the sofa.",
        "A cat is resting on the couch.",
        "The financial market closed higher today.",
    ]

    compare_embeddings(sentences, model)