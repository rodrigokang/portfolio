"""Implement Skip-Gram with Negative Sampling from first principles."""

import numpy as np


def sigmoid(x):
    """
    Description
    -----------
    Compute the logistic sigmoid function.

    Input
    -----
    x : float or ndarray
        Input value.

    Output
    ------
    y : float or ndarray
        Sigmoid of the input.
    """
    x = np.clip(x, -20.0, 20.0)

    return 1.0 / (1.0 + np.exp(-x))


def build_vocabulary(corpus):
    """
    Description
    -----------
    Construct the vocabulary and integer word IDs.

    Input
    -----
    corpus : list
        Tokenised sentences.

    Output
    ------
    word_to_id, id_to_word : dict
        Mappings between words and integer IDs.
    """
    words = sorted({
        word
        for sentence in corpus
        for word in sentence
    })

    word_to_id = {
        word: i
        for i, word in enumerate(words)
    }

    id_to_word = {
        i: word
        for word, i in word_to_id.items()
    }

    return word_to_id, id_to_word


def generate_pairs(corpus, word_to_id, window_size):
    """
    Description
    -----------
    Generate positive skip-gram centre-context pairs.

    Input
    -----
    corpus : list
        Tokenised sentences.
    word_to_id : dict
        Mapping from words to IDs.
    window_size : int
        Number of context positions on each side.

    Output
    ------
    pairs : list
        Positive centre-context ID pairs.
    """
    pairs = []

    for sentence in corpus:
        ids = [word_to_id[word] for word in sentence]

        for i, centre in enumerate(ids):
            left = max(0, i - window_size)
            right = min(len(ids), i + window_size + 1)

            for j in range(left, right):
                if i != j:
                    pairs.append((centre, ids[j]))

    return pairs


def sample_negatives(
    vocabulary_size,
    positive_context,
    centre,
    positive_pairs,
    n_samples,
    rng,
):
    """
    Description
    -----------
    Sample negative context words.

    Input
    -----
    vocabulary_size, positive_context, centre, positive_pairs,
    n_samples, rng
        Parameters required for negative sampling.

    Output
    ------
    negatives : list
        Negative context IDs.
    """
    negatives = []

    while len(negatives) < n_samples:
        candidate = rng.integers(vocabulary_size)

        if (
            candidate != positive_context
            and (centre, candidate) not in positive_pairs
        ):
            negatives.append(candidate)

    return negatives


def train_word2vec(
    corpus,
    embedding_dim=20,
    window_size=2,
    n_negative=5,
    learning_rate=0.025,
    epochs=300,
    seed=42,
):
    """
    Description
    -----------
    Train Skip-Gram with Negative Sampling.

    Input
    -----
    corpus : list
        Tokenised training corpus.

    Output
    ------
    U, V, word_to_id, id_to_word : tuple
        Learned embeddings and vocabulary mappings.
    """
    rng = np.random.default_rng(seed)

    word_to_id, id_to_word = build_vocabulary(corpus)

    pairs = generate_pairs(
        corpus,
        word_to_id,
        window_size,
    )

    positive_pairs = set(pairs)
    vocabulary_size = len(word_to_id)

    U = rng.normal(
        0.0,
        0.1,
        size=(vocabulary_size, embedding_dim),
    )

    V = rng.normal(
        0.0,
        0.1,
        size=(vocabulary_size, embedding_dim),
    )

    for epoch in range(epochs):
        total_loss = 0.0

        order = rng.permutation(len(pairs))

        for index in order:
            centre, context = pairs[index]

            negatives = sample_negatives(
                vocabulary_size,
                context,
                centre,
                positive_pairs,
                n_negative,
                rng,
            )

            u = U[centre].copy()
            v_positive = V[context].copy()
            v_negative = V[negatives].copy()

            positive_score = np.dot(
                u,
                v_positive,
            )

            negative_scores = v_negative @ u

            positive_probability = sigmoid(
                positive_score
            )

            negative_probabilities = sigmoid(
                negative_scores
            )

            total_loss += -np.log(
                positive_probability + 1e-12
            )

            total_loss += -np.sum(
                np.log(
                    sigmoid(-negative_scores) + 1e-12
                )
            )

            grad_u = (
                (positive_probability - 1.0)
                * v_positive
                + negative_probabilities @ v_negative
            )

            grad_v_positive = (
                (positive_probability - 1.0)
                * u
            )

            grad_v_negative = (
                negative_probabilities[:, None]
                * u
            )

            U[centre] -= learning_rate * grad_u
            V[context] -= (
                learning_rate * grad_v_positive
            )

            for k, negative in enumerate(negatives):
                V[negative] -= (
                    learning_rate
                    * grad_v_negative[k]
                )

        if (epoch + 1) % 50 == 0:
            average_loss = total_loss / len(pairs)

            print(
                f"Epoch {epoch + 1:3d} "
                f"| Loss: {average_loss:.4f}"
            )

    return U, V, word_to_id, id_to_word


def cosine_similarity(x, y):
    """
    Description
    -----------
    Compute cosine similarity between two vectors.

    Input
    -----
    x, y : ndarray
        Input vectors.

    Output
    ------
    similarity : float
        Cosine similarity.
    """
    return np.dot(x, y) / (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )


def most_similar(word, U, word_to_id, id_to_word, top_n=3):
    """
    Description
    -----------
    Find words with the most similar centre embeddings.

    Input
    -----
    word, U, word_to_id, id_to_word, top_n
        Word and learned embedding information.

    Output
    ------
    similarities : list
        Most similar words and cosine similarities.
    """
    word_id = word_to_id[word]
    vector = U[word_id]

    similarities = []

    for i in range(len(U)):
        if i == word_id:
            continue

        similarity = cosine_similarity(
            vector,
            U[i],
        )

        similarities.append(
            (id_to_word[i], similarity)
        )

    similarities.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return similarities[:top_n]


if __name__ == "__main__":
    corpus = [
        ["cat", "likes", "milk"],
        ["cat", "drinks", "milk"],
        ["cat", "likes", "food"],
        ["dog", "likes", "food"],
        ["dog", "eats", "food"],
        ["dog", "drinks", "water"],
        ["kitten", "likes", "milk"],
        ["kitten", "drinks", "milk"],
        ["puppy", "likes", "food"],
        ["puppy", "drinks", "water"],
    ]

    U, V, word_to_id, id_to_word = train_word2vec(
        corpus
    )

    print("\nMost similar words:")

    for word in ["cat", "dog", "kitten", "puppy"]:
        print(f"\n{word}:")

        for neighbour, similarity in most_similar(
            word,
            U,
            word_to_id,
            id_to_word,
        ):
            print(
                f"  {neighbour:<10} "
                f"{similarity:.4f}"
            )