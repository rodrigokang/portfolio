"""
Bag-of-words implementation from first principles.
"""


def tokenise(document):
    """
    Convert a document to lowercase, remove basic punctuation, and split it
    into tokens using whitespace.

    Input
    -----
    document : str
        Text document to tokenise.

    Output
    ------
    tokens : list[str]
        Sequence of tokens extracted from the document.
    """
    document = document.lower()

    for symbol in ".,!?;:":
        document = document.replace(symbol, "")

    tokens = document.split()

    return tokens


def build_vocabulary(tokenised_corpus):
    """
    Construct an ordered vocabulary containing the unique tokens in a corpus.

    Input
    -----
    tokenised_corpus : list[list[str]]
        Corpus represented as a collection of tokenised documents.

    Output
    ------
    vocabulary : list[str]
        Ordered list of unique tokens appearing in the corpus.
    """
    vocabulary = []
    seen = set()

    for document in tokenised_corpus:
        for token in document:
            if token not in seen:
                vocabulary.append(token)
                seen.add(token)

    return vocabulary


def bag_of_words(corpus):
    """
    Construct a bag-of-words representation of a corpus from first principles.

    Input
    -----
    corpus : list[str]
        Collection of text documents.

    Output
    ------
    vocabulary : list[str]
        Ordered vocabulary constructed from the corpus.
    matrix : list[list[int]]
        Document-term matrix containing token frequencies.
    """
    tokenised_corpus = [tokenise(document) for document in corpus]
    vocabulary = build_vocabulary(tokenised_corpus)

    token_to_index = {
        token: index
        for index, token in enumerate(vocabulary)
    }

    matrix = []

    for document in tokenised_corpus:
        vector = [0] * len(vocabulary)

        for token in document:
            index = token_to_index[token]
            vector[index] += 1

        matrix.append(vector)

    return vocabulary, matrix


if __name__ == "__main__":
    corpus = [
        "The product is good.",
        "The product is very good.",
        "The service is bad.",
    ]

    vocabulary, matrix = bag_of_words(corpus)

    print("Vocabulary:")
    print(vocabulary)

    print("\nDocument-term matrix:")
    for row in matrix:
        print(row)