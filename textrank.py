import networkx as nx
import numpy as np
import spacy

from networkx.exception import PowerIterationFailedConvergence
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load('en_core_web_lg')


def vectorize(sentences):
    """Takes a list of sentences and yields 300-dimensional
    row vectors.

    args:
        sentences :: List[str]

    returns:
        List[np.ndarray of shape (1, 300)]
    """
    vectors = []
    for sent in sentences:
        vectors.append(nlp(sent).vector.reshape(1, -1))
    return vectors


def cosine(u, v):
    """
    args:
        u :: vector of shape (1, 300)
        v :: vector of shape (1, 300)

    returns:
        u.dot(v.T) / (||x|| * ||y||) :: float
    """
    return cosine_similarity(u, v).item()


def _init_square_matrix(n):
    """
    args:
        n :: int

    returns:
        np.zeros(shape=(n, n)) :: float matrix of shape (n, n)
    """
    return np.zeros(shape=(n, n))


def prepare_graph(vectors, metric=cosine):
    """
    args:
        vectors :: List[np.ndarray of shape (1, 300)]
        metric :: a binary float-valued function with range [0, 1]

    returns:
        graph :: an nx graph based on the adjacency matrix
            formed by taking metric(u, v) for each u, v in vectors
    """
    size = len(vectors)

    matrix = _init_square_matrix(size)

    for i in range(size):
        for j in range(size):
            if i != j:
                matrix[i][j] = metric(vectors[i], vectors[j])

    graph = nx.from_numpy_array(matrix)

    return graph


def pagerank(graph):
    """
    args:
        graph :: an nx graph based on the adjacency matrix
            formed by taking metric(u, v) for each u, v in vectors

    returns:
        ranking :: dictionary of form {node: PageRank score} where the
            score is a float representing the page rank of the node.

    raises:
        nx.exception.PowerIterationFailedConvergence if
            the algorithm fails to converge.
    """
    return nx.pagerank(graph)


def textrank(sentences):
    """
    args:
        sentences :: List[str]

    returns:
        indices of sentences in the order of decreasing rank (highest
        ranked sentences have their indices listed before the lowest
        ranked ones)

    raises:
        nx.exception.PowerIterationFailedConvergence if
            the algorithm fails to converge.
    """
    vectors = vectorize(sentences)
    graph = prepare_graph(vectors)
    ranking = pagerank(graph)
    ordered_scored = sorted(((ranking[sent_i], sent_i) for sent_i, _ in enumerate(sentences)), reverse=True)
    ordered_indices = [index for _score, index in ordered_scored]
    return ordered_indices


if __name__ == '__main__':

    sentences = [
        'Sentence one',
        'Sentence two',
        'Sentence three'
    ]

    try:
        print('Rankings are as follows:')
        for rank, sent_i in enumerate(textrank(sentences), start=1):
            print(f'    Rank {rank}: Sentence #{sent_i} ({sentences[sent_i][:20]}...)')
    except PowerIterationFailedConvergence:
        print('-- Diverged')

