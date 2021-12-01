import numpy as np


def reciprocal_rank(relevant_item: any, ranking: np.array) -> float:
    """
    Calculate the reciprocal rank (RR) of an item in a ranked list of items.

    Args:
        item (Object): an object in the list of items.
        ranking (array-like):  An N x 1 ranking of items.
    Returns:
        RR (float): The reciprocal rank of the item
    """

    return 1.0 / (np.in1d(ranking, relevant_item).argmax() + 1)


def mean_reciprocal_rank(relevant_items: np.array, ranking: np.array):
    """
    Calculate the mean reciprocal rank (MRR) of items in a ranked list.

    Args:
        relevant_items (array-like): An N x 1 array of relevant items.
        ranking (array-like):  An N x 1 array of ordered items.
    Returns:
        MRR (float): The mean reciprocal rank of the relevant items in a ranking.
    """

    reciprocal_ranks = []
    for item in relevant_items:
        rr = reciprocal_rank(item, ranking)
        reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks)


def average_percision_at_k(relevant_items: np.array, ranking: np.array, k=10):
    """
    Calculate the average percision at k (AP@K) of items in a ranked list.

    Args:
        relevant_items (array-like): An N x 1 array of relevant items.
        ranking (array-like):  An N x 1 array of ordered items.
    Returns:
        AP@K (float): The average percision @ k of a ranking.
    """

    if len(ranking) > k:
        ranking = ranking[:k]

    hits = np.in1d(ranking, relevant_items)
    ranks = np.arange(1, k + 1)
    aps = np.arange(1, len(ranks[hits]) + 1) / ranks[hits]
    return np.mean(aps)
