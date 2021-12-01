import numpy as np


def reciprocal_rank(item: any, ranking: np.array) -> float:
    """
    Calculate the reciprocal rank (RR) of an item.

    Args:
        ranking (Object): an object in the ranking.
        ranking (array-like):  An N x 1 array of items.
    Returns:
        RR (float): The reciprocal rank of the item
    """

    return 1 / (np.where(ranking == item)[0] + 1)
