from typing import List
from collections import Counter


def item_coverage(relevant_items: List, recommendations: List[List]) -> float:
    """
    Calculates the coverage value for items in relevant_items given the collection of recommendations.
    This is defined as the fraction of items that got recommended at least once, divided by all possible items.

    Example:

        item_coverage([1, 2, 3, 4, 5],
              [[1, 2, 3], [2, 3, 4], [1, 2, 4]])

        has 80% coverage. One element out of five was never recommended.

    Args:
        relevant_items: list of items that could be recommended
        recommendations: collection (list) of recommendations, each sublist is a separate recommendation,
        e.g. for one user

    Returns:
        Single coverage statistic for the given set of recommendations.

    """
    cnt = Counter([x for y in recommendations for x in y])
    rec_totals = [cnt.get(i, 0) for i in relevant_items]
    return sum([x != 0 for x in rec_totals]) / len(rec_totals)
