from collections import Counter
from typing import List


def item_coverage(relevant_items: List, recommendations: List[List]) -> float:
    """
    Calculates the coverage value for items in relevant_items given the collection of recommendations.
    This is defined as the fraction of items that got recommended at least once, divided by all possible items.

    Example:

        item_coverage([1, 2, 3, 4, 5],
              [[1, 2, 3], [2, 3, 4], [1, 2, 4]])

        has 80% coverage. One element out of five was never recommended.

    Args:
        relevant_items (List): items that could be recommended
        recommendations (List[List]): collection of recommendations, each sublist is a separate recommendation,
        e.g. for one user

    Returns:
        coverage (float): Single coverage statistic for the given set of recommendations.

    """
    if len(relevant_items) == 0:
        raise ValueError("relevant_items cannot be empty!")

    rec_item_counter = Counter([x for y in recommendations for x in y])
    total_rec_per_item = [rec_item_counter.get(i, 0) for i in relevant_items]
    return sum([x != 0 for x in total_rec_per_item]) / len(total_rec_per_item)
