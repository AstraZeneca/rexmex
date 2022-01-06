from typing import List, Tuple, Union

import numpy as np


def user_coverage(
    possible_users_items: Tuple[List[Union[int, str]], List[Union[int, str]]],
    recommendations: List[Tuple[Union[int, str], Union[int, str]]],
) -> float:
    """
    Calculates the coverage value for users in possible_users_items[0] given the collection of recommendations.
    Recommendations over users/items not in possible_users_items are discarded.

    Args:
        possible_users_items (Tuple[List[Union[int, str]], List[Union[int, str]]]): contains exactly TWO sub-lists,
        first one with users, second with items
        recommendations (List[Tuple[Union[int, str], Union[int, str]]]): contains user-item recommendation tuples,
        e.g. [(user1, item1),(user2, item2),]

    Returns: user coverage (float): a metric showing the fraction of users who got at least one recommendation out
    of all possible users.
    """
    if len(possible_users_items) != 2:
        raise ValueError("possible_users_items must be of length 2: [users, items]")

    if np.any([len(x) == 0 for x in possible_users_items]):
        raise ValueError("possible_users_items cannot hold empty lists!")

    possible_users = set(possible_users_items[0])
    users_with_recommendations = set([x[0] for x in recommendations])
    users_without_recommendations = possible_users.difference(users_with_recommendations)
    user_cov = 1 - len(users_without_recommendations) / len(possible_users)

    return round(user_cov, 3)


def item_coverage(
    possible_users_items: Tuple[List[Union[int, str]], List[Union[int, str]]],
    recommendations: List[Tuple[Union[int, str], Union[int, str]]],
) -> float:
    """
    Calculates the coverage value for items in possible_users_items[1] given the collection of recommendations.
    Recommendations over users/items not in possible_users_items are discarded.

    Args:
        possible_users_items (Tuple[List[Union[int, str]], List[Union[int, str]]]): contains exactly TWO sub-lists,
        first one with users, second with items
        recommendations (List[Tuple[Union[int, str], Union[int, str]]]): contains user-item recommendation tuples,
        e.g. [(user1, item1),(user2, item2),]

    Returns: item coverage (float): a metric showing the fraction of items which got recommended at least once.
    """
    if len(possible_users_items) != 2:
        raise ValueError("possible_users_items must be of length 2: [users, items]")

    if np.any([len(x) == 0 for x in possible_users_items]):
        raise ValueError("possible_users_items cannot hold empty lists!")

    possible_items = set(possible_users_items[1])
    items_with_recommendations = set([x[1] for x in recommendations])
    items_without_recommendations = possible_items.difference(items_with_recommendations)
    item_cov = 1 - len(items_without_recommendations) / len(possible_items)

    return round(item_cov, 3)
