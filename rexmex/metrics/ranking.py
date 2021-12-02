from math import log2
from typing import List
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from sklearn.metrics import dcg_score, ndcg_score


def reciprocal_rank(relevant_item: any, ranking: np.array) -> float:
    """
    Calculate the reciprocal rank (RR) of an item in a ranked list of items.

    Args:
        item (Object): an object in the list of items.
        ranking (array-like):  An N x 1 ranking of items.
    Returns:
        RR (float): The reciprocal rank of the item.
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
        k (int): the number of items considered in the ranking.
    Returns:
        AP@K (float): The average percision @ k of a ranking.
    """

    if len(ranking) > k:
        ranking = ranking[:k]

    hits = np.in1d(ranking, relevant_items)
    ranks = np.arange(1, k + 1)
    p_at_ks = np.arange(1, len(ranks[hits]) + 1) / ranks[hits]
    return np.mean(p_at_ks)


def average_recall_at_k(relevant_items: List, ranking: List, k: int = 10):
    """
    Calculate the average recall at k (AR@K) of items in a ranked list.

    Args:
        relevant_items (array-like): An N x 1 array of relevant items.
        ranking (array-like):  An N x 1 array of items.
        k (int): the number of items considered in the ranking.
    Returns:
        AR@K (float): The average percision @ k of a ranking.
    """
    if len(ranking) > k:
        ranking = ranking[:k]
    else:
        k = len(ranking)

    num_hits = 0.0
    score = 0.0

    for i, item in enumerate(ranking):
        if item in relevant_items and item not in ranking[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(relevant_items)


def mean_average_recall_at_k(relevant_items: List[list], rankings: List[list], k: int = 10):
    """
    Calculate the mean average recall at k (MAR@K) for a list of recommendations.
    Each recommendation should be paired with a list of relevant items. First recommendation list is
    evaluated against the first list of relevant items, and so on.

    Args:
        relevant_items (array-like): An M x R list where M is the number of recommendation lists,
                                     and R is the number of relevant items.
        rankings (array-like):  An M x N list where M is the number of recommendation lists and
                                N is the number of recommended items.
        k (int): the number of items considered in the recommendation.
    Returns:
        MAR@K (float): The mean average recall @ k across the recommendations.

    """
    ars = []
    for items, ranking in zip(relevant_items, rankings):
        ar = average_recall_at_k(items, ranking, k)
        ars.append(ar)

    return np.mean(ars)


def mean_average_percision_at_k(relevant_items: np.array, rankings: np.array, k: int = 10):
    """
    Calculate the mean average percision at k (MAP@K) for a list of rankings.
    Each ranking should be paired with a list of relevant items. First ranking list is
    evaluated against the first list of relevant items, and so on.

    Example usage:
    .. code-block:: python

        import numpy as np
        from rexmex.metrics.ranking import mean_average_percision_at_k

        mean_average_percision_at_k(
            relevant_items=np.array(
                [
                    [1,2],
                    [2,3]
                ]
            ),
            rankings=np.array([
                [3,2,1],
                [2,1,3]
            ])
        )
        >>> 0.708333...

    Args:
        relevant_items (array-like): An M x N array of relevant items.
        rankings (array-like):  An M x N array of ranking arrays.
        k (int): the number of items considered in the rankings.
    Returns:
        MAP@K (float): The mean average percision @ k across recommendations.
    """

    aps = []
    for items, ranking in zip(relevant_items, rankings):
        ap = average_percision_at_k(items, ranking, k)
        aps.append(ap)

    return np.mean(aps)


def hits_at_k(relevant_items: np.array, ranking: np.array, k=10):
    """
    Calculate the number of hits of relevant items in a ranked list HITS@K.

    Args:
        relevant_items (array-like): An 1 x N array of relevant items.
        rankings (array-like):  An 1 x N array of ranking arrays
        k (int): the number of items considered in the ranking
    Returns:
        HITS@K (float):  The number of relevant items in the first k items in a ranking.
    """
    if len(ranking) > k:
        ranking = ranking[:k]

    hits = np.array(np.in1d(ranking, relevant_items), dtype=int).sum()
    return hits / len(ranking)


def spearmanns_rho(list_a: np.array, list_b: np.array):
    """
    Calculate the Spearmann's rank correlation coefficient (Spearmann's rho) between two arrays.

    Args:
        list_a (array-like): An 1 x N array of items.
        list_b (array-like):  An 1 x N array of items.
    Returns:
        Spearmann's rho (float): Spearmann's rho.
        p-value (float): two-sided p-value for null hypothesis that both rankings are uncorrelated.
    """
    return stats.spearmanr(list_a, list_b)


def kendall_tau(ranking_a: np.array, ranking_b: np.array):
    """
    Calculate the Kendall's tau, measuring the correspondance between two rankings.

    Args:
        ranking_a (array-like): An 1 x N array of items.
        ranking_b (array-like):  An 1 x N array of items.
    Returns:
        Kendall tau (float): The tau statistic.
        p-value (float): two-sided p-value for null hypothesis that there's no association between the rankings.
    """
    return stats.kendalltau(ranking_a, ranking_b)


def intra_list_similarity(rankings: List[list], items_feature_matrix: np.array):
    """
    Calculate the intra list similarity of recommended items. The items
    are represented by feature vectors, which compared with cosine similarity.
    The ranking consists of item indices, which are used to fetch the item
    features.

    Args:
        rankings (List[list]): A M x N array of rankings, where M is the number
                               of rankings and N the number of recommended items
        features (matrix-link): A N x D matrix, where N is the number of items and D the
                                number of features representing one item

    Returns:
        (float): Average intra list similarity across rankings
    """

    intra_list_similarities = []
    for ranking in rankings:
        ranking_features = items_feature_matrix[ranking]
        similarity = cosine_similarity(ranking_features)
        upper_right = np.triu_indices(similarity.shape[0], k=1)
        avg_similarity = np.mean(similarity[upper_right])
        intra_list_similarities.append(avg_similarity)

    return np.mean(intra_list_similarities)


def personalization(rankings: List[list]):
    """
    Calculates personalization, a measure of similarity between recommendations.
    A high value indicates that the recommendations are disimillar, or "personalized".

    Args:
        rankings (List[list]): A M x N array of rankings, where M is the number
                               of rankings and N the number of recommended items

    Returns:
        (float): personalization
    """

    n_rankings = len(rankings)

    # map each ranked item to index
    item2ix = {}
    counter = 0
    for ranking in rankings:
        for item in ranking:
            if item not in item2ix:
                item2ix[item] = counter
                counter += 1

    n_items = len(item2ix.keys())

    # create matrix of rankings x items
    items_matrix = np.zeros((n_rankings, n_items))

    for i, ranking in enumerate(rankings):
        for item in ranking:
            item_ix = item2ix[item]
            items_matrix[i][item_ix] = 1

    similarity = cosine_similarity(X=items_matrix)
    dim = similarity.shape[0]
    personalization = (similarity.sum() - dim) / (dim * (dim - 1))

    return 1 - personalization


def novelty(rankings: List[list], item_popularities: dict, num_users: int, k: int = 10):
    """
    Calculates the capacity of the recommender system to to generate novel
    and unexpected results.

    Args:
        rankings (List[list]): A M x N array of rankings, where M is the number
                               of rankings and N the number of recommended items
        item_popularities (dict): A dict mapping each item in the recommendations to a popularity value.
                                  Popular items have higher values.
        num_users (int): The number of users
        k (int): The number of items considered in each recommendation.

    Returns:
        (float): novelty
    """

    epsilon = 1e-10
    all_self_information = []
    for ranking in rankings:
        self_information_sum = 0
        for i in range(k):
            item = ranking[i]
            item_pop = item_popularities[item]
            self_information_sum += -log2((item_pop + epsilon) / num_users)

        avg_self_information = self_information_sum / k
        all_self_information.append(avg_self_information)

    return np.mean(all_self_information)


def NDPM(actual_ranking: List, system_ranking: List):
    """
    Calculates the Normalized Distance-based Performance Measure (NPDM) between two
    ranked lists. Two matching rankings return 0.0 while two unmatched rankings
    returns 1.0.

    Args:
        actual_ranking (List): List of items
        system_ranking (List): The predicted list of items

    Returns:
        NDPM (float): Normalized Distance-based Performance Measure

    Yao, Y. Y. "Measuring retrieval effectiveness based on user preference of documents."
    Journal of the American Society for Information science 46.2 (1995): 133-145.
    """
    assert set(actual_ranking) == set(system_ranking)

    item_actual_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(actual_ranking))}
    item_system_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(system_ranking))}

    items = set(actual_ranking)

    item_combinations = itertools.combinations(items, 2)

    C_minus = 0
    C_plus = 0
    C_u = 0

    for item1, item2 in item_combinations:
        item1_actual_rank = item_actual_rank[item1]
        item2_actual_rank = item_actual_rank[item2]

        item1_system_rank = item_system_rank[item1]
        item2_system_rank = item_system_rank[item2]

        C = np.sign(item1_system_rank - item2_system_rank) * np.sign(item1_actual_rank - item2_actual_rank)

        C_u += C ** 2

        if C < 0:
            C_minus += 1
        else:
            C_plus += 1

    C_u0 = C_u - (C_plus + C_minus)

    NDPM = (C_minus + 0.5 * C_u0) / C_u
    return NDPM


def DCG(y_true: np.array, y_score: np.array):
    """
    Computes the Discounted Cumulative Gain (DCG), a sum of the true scores ordered
    by the predicted scores, and then penalized by a logarithmic discount based on ordering.

    Args:
        y_true (array-like): An N x M array of ground truth values, where M > 1 for multilabel classification problems.
        y_score (array-like): An N x M array of predicted values, where M > 1 for multilabel classification problems..
    Returns:
        DCG (float): Discounted Cumulative Gain
    """
    return dcg_score(y_true, y_score)


def NDCG(y_true: np.array, y_score: np.array):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG), a sum of the true scores ordered
    by the predicted scores, and then penalized by a logarithmic discount based on ordering.
    The score is normalized between [0.0, 1.0]

    Args:
        y_true (array-like): An N x M array of ground truth values, where M > 1 for multilabel classification problems.
        y_score (array-like): An N x M array of predicted values, where M > 1 for multilabel classification problems..
    Returns:
        NDCG (float) : Normalized Discounted Cumulative Gain
    """
    return ndcg_score(y_true, y_score)
