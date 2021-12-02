from math import log2
from typing import List
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from sklearn.metrics import dcg_score, ndcg_score


def reciprocal_rank(relevant_item: any, predicted: List) -> float:
    """
    Calculate the reciprocal rank (RR) of an item in a ranked list of items.

    Args:
        relevant_item (any): a target item in the predicted list of items.
        predicted (array-like): An N x 1 predicted of items.
    Returns:
        RR (float): The reciprocal rank of the item.
    """

    assert relevant_item in predicted

    for i, item in enumerate(predicted):
        if item == relevant_item:
            return 1.0 / (i + 1.0)


def mean_reciprocal_rank(actual: List, predicted: List):
    """
    Calculate the mean reciprocal rank (MRR) of items in a ranked list.

    Args:
        actual (array-like): An N x 1 array of relevant items.
        predicted (array-like):  An N x 1 array of ordered items.
    Returns:
        MRR (float): The mean reciprocal rank of the relevant items in a predicted.
    """

    reciprocal_ranks = []
    for item in actual:
        rr = reciprocal_rank(item, predicted)
        reciprocal_ranks.append(rr)

    return np.mean(reciprocal_ranks)


def average_percision_at_k(actual: np.array, predicted: np.array, k=10):
    """
    Calculate the average percision at k (AP@K) of items in a ranked list.

    Args:
        actual (array-like): An N x 1 array of relevant items.
        predicted (array-like):  An N x 1 array of ordered items.
        k (int): the number of items considered in the predicted list.
    Returns:
        AP@K (float): The average percision @ k of a predicted list.
    """

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    hits = 0.0
    for i, item in enumerate(predicted):
        if item in actual and item not in predicted[:i]:
            hits += 1.0
            score += hits / (i + 1.0)

    return score / hits


def average_recall_at_k(actual: List, predicted: List, k: int = 10):
    """
    Calculate the average recall at k (AR@K) of items in a ranked list.

    Args:
        actual (array-like): An N x 1 array of relevant items.
        predicted (array-like):  An N x 1 array of items.
        k (int): the number of items considered in the predicted list.
    Returns:
        AR@K (float): The average percision @ k of a predicted list.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    else:
        k = len(predicted)

    num_hits = 0.0
    score = 0.0

    for i, item in enumerate(predicted):
        if item in actual and item not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / len(actual)


def mean_average_recall_at_k(actual: List[list], predicted: List[list], k: int = 10):
    """
    Calculate the mean average recall at k (MAR@K) for a list of recommendations.
    Each recommendation should be paired with a list of relevant items. First recommendation list is
    evaluated against the first list of relevant items, and so on.

    Args:
        actual (array-like): An M x R list where M is the number of recommendation lists,
                                     and R is the number of relevant items.
        predicted (array-like):  An M x N list where M is the number of recommendation lists and
                                N is the number of recommended items.
        k (int): the number of items considered in the recommendation.
    Returns:
        MAR@K (float): The mean average recall @ k across the recommendations.

    """
    ars = []
    for items, predicted in zip(actual, predicted):
        ar = average_recall_at_k(items, predicted, k)
        ars.append(ar)

    return np.mean(ars)


def mean_average_percision_at_k(actual: List[list], predicted: List[list], k: int = 10):
    """
    Calculate the mean average percision at k (MAP@K) across predicted lists.
    Each prediction should be paired with a list of relevant items. First predicted list is
    evaluated against the first list of relevant items, and so on.

    Example usage:
    .. code-block:: python

        import numpy as np
        from rexmex.metrics.predicted import mean_average_percision_at_k

        mean_average_percision_at_k(
            actual=np.array(
                [
                    [1,2],
                    [2,3]
                ]
            ),
            predicted=np.array([
                [3,2,1],
                [2,1,3]
            ])
        )
        >>> 0.708333...

    Args:
        actual (array-like): An M x N array of relevant items.
        predicted (array-like):  An M x N array of predicted items.
        k (int): the number of items considered in the predicted list.
    Returns:
        MAP@K (float): The mean average percision @ k across recommendations.
    """

    aps = []
    for items, predicted in zip(actual, predicted):
        ap = average_percision_at_k(items, predicted, k)
        aps.append(ap)

    return np.mean(aps)


def hits_at_k(actual: np.array, predicted: np.array, k=10):
    """
    Calculate the number of hits of relevant items in a ranked list HITS@K.

    Args:
        actual (array-like): An 1 x N array of relevant items.
        predicted (array-like):  An 1 x N array of predicted arrays
        k (int): the number of items considered in the predicted list
    Returns:
        HITS@K (float):  The number of relevant items in the first k items of a prediction.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    hits = np.array(np.in1d(predicted, actual), dtype=int).sum()
    return hits / len(predicted)


def spearmanns_rho(actual: np.array, predicted: np.array):
    """
    Calculate the Spearmann's rank correlation coefficient (Spearmann's rho) between two lists.

    Args:
        actual (array-like): An 1 x N array of items.
        predicted (array-like):  An 1 x N array of items.
    Returns:
        Spearmann's rho (float): Spearmann's rho.
        p-value (float): two-sided p-value for null hypothesis that both predicted are uncorrelated.
    """
    return stats.spearmanr(actual, predicted)


def kendall_tau(actual: np.array, predicted: np.array):
    """
    Calculate the Kendall's tau, measuring the correspondance between two lists.

    Args:
        actual (array-like): An 1 x N array of items.
        predicted (array-like):  An 1 x N array of items.
    Returns:
        Kendall tau (float): The tau statistic.
        p-value (float): two-sided p-value for null hypothesis that there's no association between the predicted.
    """
    return stats.kendalltau(actual, predicted)


def intra_list_similarity(recommendations: List[list], items_feature_matrix: np.array):
    """
    Calculate the intra list similarity of recommended items. The items
    are represented by feature vectors, which compared with cosine similarity.
    The predicted consists of item indices, which are used to fetch the item
    features.

    Args:
        recommendations (List[list]): A M x N array of predicted, where M is the number
                               of predicted and N the number of recommended items
        features (matrix-link): A N x D matrix, where N is the number of items and D the
                                number of features representing one item

    Returns:
        (float): Average intra list similarity across predicted
    """

    intra_list_similarities = []
    for predicted in recommendations:
        predicted_features = items_feature_matrix[predicted]
        similarity = cosine_similarity(predicted_features)
        upper_right = np.triu_indices(similarity.shape[0], k=1)
        avg_similarity = np.mean(similarity[upper_right])
        intra_list_similarities.append(avg_similarity)

    return np.mean(intra_list_similarities)


def personalization(recommendations: List[list]):
    """
    Calculates personalization, a measure of similarity between recommendations.
    A high value indicates that the recommendations are disimillar, or "personalized".

    Args:
        recommendations (List[list]): A M x N array of predicted items, where M is the number
                                    of predicted lists and N the number of items

    Returns:
        (float): personalization
    """

    n_predictions = len(recommendations)

    # map each ranked item to index
    item2ix = {}
    counter = 0
    for prediction in recommendations:
        for item in prediction:
            if item not in item2ix:
                item2ix[item] = counter
                counter += 1

    n_items = len(item2ix.keys())

    # create matrix of predicted x items
    items_matrix = np.zeros((n_predictions, n_items))

    for i, prediction in enumerate(recommendations):
        for item in prediction:
            item_ix = item2ix[item]
            items_matrix[i][item_ix] = 1

    similarity = cosine_similarity(X=items_matrix)
    dim = similarity.shape[0]
    personalization = (similarity.sum() - dim) / (dim * (dim - 1))

    return 1 - personalization


def novelty(recommendations: List[list], item_popularities: dict, num_users: int, k: int = 10):
    """
    Calculates the capacity of the recommender system to to generate novel
    and unexpected results.

    Args:
        recommendations (List[list]): A M x N array of items, where M is the number
                               of predicted lists and N the number of recommended items
        item_popularities (dict): A dict mapping each item in the recommendations to a popularity value.
                                  Popular items have higher values.
        num_users (int): The number of users
        k (int): The number of items considered in each recommendation.

    Returns:
        (float): novelty
    """

    epsilon = 1e-10
    all_self_information = []
    for recommendations in recommendations:
        self_information_sum = 0
        for i in range(k):
            item = recommendations[i]
            item_pop = item_popularities[item]
            self_information_sum += -log2((item_pop + epsilon) / num_users)

        avg_self_information = self_information_sum / k
        all_self_information.append(avg_self_information)

    return np.mean(all_self_information)


def NDPM(actual: List, predicted: List):
    """
    Calculates the Normalized Distance-based Performance Measure (NPDM) between two
    ordered lists. Two matching orderings return 0.0 while two unmatched orderings returns 1.0.

    Args:
        actual (List): List of items
        predicted (List): The predicted list of items

    Returns:
        NDPM (float): Normalized Distance-based Performance Measure

    Reference:
    Yao, Y. Y. "Measuring retrieval effectiveness based on user preference of documents."
    Journal of the American Society for Information science 46.2 (1995): 133-145.
    """
    assert set(actual) == set(predicted)

    item_actual_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(actual))}
    item_predicted_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(predicted))}

    items = set(actual)

    item_combinations = itertools.combinations(items, 2)

    C_minus = 0
    C_plus = 0
    C_u = 0

    for item1, item2 in item_combinations:
        item1_actual_rank = item_actual_rank[item1]
        item2_actual_rank = item_actual_rank[item2]

        item1_pred_rank = item_predicted_rank[item1]
        item2_pred_rank = item_predicted_rank[item2]

        C = np.sign(item1_pred_rank - item2_pred_rank) * np.sign(item1_actual_rank - item2_actual_rank)

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
