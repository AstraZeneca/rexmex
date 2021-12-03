import numpy as np

from rexmex.utils import normalize


def test_normalize():
    @normalize
    def dummy(*args):
        y_true = args[0]
        y_score = args[1]
        return y_true, y_score

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.3, 0.5, 0.5, 0.7, 1.0])

    y_true_norm, y_score_norm = dummy(y_true, y_score)

    np.testing.assert_array_equal(y_true_norm, np.array([-1, -1, -1, 1, 1, 1]))
    np.testing.assert_allclose(y_score_norm, np.array([-0.8, -0.4, 0.0, 0.0, 0.4, 1.0]))
