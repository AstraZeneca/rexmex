import pytest
import unittest

from sklearn.metrics import precision_score, recall_score

class TestMetrics(unittest.TestCase):
    def __init__(self):
        self.x = 2

    def test_representation(self):
        assert self.x == 2