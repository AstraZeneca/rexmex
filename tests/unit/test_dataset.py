import unittest

from rexmex.dataset import DatasetReader


class TestErdosRenyiDataset(unittest.TestCase):
    def test_erdos_renyi_structure(self):
        reader = DatasetReader()
        dataset = reader.read_dataset()

        assert dataset.shape[0] == 50378
        assert dataset.shape[1] == 6
        assert {
            "source_group",
            "target_group",
            "source_id",
            "target_id",
            "y_score",
            "y_true",
        }.issubset(dataset.columns)
