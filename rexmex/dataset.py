import io

import pandas as pd
from six.moves import urllib


class DatasetReader(object):
    """Class to read synthetic test datasets."""

    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/AstraZeneca/rexmex/main/dataset/"

    def read_dataset(self, dataset: str = "erdos_renyi_example"):
        """Method to read the dataset.
        Args:
            dataset (str): Dataset of interest, one of:
                (:obj:`"erdos_renyi_example"`). Default is 'erdos_renyi_example'.
        Returns:
            data (pd.DataFrame): The example dataset for testing the library.
        """
        assert dataset in ["erdos_renyi_example"], "Wrong dataset."
        path = self.base_url + dataset + ".csv"
        bytes = urllib.request.urlopen(path).read()
        data = pd.read_csv(io.BytesIO(bytes), encoding="utf8", sep=",")
        return data
