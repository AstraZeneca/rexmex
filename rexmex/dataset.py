import io
import pandas as pd
from six.moves import urllib

class DatasetReader(object):
    r"""Class to read benchmark datasets.
    Args:
        dataset (str): Dataset of interest, one of:
            (:obj:`"erdos_renyi_example"`). Default is 'erdos_renyi_example'.
    """

    def __init__(self):
        self.base_url = "https://github.com/AstraZeneca/rexmex/raw/master/dataset/"

    def read_dataset(self, dataset: str = "erdos_renyi_example"):
        """
        Reading the dataset from the web.
        """
        assert dataset in ["erdos_renyi_example"], "Wrong dataset."
        path = self.base_url + dataset + ".csv"
        bytes = urllib.request.urlopen(path).read()
        data = pd.read_csv(io.BytesIO(bytes), encoding="utf8", sep=",", dtype={"switch": np.int32})
        return data