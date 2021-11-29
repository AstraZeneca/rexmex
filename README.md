![Version](https://badge.fury.io/py/rexmex.svg?style=plastic)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![repo size](https://img.shields.io/github/repo-size/AstraZeneca/rexmex.svg)](https://github.com/AstraZeneca/rexmex/archive/master.zip)
[![build badge](https://github.com/AstraZeneca/rexmex/workflows/CI/badge.svg)](https://github.com/AstraZeneca/rexmex/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/AstraZeneca/rexmex/branch/main/graph/badge.svg?token=cYgAejRA0Z)](https://codecov.io/gh/AstraZeneca/rexmex)

<p align="center">
  <img width="90%" src="https://github.com/AstraZeneca/rexmex/blob/main/rexmex_small.jpg?raw=true?sanitize=true" />
</p>

--------------------------------------------------------------------------------

**reXmeX** is recommender system evaluation metric library.

Please look at the **[Documentation](https://rexmex.readthedocs.io/en/latest/)** and **[External Resources](https://rexmex.readthedocs.io/en/latest/notes/resources.html)**.

**reXmeX** consists of utilities for recommender system evaluation. First, it provides a comprehensive collection of metrics for the evaluation of recommender systems. Second, it includes a variety of methods for reporting and plotting the performance results. Implemented metrics cover a range of well-known metrics and newly proposed metrics from data mining ([ICDM](http://icdm2019.bigke.org/), [CIKM](http://www.cikm2019.net/), [KDD](https://www.kdd.org/kdd2020/)) conferences and prominent journals.

--------------------------------------------------------------------------------

**An introductory example**

The following example loads a synthetic dataset which has the `source_id`, `target_id`, `source_group` and `target group` keys besides the mandatory `y_true` and `y_scores`.  The dataset has binary labels and predictied probability scores. We read the dataset and define a defult `ClassificationMetric` instance for the evaluation of the predictions. Using this metric set we create a score card, group the predictions on with the `source_group` key and return a performance metric report.

```python
from rexmex.scorecard import ScoreCard
from rexmex.dataset import DatasetReader
from rexmex.metricset import ClassificationMetricSet

reader = DatasetReader()
scores = reader.read_dataset()

metric_set = ClassificationMetricSet()

score_card = ScoreCard(metric_set)

report = score_card.generate_report(scores, groupping=["source_group"])
```

--------------------------------------------------------------------------------

**Scorecard**

A **rexmex** score card allows the reporting of recommender system performance metrics, plotting the performance metrics and saving those. Our framework provides 7 rating, 38 classification, Z ranking, and W coverage metrics.

**Metric Sets**

Metric sets allow the users to calculate a range of evaluation metrics for a label - predicted label vector pair. We provide a general `MetricSet` class and specialized metric sets with pre-set metrics have the following general categories:

- **Rating**
- **Classification**
- **Ranking**
- **Coverage**

--------------------------------------------------------------------------------

**Rating Metric Set**

These metrics assume that items are scored explicitly and ratings are predicted by a regression model. 

* **[Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error)**
* **[Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Mean_squared_error)**
* **[Mean Absolute Error (MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error)**
* **[Mean Absolute Percentage Error (MAPE)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)**


<details>
<summary><b>Expand to see all rating metrics in the metric set.</b></summary>

* **[Symmetric Mean Absolute Percentage Error (SMAPE)](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)**
* **[Pearson Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)**
* **[Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)**

</details>

--------------------------------------------------------------------------------

**Classification Metric Set**

These metrics assume that the items are scored with raw probabilities (these can be binarized).

* **[Precision (or Positive Predictive Value)](https://en.wikipedia.org/wiki/Precision_and_recall)**
* **[Recall (Sensitivity, Hit Rate, or True Positive Rate)](https://en.wikipedia.org/wiki/Precision_and_recall)**
* **[PR AUC](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13140)**
* **[ROC AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)**

<details>
<summary><b>Expand to see all classification metrics in the metric set.</b></summary>

* **[F-1 Score](https://en.wikipedia.org/wiki/F-score)**
* **[Average Precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)**
* **[Specificty](https://en.wikipedia.org/wiki/Precision_and_recall)**
* **[Matthew's Correlation](https://en.wikipedia.org/wiki/Precision_and_recall)**
* **[Accuracy](https://en.wikipedia.org/wiki/Precision_and_recall)**
* **[Balanced Accuracy](https://en.wikipedia.org/wiki/Precision_and_recall)**
* **[Fowlkes-Mallows Index](https://en.wikipedia.org/wiki/Precision_and_recall)**

</details>

--------------------------------------------------------------------------------

**Ranking Metric Set**

* **[DPM](docs)**
* **[NDPM](docs)**
* **[DCG](docs)**
* **[NDCG](docs)**

<details>
<summary><b>Expand to see all ranking metrics in the metric set.</b></summary>

* **[Reciprocal Rank](docs)**
* **[Mean Reciprocal Rank](docs)**
* **[Spearmanns Rho](docs)**
* **[Kendall Tau](docs)**
* **[HITS@k](docs)**
* **[Novelty](docs)**
* **[Average Recall @ k](docs)**
* **[Mean Average Recall @ k](docs)**
* **[Average Precision @ k](docs)**
* **[Mean Average Precision @ k](docs)**
* **[Personalisation](docs)**
* **[Intra List Similarity](docs)**

</details>

--------------------------------------------------------------------------------

**Coverage Metric Set**

These merics measure how well the recommender system covers the available items in the catalog. In other words measure the diversity of predictions.

* **[Prediction Coverage](docs)**
* **[Weighted Prediction Coverage](docs)**
* **[Catalog Coverage](docs)**
* **[Weighted Catalog Coverage](docs)**


--------------------------------------------------------------------------------
**Documentation and Reporting Issues**

Head over to our [documentation](https://rexmex.readthedocs.io) to find out more about installation and data handling, a full list of implemented methods, and datasets. For a quick start, check out our [examples](https://github.com/AstraZeneca/rexmex/tree/master/examples/).

If you notice anything unexpected, please open an [issue](https://github.com/AstraZeneca/rexmex/issues) and let us know. If you are missing a specific method, feel free to open a [feature request](https://github.com/AstraZeneca/rexmex/issues).
We are motivated to constantly make RexMex even better.

--------------------------------------------------------------------------------

**Installation via the command line**

RexMex can be installed with the following command after the repo is cloned.

```sh
$ python setup.py install
```

**Installation via pip**

RexMex can be installed with the following pip command.

```sh
$ pip install rexmex
```

As we create new releases frequently, upgrading the package casually might be beneficial.

```sh
$ pip install rexmex --upgrade
```

--------------------------------------------------------------------------------

**Running tests**

```sh
$ pytest ./tests/unit -cov rexmex/
$ pytest ./tests/integration -cov rexmex/
```

--------------------------------------------------------------------------------

**License**

- [Apache-2.0 License](https://github.com/AZ-AI/rexmex/blob/master/LICENSE)