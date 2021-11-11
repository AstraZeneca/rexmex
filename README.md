![Version](https://badge.fury.io/py/rexmex.svg?style=plastic)
![License](https://img.shields.io/github/license/AstraZeneca/rexmex.svg)
[![repo size](https://img.shields.io/github/repo-size/AstraZeneca/rexmex.svg)](https://github.com/AstraZeneca/rexmex/archive/master.zip)
[![build badge](https://github.com/AstraZeneca/rexmex/workflows/CI/badge.svg)](https://github.com/AstraZeneca/rexmex/actions?query=workflow%3ACI)

<p align="center">
  <img width="90%" src="https://github.com/AstraZeneca/rexmex/blob/main/rexmex_small.jpg?raw=true?sanitize=true" />
</p>

--------------------------------------------------------------------------------

**reXmeX** is recommender system evaluation metric library.

Please look at the **[Documentation](https://rexmex.readthedocs.io/en/latest/)** and **[External Resources](bla)**.

**reXmeX** consists of utilities for recommender system evaluation. First, it provides a comprehensive collection of metrics for the evaluation of recommender systems. Second, it includes a variety of methods for reporting plotting the performance results. Implemented metrics cover a range of well-known metrics and newly proposed metrics ones from data mining ([ICDM](http://icdm2019.bigke.org/), [CIKM](http://www.cikm2019.net/), [KDD](https://www.kdd.org/kdd2020/)) conferences and pieces from prominent journals.

--------------------------------------------------------------------------------

**A simple example**

The following example loads a synthetic dataset which has the `source_id`, `target_id`, `source_group` and `target group` keys besides the mandatory `y_true` and `y_scores`.  The dataset has binary labels and predictied probability scores. We read the dataset and define a defult `ClassificationMetric` instance for the evaluation of the predictions. Using this we create a score card, group the predictions on the `source_group` key and return a performance metric report.

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

## Metrics Included with the Specialized Scorecards

The specialized scorecards with pre-set metrics have the following general categories:

- **Rating**
- **Classification**
- **Ranking**
- **Coverage**

## Rating Metric Scorecard

These metrics assume that items are scored with explicit ratings and these ratings are predicted by a regression model. 

* **[Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)**
* **[Root Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error)**
* **[Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error)**
* **[Mean Absolute Percentage Error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)**
* **[Symmetric Mean Absolute Percentage Error](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)**
* **[Pearson Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)**
* **[Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)**

##  Classification Metric Scorecard

* **[Precision](docs)**
* **[Recall](docs)**
* **[PR AUC](docs)**
* **[ROC AUC](docs)**
* **[F-1 Score](docs)**
* **[Average Precision](docs)**

* **[Miss Rate](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Inverse Precision](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Inverse Recall](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Markedness](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Informedness](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Matthews Correlation](docs)** from Author *et al.*: [Title](paper_link) (Venue year)

## Ranking Metric Scorecard

* **[DPM](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[NDPM](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[DCG](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[NDCG](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Reciprocal Rank](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Mean Reciprocal Rank](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Spearmanns Rho](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Kendall Tau](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[HITS@k](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Novelty](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Average Recall @ k](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Mean Average Recall @ k](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Average Precision @ k](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Mean Average Precision @ k](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Personalisation](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Intra List Similarity](docs)** from Author *et al.*: [Title](paper_link) (Venue year)

## Coverage Metric Scorecard

* **[Prediction Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Weighted Prediction Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Catalog Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Weighted Catalog Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)

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
