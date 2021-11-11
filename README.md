![Version](https://badge.fury.io/py/rexmex.svg?style=plastic)
![License](https://img.shields.io/github/license/AZ-AI/rexmex.svg)
[![repo size](https://img.shields.io/github/repo-size/AZ-AI/rexmex.svg)](https://github.com/AZ-AI/rexmex/archive/master.zip)
[![build badge](https://github.com/AZ-AI/rexmex/workflows/CI/badge.svg)](https://github.com/AZ-AI/rexmex/actions?query=workflow%3ACI)

<p align="center">
  <img width="90%" src="https://github.com/AZ-AI/rexmex/blob/main/rexmex_small.jpg?raw=true?sanitize=true" />
</p>

--------------------------------------------------------------------------------

**reXmeX** is recommender system evaluation metric library.

Please look at the **[Documentation](bla)** and **[External Resources](bla)**.

**reXmeX** consists of utilities for recommender system evaluation. First, it provides a comprehensive collection of metrics for the evaluation of recommender systems. Second, it includes a variety of methods for plotting the performance results. Implemented metrics cover a range of data mining ([ICDM](http://icdm2019.bigke.org/), [CIKM](http://www.cikm2019.net/), [KDD](https://www.kdd.org/kdd2020/)) conferences and pieces from prominent journals.

--------------------------------------------------------------------------------

**A simple example**

Text. Bla.

```python
import rexmex

A =
B = 

rec_value = rexmex.metric.recall()
print(rec_value)
```

**Metrics Included with the Specialized Scorecards**

The specialized scorecards with pre-set metrics have the following general categories:

- **Rating**
- **Classification**
- **Ranking**
- **Coverage**


**Rating Metric Scorecard**

* **[Mean Squared Error](docs)**
* **[Root Mean Squared Error](docs)**
* **[Mean Absolute Error](docs)**
* **[Mean Absolute Percentile Error](docs)**
* **[Symmetric Mean Absolute Percentage Error](docs)**
* **[Pearson Correlation](docs)**
* **[Coefficient of Determination](docs)**

**Classification Accuracy Metrics**

* **[Precision](docs)**
* **[Recall](docs)**
* **[Fallout](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Miss Rate](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Inverse Precision](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Inverse Recall](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Markedness](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Informedness](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Matthews Correlation](docs)** from Author *et al.*: [Title](paper_link) (Venue year)

**Rank Accuracy Metrics**

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

**Coverage Metrics**

* **[Prediction Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Weighted Prediction Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Catalog Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Weighted Catalog Coverage](docs)** from Author *et al.*: [Title](paper_link) (Venue year)

Head over to our [documentation](https://rexmex.readthedocs.io) to find out more about installation and data handling, a full list of implemented methods, and datasets. For a quick start, check out our [examples](https://github.com/AZ-AI/rexmex/tree/master/examples/).

If you notice anything unexpected, please open an [issue](https://github.com/AZ-AI/rexmex/issues) and let us know. If you are missing a specific method, feel free to open a [feature request](https://github.com/AZ-AI/rexmex/issues).
We are motivated to constantly make RexMex even better.


--------------------------------------------------------------------------------


**Installation via pip**

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
