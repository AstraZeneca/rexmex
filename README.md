![Version](https://badge.fury.io/py/rexmex.svg?style=plastic)
![License](https://img.shields.io/github/license/AZ-AI/rexmex.svg)
[![repo size](https://img.shields.io/github/repo-size/AZ-AI/rexmex.svg)](https://github.com/AZ-AI/rexmex/archive/master.zip)
[![build badge](https://github.com/AZ-AI/rexmex/workflows/CI/badge.svg)](https://github.com/AZ-AI/rexmex/actions?query=workflow%3ACI)

<p align="center">
  <img width="90%" src="https://github.com/AZ-AI/rexmex/blob/main/rexmex_logo.jpeg?raw=true?sanitize=true" />
</p>


**Metrics included**

In detail, the following evluation metrics were included:

* **[Metric](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Metric](docs)** from Author *et al.*: [Title](paper_link) (Venue year)
* **[Metric](docs)** from Author *et al.*: [Title](paper_link) (Venue year)


Head over to our [documentation](https://rexmex.readthedocs.io) to find out more about installation and data handling, a full list of implemented methods, and datasets. For a quick start, check out our [examples](https://github.com/AZ-AI/rexmex/tree/master/examples/).

If you notice anything unexpected, please open an [issue](https://github.com/AZ-AI/rexmex/issues) and let us know. If you are missing a specific method, feel free to open a [feature request](https://github.com/AZ-AI/rexmex/issues).
We are motivated to constantly make RexMex even better.


--------------------------------------------------------------------------------

**Installation**

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
