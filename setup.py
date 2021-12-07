from setuptools import find_packages, setup

install_requires = ["numpy", "sklearn", "pandas", "scipy"]


setup_requires = ["pytest-runner"]


tests_require = ["pytest", "pytest-cov", "mock", "unittest"]


keywords = [
    "recommender",
    "nmf",
    "matrix factorisation",
    "system",
    "evaluation metric",
    "ranking",
    "recsys",
    "metric",
    "ranking metric",
    "performance metric",
]


setup(
    name="rexmex",
    packages=find_packages(),
    version="0.0.11",
    license="Apache License, Version 2.0",
    description="A General Purpose Recommender Metrics Library for Fair Wvaluation.",
    author="Benedek Rozemberczki, Sebastian Nilsson, Piotr Grabowski, Charles Tapley Hoyt, Gavin Edwards",
    author_email="bikg@astrazeneca.com",
    url="https://github.com/AstraZeneca/rexmex",
    download_url="https://github.com/AstraZeneca/rexmex/archive/v_00011.tar.gz",
    keywords=keywords,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
    ],
)
