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
    version="0.0.9",
    license="Apache License, Version 2.0",
    description="",
    author="AstraZeneca BIKG Team",
    author_email="bikg@astrazeneca.com",
    url="https://github.com/AstraZeneca/rexmex",
    download_url="https://github.com/AstraZeneca/rexmex/archive/v_00009.tar.gz",
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
