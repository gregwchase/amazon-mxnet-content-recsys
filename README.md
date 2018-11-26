# Content Based Filtering With MXNet

This is the repository for the `Content Based Recommendation System` article created for the MXNet Medium blog, in partnership with Amazon.

## Prerequisites

* [Gensim](https://radimrehurek.com/gensim/install.html)
* [Kaggle API](https://github.com/Kaggle/kaggle-api)
* [SpaCy](https://spacy.io/usage/)

## Instructions - Library Installation

If the prerequisites aren't already installed, run the following commands.

### _Gensim Installation_

```markdown
pip install gensim
```

### _Kaggle API Installation_

```markdown
pip install kaggle
```

### _SpaCy Installation_

```markdown
pip install -U spacy

python -m spacy download en

python -m spacy download en_core_web_md
```

## Data

* Articles via the Kaggle CLI

### Download Data

You must have your Kaggle API credentials to proceed with the download!

```bash
bash download_articles.sh
```