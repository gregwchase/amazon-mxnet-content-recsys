# Content Based Filtering With MXNet

This is the repository for the `Content Based Recommendation System` article created for the MXNet Medium blog, in partnership with Amazon.

## Prerequisites

* [Kaggle API](https://github.com/Kaggle/kaggle-api)
* [SpaCy](https://spacy.io/usage/)

## Instructions - Library Installation

If the prerequisites aren't already installed, run the following commands.

### _Libraries_

```markdown
Kaggle API - pip install kaggle

Spacy
pip install -U spacy

python -m spacy download en

python -m spacy download en_core_web_md
```

## Data

* All news article data is downloaded via `download_articles.sh`.

### Download Data

You must have your Kaggle API credentials to proceed with the download!

```bash
bash download_articles.sh
```