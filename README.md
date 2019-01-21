# Content Based Filtering With MXNet

This is the repository for the `Content-Based Recommendation Systems with Apache MXNet` article created for the [MXNet Medium blog](https://medium.com/apache-mxnet), in partnership with [Amazon](https://aws.amazon.com/mxnet/).

## Prerequisites

* [Kaggle API](https://github.com/Kaggle/kaggle-api)
* [MXNet](https://mxnet.apache.org/versions/master/install/index.html?platform=Linux&language=Python&processor=CPU)
* [SpaCy](https://spacy.io/usage/)

## Instructions - Library Installation

If the prerequisites aren't already installed, run the following commands.

### _Libraries_

```markdown
Kaggle API - pip install kaggle

MXNet - pip install mxnet

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

## References

[Amazon - Item to Item Collaborative Filtering](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)

[Create Custom Stop-Word List](https://stackoverflow.com/questions/36369870/sklearn-how-to-add-custom-stopword-list-from-txt-file)

[Kaggle - All The News Data Set](https://www.kaggle.com/snapcrack/all-the-news)

[Kaggle API](https://github.com/Kaggle/kaggle-api)

[Keras shoot-out: TensorFlow vs MXNet](https://medium.com/@julsimon/keras-shoot-out-tensorflow-vs-mxnet-51ae2b30a9c0)

[Machine Learning :: Cosine Similarity for Vector Space Models (Part III)](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)

[MXNet - Sparse NDArray API](https://mxnet.incubator.apache.org/api/python/ndarray/sparse.html#sparse-ndarray-api)

[MXNet - Tutorials](https://mxnet.incubator.apache.org/versions/master/tutorials/index.html)

[Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize)

[Scikit-Learn: TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

[spaCy](https://spacy.io/)

[Stemming and Lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)