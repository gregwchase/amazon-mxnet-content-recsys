import glob
from pprint import pprint

import mxnet as mx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys import get_recommendations, extract_features
from recsys import PreprocessText

if __name__ == "__main__":
	file_path = "../data/"
	all_files = glob.glob(file_path + "*.csv")

	# Concatenate features across all files into single DataFrame
	articles = pd.concat((extract_features(f) for f in all_files))

	articles = articles.head(50000)

	"""
	PROCESS ENTIRE DATASET
	"""

	prep = PreprocessText()

	# Lemmatize articles
	articles["articles_lemmatized"] = prep.lemmatization(articles["content"].values)

	# Convert each article from a list of strings to single string
	articles["articles_lemmatized"] = articles["articles_lemmatized"].apply(" ".join)

	tf = TfidfVectorizer(analyzer="word",
						 ngram_range=(1, 3),
						 min_df=0.2, # ignore terms with document frequency lower than 0.2 (20%)
						 stop_words="english")

	tfidf_matrix = tf.fit_transform(articles["articles_lemmatized"])

	mx_tfidf = mx.nd.sparse.array(tfidf_matrix, ctx=mx.cpu())

	mx_recsys = mx.nd.sparse.dot(mx_tfidf, mx_tfidf.T)

	article_recs = get_recommendations(df_articles=articles,
						article_idx=3, mx_mat=mx_recsys, n_recs=10)

	pprint(article_recs)
