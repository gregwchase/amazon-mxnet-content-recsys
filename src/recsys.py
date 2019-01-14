import pandas as pd
import mxnet as mx
import numpy as np
# import os
import glob
import gensim
from gensim.utils import simple_preprocess
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

class PreprocessText():

	def __init__(self):
		self.additional_stop_words = {"-PRON-"}
		self.stop_words = set(STOP_WORDS.union(self.additional_stop_words))

	def make_bigrams(self, texts):
		"""
		Create bigrams from documents.
		Higher thresholds yield fewer phrases
		"""
		bigram = gensim.models.Phrases(texts, min_count=2, threshold=10)
		bigram_mod = gensim.models.phrases.Phraser(bigram)
		return [bigram_mod[doc] for doc in texts]

	def lemmatization(self, texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
		"""
		Tokenize and lemmatize all documents. The following criteria are used to evaluate each word.
				Is the token a stop word?
				Is the token comprised of letters?
				Is the token longer than 1 letter?
				Is the token an allowed POS tag?
				Is the lemmatized token a stop word?

		"""
		print("Lemmatizing Text")

		# Initialize spaCy
		nlp = spacy.load('en_core_web_md', disable=["parser", "ner"])

		texts_out = []

		for text in texts:
			doc = nlp(text)
			texts_out.append([token.lemma_ for token in doc
							  if not token.is_stop
							  and token.lemma_ not in self.stop_words
							  and token.is_alpha
							  and len(token) > 1
							  and token.pos_ in allowed_postags])

			if len(texts_out) % 1000 == 0:
				print("Lemmatized {0} of {1} documents".format(
					len(texts_out), len(texts)))

		return texts_out

def get_recommendations(df_articles, article_idx, mx_mat, n_recs=10):
    """
    Request top N article recommendations.

    INPUT
        df_articles: Pandas DataFrame containing all articles.
        user_id: User ID being provided matches.
        mx_mat: MXNet cosine similarity matrix
    OUTPUT
        Pandas DataFrame of top N article recommendations.
    """

    # user_idx = article_idx

    article_sims = mx_mat[article_idx].asnumpy()
    article_recs = np.argsort(-article_sims)[:n_recs + 1]

    # Top recommendations
    df_recs = df_articles.loc[list(article_recs)]
    df_recs["similarity"] = article_sims[article_recs]

    return df_recs

if __name__ == '__main__':

	file_path = "../data/"
	all_files = glob.glob(file_path + "*.csv")

	extract_features = lambda f : pd.read_csv(f, usecols = ["id", "title", "publication", "content"])

	# Concatenate features across all files into single DataFrame
	articles = pd.concat((extract_features(f) for f in all_files))

	articles = articles.head(1000)

	tf = TfidfVectorizer(analyzer="word",
						ngram_range=(1, 3),
						min_df=0.2, # ignore terms with document frequency lower than 0.2 (20%)
						stop_words="english")

	# Create TF-IDF Matrix
	mx_tfidf = tf.fit_transform(articles["content"])

	# Convert TF-IDF Matrix to MXNet format
	mx_tfidf = mx.nd.sparse.array(mx_tfidf, ctx=mx.cpu())

	# Compute cosine similarities via dot product
	mx_recsys = mx.nd.sparse.dot(mx_tfidf, mx_tfidf.T)

	article_recs = get_recommendations(df_articles = articles,
		article_idx = 3, mx_mat = mx_recsys, n_recs=10)

	pprint(article_recs)

	"""
	PROCESS ENTIRE DATASET
	"""

	prep = PreprocessText()

	# Lemmatize articles
	articles["articles_lemmatized"] = prep.lemmatization(articles["content"].values)

	# Convert each article from a list of strings to single string
	articles["articles_lemmatized"] = articles["articles_lemmatized"].apply(" ".join)

	tfidf_matrix = tf.fit_transform(articles["articles_lemmatized"])

	mx_tfidf = mx.nd.sparse.array(tfidf_matrix, ctx=mx.cpu())

	mx_recsys = mx.nd.sparse.dot(mx_tfidf, mx_tfidf.T)
