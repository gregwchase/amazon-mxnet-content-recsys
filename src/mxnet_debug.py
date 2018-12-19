import mxnet as mx
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


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

if __name__ == "__main__":

    articles = pd.read_csv("../data/articles_lemmatized_20181125.csv",
        usecols=["id", "title"])

    tfidf_matrix = load_npz("../data/tfidf_matrix_all_articles_20181217.npz")

    mx_tfidf = mx.nd.sparse.array(tfidf_matrix, ctx=mx.cpu())

    mx_recsys = mx.nd.sparse.dot(mx_tfidf, mx_tfidf.T)

    # Get the article recommendations
    article_recs = get_recommendations(df_articles=articles,
        article_idx=3, mx_mat=mx_recsys, n_recs=10)
