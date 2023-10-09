import numpy as np
import pandas as pd
import gc
from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight, BM25Recommender
from implicit.als import AlternatingLeastSquares
from implicit.cpu.bpr import BayesianPersonalizedRanking
from implicit.recommender_base import RecommenderBase
from implicit import evaluation
from utils import pandas_df_to_csr
import json
import statistics

# Specify model weights
BPR_WEIGHT = 1
ALS_WEIGHT = 1
BM25_WEIGHT = 1
K = 10

# Load dataset
amazon_beauty_df = pd.read_csv("ratings_Beauty.csv")
user_map, item_map, amazon_beauty_csr = pandas_df_to_csr(amazon_beauty_df)

# Test-Train Split
train_csr, test_csr = evaluation.train_test_split(amazon_beauty_coo_bm25, train_percentage=0.8, random_state=55)
print(f"Train size: {train_csr.size} \n Test size: {test_csr.size}")
test_coo = test_csr.tocoo()

# Load pre-trained models
ALS_model = AlternatingLeastSquares.load("4_CF_ALS_implicit")
BPR_model = BayesianPersonalizedRanking.load("5_CF_BPR_implicit")
BM25_model = BM25Recommender.load("6_BM25")

# Create dictionary to store recommedations and other meta data of each model
results = {}
results["BPR"] = {"weight": BPR_WEIGHT}
results["ALS"] = {"weight": ALS_WEIGHT}
results["BM25"] = {"weight": BM25_WEIGHT}
results["ENSEMBLE"] = {"weight": 1}


# Ineference for each user in test set and calculate evaluation metrics
actual_dict = {}
eval_dict = {"precision": [], "recall": [], "f1_score": []}
for user_id, product_id, rating in zip(test_coo.row, test_coo.col, test_coo.data):
    # print(f"Processing: user_id: {user_id}, product_id: {product_id}, rating: {rating}")
    print(user_id, end = ', ')
    # Retrieve actual products and ratings
    if user_id in actual_dict:
        actual_dict[user_id].append(product_id)
    else:
        actual_dict[user_id] = [product_id]

    # Get recommendation from each model
    ids, scores = BPR_model.recommend(user_id, test_csr[user_id], N=K, filter_already_liked_items=False)
    results["BPR"][user_id] = {"product_ids": list(ids), "scores": list(scores)}
    
    ids, scores = ALS_model.recommend(user_id, test_csr[user_id], N=K, filter_already_liked_items=False)
    results["ALS"][user_id] = {"product_ids": list(ids), "scores": list(scores)}

    ids, scores = BM25_model.recommend(user_id, test_csr[user_id], N=K, filter_already_liked_items=False)
    results["BM25"][user_id] = {"product_ids": list(ids), "scores": list(scores)}

    # Ensemble results
    ensemble_results = {}
    for _model, _results in results.items():
        if _model == "ENSEMBLE":
            continue
        scores = _results[user_id]['scores']
        # Check if all product scores are equal or empty
        if len(set(scores)) <= 1:
            print(f"Skipping Model: {_model}, User Id: {user_id}")
            continue
        # Score is normalized to range 0-1 and then weighted by the specified model weight 
        normalized_scores = normalize(arr=scores, t_max=_results['weight'])
        # print("Normalized scores: ", normalized_scores)
        for id, score in zip(_results[user_id]['product_ids'], normalized_scores):
            # Case where product is already recommended by one or more other models
            if id in ensemble_results:
                ensemble_results[id] += score # Add the score to the previous value
                ensemble_results[id] /= 2 # Average the score (This is a rough average)
            # Case where product is recommended first time by current model
            else:
                ensemble_results[id] = score
        
    ensemble_results = sort_by_score(ensemble_results)
    results["ENSEMBLE"][user_id] = {"product_ids": list(ensemble_results.keys()), "scores": list(ensemble_results.values())}

    # Evaluate ensemble results
    actual_products = actual_dict[user_id]
    ensemble_products = results["ENSEMBLE"][user_id]["product_ids"]
    eval_dict["precision"].append(precision_at_k(actual_products, ensemble_products, K))
    eval_dict["recall"].append(recall_at_k(actual_products, ensemble_products, K))
    eval_dict["f1_score"].append(f1_acore_at_k(actual_products, ensemble_products, K))


print("Ensemble recommendation evaluation results -------")
print("Precision at K: ", statistics.fmean(eval_dict["precision"]))
print("Recall at K: ", statistics.fmean(eval_dict["recall"]))
print("F1 score at K: ", statistics.fmean(eval_dict["f1_score"]))
