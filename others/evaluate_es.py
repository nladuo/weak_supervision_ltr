import pymongo
from elasticsearch import Elasticsearch
import json
import time
import random
import math
import numpy as np
from metrics import get_ndcg


es = Elasticsearch()
client = pymongo.MongoClient()
db = client.weak_supervision_ltr
query_coll = db.queries

with open("../data/rank_labels.json", "r") as f:
    rank_labels = json.load(f)

queries = []
for q in query_coll.find():
    queries.append(q)
    print(q)


result = {}
for count, q in enumerate(queries):
    if str(q["number"]) not in rank_labels.keys():
        continue
    if count < 200:
        continue
    query_contains = {
        "query": {
            "match": {
                "text": q["title"]
            }
        },
        "explain": True
    }
    print(q["number"], q["title"], len(rank_labels[str(q["number"])]))
    # searched = es.search("robo04_index2", doc_type="docs", body=query_contains, size=len(rank_labels[str(q["number"])]))
    searched = es.search("robo04_index", doc_type="docs", body=query_contains, size=1000)

    result[str(q["number"])] = []

    for hit in searched["hits"]["hits"]:
        result[str(q["number"])].append(hit["_id"])


def compute_MAP(rank_labels, result):
    sum_AP = 0
    count = 0
    for qid in result.keys():
        if len(rank_labels[qid]) == 0:
            continue
        AP = 0
        relevant_count = 0
        for i, docId in enumerate(result[qid]):
            if docId in rank_labels[qid]:
                relevant_count += 1
                AP += float(relevant_count) / (i + 1)

        AP /= len(rank_labels[qid])

        print(count, qid, "-->", AP)
        sum_AP += AP
        count += 1

    return sum_AP / count


def compute_NDCG(rank_labels, result, length=20):
    sum_NDCG = 0
    count = 0
    for qid in result.keys():
        if len(rank_labels[qid]) == 0:
            continue
        label_list = []

        for i, docId in enumerate(result[qid]):
            if docId in rank_labels[qid]:
                label_list.append(rank_labels[qid][docId])
            else:
                label_list.append(0)

        NDCG = get_ndcg(r=label_list, k=length)
        sum_NDCG += NDCG
        count += 1

    return sum_NDCG / count


def compute_Recall(rank_labels, result):
    sum_recall = 0
    count = 0
    for qid in result.keys():
        if len(rank_labels[qid]) == 0:
            continue
        # P = 0
        relevant_count = 0
        for i, docId in enumerate(result[qid]):
            if docId in rank_labels[qid]:
                relevant_count += 1

        recall = relevant_count / len(rank_labels[qid])

        print(count, qid, "-->", recall)
        sum_recall += recall
        count += 1

    return sum_recall / count


def compute_precision(rank_labels, result, length):
    sum_precision = 0
    count = 0
    for qid in result.keys():
        if len(rank_labels[qid]) == 0:
            continue
        # P = 0
        relevant_count = 0
        _list = result[qid][:length]
        for i, docId in enumerate(_list):
            if docId in rank_labels[qid]:
                relevant_count += 1

        precision = relevant_count / length

        # print(count, qid, "-->", precision)
        sum_precision += precision
        count += 1

    return sum_precision / count


print("MAP : ", compute_MAP(rank_labels, result))
print("NDCG@20 : ", compute_NDCG(rank_labels, result, 20))
print("NDCG@10 : ", compute_NDCG(rank_labels, result, 10))
print("P@20 : ", compute_precision(rank_labels, result, 20))
print("Recall : ", compute_Recall(rank_labels, result))
