import pymongo
from elasticsearch import Elasticsearch
import json
from multiprocessing import Pool
import random
import hashlib


def get_md5(_str):
    hl = hashlib.md5()
    _bytes = _str.encode("utf-8")
    hl.update(_bytes)
    return hl.hexdigest()


es = Elasticsearch()
client = pymongo.MongoClient()
db = client.weak_supervision_ltr
query_coll = db.aol_queries_shuffled

pair_wise_data_coll = db.pair_wise_data_top10

with open("../data/docIds.json", "r") as f:
    docIds = json.load(f)

queries = []
count = 0
# for q in query_coll.find({"is_set": False}):
for q in query_coll.find({}):
    queries.append(q)


def choose_negative_sample(relevant_ids):
    global docIds
    res = random.choice(docIds)
    while res in relevant_ids:
        res = random.choice(docIds)

    return res


def extract_two_random_id():
    res1 = random.choice([a for a in range(0, 5)])
    res2 = random.choice([a for a in range(5, 10)])

    return res1, res2


def extract_one_random_id():
    return random.choice([a for a in range(0, 10)])


def insert_one_query(count, q):
    while True:
        try:
            query_contains = {
                "query": {
                    "match": {
                        "text": q["query"]
                    }
                },
                "explain": True
            }
            searched = es.search("robo04_index", doc_type="docs", body=query_contains, size=10)
            break
        except:
            pass

    relevant_ids = []
    for i, hit in enumerate(searched["hits"]["hits"]):
        relevant_ids.append(hit["_id"])

    if len(relevant_ids) != 10:
        return
    print(count, q["query"])

    positive_id1 = relevant_ids[extract_one_random_id()]
    negative_id1 = choose_negative_sample(relevant_ids)

    id1, id2 = extract_two_random_id()
    positive_id2 = relevant_ids[id1]
    negative_id2 = relevant_ids[id2]

    pair_wise_data_coll.insert(
        {"q": q["query"], "d1_id": positive_id1, "d2_id": negative_id1, "label": 1}
    )
    pair_wise_data_coll.insert(
        {"q": q["query"], "d1_id": positive_id2, "d2_id": negative_id2, "label": 1}
    )

    query_coll.update({"_id": q["_id"]}, {
        "$set": {
            "is_set": True
        }
    })


pool = Pool(10)  # use 10 core

for count, q in enumerate(queries):
    pool.apply_async(insert_one_query, args=(count, q))

pool.close()
pool.join()
