import pymongo
from elasticsearch import Elasticsearch
import json


es = Elasticsearch()
client = pymongo.MongoClient()
db = client.weak_supervision_ltr
query_coll = db.queries
doc_coll = db.docs_with_xml


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
                "text": q["original_title"]
            }
        },
        "explain": True
    }

    searched = es.search("robo04_index", doc_type="docs", body=query_contains, size=100)
    print(q["number"], q["title"], len(rank_labels[str(q["number"])]),
          " ; ", len(searched["hits"]["hits"]))
    result[str(q["number"])] = []

    for hit in searched["hits"]["hits"]:
        doc = doc_coll.find_one({"docNo": hit["_source"]["id"]})
        text = doc["text"]
        result[str(q["number"])].append({
            "id": hit["_source"]["id"],
            "text": text,
        })


with open("../data/test_data_to_re_rank_100.json", "w") as f:
    json.dump(result, f)

