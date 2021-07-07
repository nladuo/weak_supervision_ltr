from elasticsearch import Elasticsearch
import pymongo

es = Elasticsearch()

index_mappings = {
    "mappings": {
        "docs": {
            "properties": {
                "text": {
                    "type": "text",
                }
            }
        }
    }
}

es.indices.delete(index='robo04_index')

if es.indices.exists(index='robo04_index') is not True:
    print("create robo04_index")
    es.indices.create(index='robo04_index', body=index_mappings)


client = pymongo.MongoClient()
db = client.weak_supervision_ltr
coll = db.docs4


docs = []
for doc in coll.find():
    docs.append(doc)


for count, doc in enumerate(docs):
    _id = str(doc["docNo"])
    text = doc["text"]

    doc = {
        "id": _id,
        "text": text,
    }
    res = es.index(index="robo04_index", doc_type="docs", id=_id, body=doc)
    print(count, res)

