import pymongo
import json

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
coll = db.docs_with_xml


docIds = []


for doc in coll.find({}):
    docIds.append(doc["docNo"])

with open("../data/docIds.json", "w") as f:
    json.dump(docIds, f)
