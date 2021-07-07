import pymongo
import random

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
coll = db.aol_queries_all
coll_shuffled = db.aol_queries_shuffled


query_list = []

count = 0
for q in coll.find():
    print(count, q["query"])
    del q["_id"]
    query_list.append(q)
    count += 1

print("shuffling the query_list.....")
random.shuffle(query_list)


count = 0
for q in query_list:
    coll_shuffled.insert(q)
    print(count, q["query"])
    count += 1
