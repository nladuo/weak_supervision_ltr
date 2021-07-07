import pymongo
import re

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
query_coll = db.aol_queries_shuffled
query_coll2 = db.aol_queries_filtered

count = 0
satisfied_count = 0
q_len_map = {}



for q in query_coll.find({}):
    q_len = len(q["query"].split(" "))
    if q_len in q_len_map.keys():
        q_len_map[q_len] += 1
    else:
        q_len_map[q_len] = 1

    count += 1
    if (q_len <= 5) and (q_len >= 2):
        if q["count"] > 1:
            query_coll2.insert(q)
            satisfied_count += 1

    if count % 1000 == 0:
        print(count, satisfied_count)


print(q_len_map)
