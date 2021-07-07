from transformers import BertTokenizer
from multiprocessing import Pool
import pymongo

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
query_coll = db.aol_queries_filtered


def tokenize_docs(part_id):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for i, q in enumerate(query_coll.find()):
        if i % 10 == part_id:
            wordpieces = tokenizer.tokenize(q["query"])

            if len(wordpieces) > 1000:
                wordpieces = wordpieces[:1000]

            query_coll.update({"_id": q["_id"]}, {
                "$set": {
                    "wordpieces": wordpieces
                }
            })
            print(i, part_id, q["query"])


pool = Pool(10)

for i in range(10):
    pool.apply_async(tokenize_docs, args=(i,))

pool.close()
pool.join()

