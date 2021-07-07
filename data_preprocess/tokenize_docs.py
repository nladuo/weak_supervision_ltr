from transformers import BertTokenizer
from multiprocessing import Pool
import pymongo

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
doc_coll = db.docs_with_xml


def tokenize_docs(part_id):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i, doc in enumerate(doc_coll.find()):
        if i % 5 == part_id:
            wordpieces = tokenizer.tokenize(doc["text"])

            if len(wordpieces) > 1000:
                wordpieces = wordpieces[:1000]

            doc_coll.update({"_id": doc["_id"]}, {
                "$set": {
                    "wordpieces": wordpieces
                }
            })
            print(i, part_id, doc["docNo"])


pool = Pool(5)

for i in range(5):
    pool.apply_async(tokenize_docs, args=(i,))

pool.close()
pool.join()
