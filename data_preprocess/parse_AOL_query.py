import pymongo
import string
from nltk.corpus import stopwords
from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')


StopWords = set(stopwords.words('english') + list(string.punctuation))

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
coll = db.aol_queries_all

print(coll.remove({}))


def is_continue(query):
    not_contain = ["http", "www.", ".com", ".net", ".org", ".edu", ".gov", ".html"]

    for term in not_contain:
        if term in query:
            return True

    not_contain_terms = ["www", "com"]
    for term in not_contain_terms:
        if term in query.split(" "):
            return True

    return False

c_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


def parse_one_collection(part_id, f_path):
    global count
    with open(f_path, "r") as f:
        for line in f.readlines():
            parts = line.split("\t")
            if parts[0] != "AnonID":

                query = parts[1].lower()

                if is_continue(query):
                    continue

                if coll.find({"query": query}).count() == 0:
                    coll.insert({
                        "query": query,
                        "count": 1,
                        "is_set": False
                    })
                    count += 1
                    print(count, part_id, query)
                else:
                    q_count = coll.find_one({"query": query})["count"]
                    coll.update({"query": query}, {
                        "$set": {
                            "count": q_count + 1
                        }
                    })

count = 0

pool = Pool(15)

for c_id in c_list:
    f_path = "../data/AOL-user-ct-collection/user-ct-test-collection-{c_id}.txt".format(c_id=c_id)

    pool.apply_async(parse_one_collection, args=(c_id, f_path,))

pool.close()
pool.join()


