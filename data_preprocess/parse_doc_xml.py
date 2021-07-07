import pymongo
from bs4 import BeautifulSoup


client = pymongo.MongoClient()
db = client.weak_supervision_ltr
coll = db.docs_with_xml

not_care_tags = [
    "docno",
    "ht",
    "date1",
    "parent",
    "date",
    "correction-date"
]

count = 0
for doc in coll.find({}):
    count += 1
    soup = BeautifulSoup(doc["xml"], "lxml").find("doc")
    print(count, doc["docNo"])

    # print(soup)

    text = ""
    for child in soup.children:
        if child.name is not None:

            if child.name not in not_care_tags:
                text += child.get_text() + "\n"
    coll.update({"_id": doc["_id"]}, {
        "$set": {
            "text": text
        }
    })

