import pymongo
from bs4 import BeautifulSoup

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
coll = db.docs_with_xml

coll.create_index([('docNo', pymongo.ASCENDING)])


files = [
    "fbis5.dat",
    "fr94.dat",
    "ft91.dat",
    "ft92.dat",
    "ft93.dat",
    "ft94.dat",
    "latimes.dat",
]

count = 0
for f_name in files:
    print(f_name)
    with open("../data/src-data/{f_name}".format(f_name=f_name), "rb") as f:
        doc = {
            "docNo": "",
            "xml": "",
        }
        for line in f.readlines():
            line = line.decode("utf-8", errors='ignore')
            if line != "\n":
                doc["xml"] += line

                # if doc["docNo"].strip(" ") == "FR941202-2-00138":
                #     print(line)
            if line.startswith("</DOC>"):
                count += 1
                doc["docNo"] = doc["docNo"].strip(" ")
                print(count, doc["docNo"], f_name)

                if coll.find({"docNo": doc["docNo"]}).count() == 0:
                    coll.insert(doc)
                doc = {
                    "docNo": "",
                    "xml": "",
                }
            elif line.startswith("<DOCNO>"):
                doc["docNo"] = line.replace("<DOCNO>", "").replace("</DOCNO>", "").replace("\n", "")
