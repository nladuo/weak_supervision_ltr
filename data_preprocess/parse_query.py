import pymongo

client = pymongo.MongoClient()
db = client.weak_supervision_ltr
coll = db.queries

TYPE_NONE = -1
TYPE_TITLE = 0
TYPE_DESC = 1
TYPE_NARR = 2


index = 0
status = TYPE_NONE
query = {
    "title": "",
    "narr": "",
    "desc": ""
}

with open("../data/04.testset", "r") as f:
    for line in f.readlines():

        if line.startswith("</top>"):
            query["title"] = query["title"].strip(" ")
            query["narr"] = query["narr"].strip(" ")
            query["desc"] = query["desc"].strip(" ")
            print(query)

            if coll.find({"number": query["number"]}).count() == 0:
                query["title"] = query["title"].replace("\n", "")
                coll.insert(query)
            query = {
                "title": "",
                "narr": "",
                "desc": ""
            }
            status = TYPE_NONE
        elif line.startswith("<num>"):
            query["number"] = int(line.replace("\n", "").replace("<num> Number:", ""))
        elif line.startswith("<title>"):
            status = TYPE_TITLE
            query["title"] += line.replace("<title>", "").replace("\n", " ")
        elif line.startswith("<desc>"):
            status = TYPE_DESC
        elif line.startswith("<narr>"):
            status = TYPE_NARR
        else:
            if line != "\n":
                if status == TYPE_TITLE:
                    query["title"] += line.replace("\n", " ")
                elif status == TYPE_DESC:
                    query["desc"] += line.replace("\n", " ")
                elif status == TYPE_NARR:
                    query["narr"] += line.replace("\n", " ")


