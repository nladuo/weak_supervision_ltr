import json

result = {}

with open("./data/qrels.robust2004.txt", "r") as f:
    f.readline()
    for line in f.readlines():
        print(line)
        items = line.split(" ")
        if len(items) == 4:
            qid = items[0]
            docNo = items[2]
            score = int(items[3])

            if qid not in result.keys():
                result[qid] = {}

            if score != 0:
                result[qid][docNo] = score


with open("../data/rank_labels.json", "w") as f:
    json.dump(result, f)
