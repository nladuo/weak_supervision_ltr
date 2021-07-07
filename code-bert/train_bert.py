import datetime
import pymongo
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
from torch.utils.data import Dataset, DataLoader
from model import Bert
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from metrics import compute_MAP, compute_NDCG, compute_precision
from validation_iterator import ValidDataIterator


def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


DEVICE = torch.device("cuda:0")  # torch.device("cpu"), if you want to run on CPU instead
NUM_EPOCHS = 100
learning_rate = 5e-5
BATCH_SIZE = 9

logging.basicConfig(filename="result2.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

c = pymongo.MongoClient()
db = c.weak_supervision_ltr
query_coll = db.queries
doc_coll = db.docs_with_xml
aol_queries_coll = db.aol_queries_filtered
all_doc_tokens = {}
all_aol_query_tokens = {}

print("loading all docs...")

print("loading all test queries...")
all_queries_map = {}
for q in query_coll.find():
    all_queries_map[str(q["number"])] = q["original_title"]
    # print(q)

# re rank bm25 top 100
with open("../data/test_data_to_re_rank.json", "r") as f:
    valid_data_bm25_top100 = json.load(f)

# with open("../data/test_data_to_re_rank_1000.json", "r") as f:
#     valid_data_bm25_top1000 = json.load(f)

print("loading all docs")
for doc in doc_coll.find({}):
    all_doc_tokens[doc["docNo"]] = doc["wordpieces"]


print("loading all aol_queries")
for q in aol_queries_coll.find({}):
    all_aol_query_tokens[q["query"]] = q["wordpieces"]

def get_tokens(doc_no):
    return all_doc_tokens[doc_no]


def pad_input_with_zero(inputs, length=300):
    if len(inputs) >= length:
        inputs[length-1] = 102  # tokenizer.convert_tokens_to_ids(["[SEP]"]) == 102
        return inputs[:length]
    else:
        inputs = inputs + (length - len(inputs)) * [0]
        return inputs


def pad_segment_ids_with_zero(segment_ids, length=300):
    if len(segment_ids) >= length:
        return segment_ids[:length]
    else:
        segment_ids = segment_ids + (length - len(segment_ids)) * [0]
        return segment_ids


class RobustDataSet(Dataset):
    def __init__(self):
        with open(f'../data/pair_wise_data_top10.json', "r") as f:
            self.pair_wise_data = json.load(f)

    def __len__(self):
        return len(self.pair_wise_data)

    def __getitem__(self, idx):
        data = self.pair_wise_data[idx]

        q_tokens = all_aol_query_tokens[data["q"]]
        q_len = len(q_tokens) + 2
        d1_tokens = all_doc_tokens[data["d1_id"]]
        d2_tokens = all_doc_tokens[data["d2_id"]]

        input1 = tokenizer.convert_tokens_to_ids(["[CLS]"]+q_tokens+["[SEP]"]+d1_tokens+["[SEP]"])
        segments_id1 = [0] * q_len + [1] * (len(d1_tokens) + 1)

        input2 = tokenizer.convert_tokens_to_ids(["[CLS]"]+q_tokens+["[SEP]"]+d2_tokens+["[SEP]"])
        segments_id2 = [0] * q_len + [1] * (len(d2_tokens) + 1)

        label = data["label"]

        input1_ = pad_input_with_zero(input1)
        segments_id1_ = pad_segment_ids_with_zero(segments_id1)
        input2_ = pad_input_with_zero(input2)
        segments_id2_ = pad_segment_ids_with_zero(segments_id2)

        return np.array(input1_), np.array(segments_id1_), \
               np.array(input2_), np.array(segments_id2_), label


train_dataiter = DataLoader(RobustDataSet(), batch_size=BATCH_SIZE, shuffle=True)

total_step = len(train_dataiter.dataset.pair_wise_data)

gradient_accumulation_steps = 12
num_train_steps = 100000
max_grad_norm = 1.0

warmup_steps = 10000

t_total = total_step // gradient_accumulation_steps * num_train_steps

print("start loading bert")
if __name__ == "__main__":
    net = Bert()
    net = net.to(DEVICE)

    criterion = nn.MarginRankingLoss(margin=0.5).cuda(DEVICE)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    model_optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        model_optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # model_optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    now_step = 0
    global_step = 0

    print("start training")

    for ep_idx in range(NUM_EPOCHS):
        train_loss = 0.0
        # for mb_idx in range(EPOCH_SIZE):
        for i, (input1s, segments_id1, input2s, segments_id2, labels) in enumerate(train_dataiter):
            input1s = Variable(input1s).type(torch.LongTensor).to(DEVICE)
            segments_id1 = Variable(segments_id1).type(torch.LongTensor).to(DEVICE)
            input2s = Variable(input2s).type(torch.LongTensor).to(DEVICE)
            segments_id2 = Variable(segments_id2).type(torch.LongTensor).to(DEVICE)
            labels = Variable(labels).type(torch.FloatTensor).to(DEVICE)

            rele_score = net(input1s, segments_id1)
            irrele_score = net(input2s, segments_id2)

            loss = criterion(rele_score, irrele_score, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            train_loss += loss.item()
            now_step += 1

            if now_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)

                model_optimizer.step()
                scheduler.step()
                model_optimizer.zero_grad()

                global_step += 1

                if global_step % 8 == 0:
                    print('Epoch [{}], Step {} [{}/{}], Loss: {:.8f}'
                          .format(ep_idx + 1, now_step, (i + 1) * BATCH_SIZE, total_step,
                                  train_loss / (i+1) * gradient_accumulation_steps))

                # if ((global_step % 80 == 0) and (now_step > 10000)) \
                #         or ((global_step % 160 == 0) and (now_step <= 10000)):
                if global_step % 160 == 0:
                    # evaluation for compute MAP
                    net.eval()
                    valid_result = {}
                    c_1 = 0
                    for qid in valid_data_bm25_top100.keys():
                        inputs_ = []
                        segments_id_ = []
                        d_ids = []

                        q_tokens = tokenizer.tokenize(all_queries_map[qid])
                        for d in valid_data_bm25_top100[qid]:
                            d_id = d["id"]
                            d_text = d["text"]
                            d_tokens = all_doc_tokens[d_id]

                            input_ = tokenizer.convert_tokens_to_ids(["[CLS]"]+q_tokens+["[SEP]"]+d_tokens+["[SEP]"])
                            input_ = pad_input_with_zero(input_)

                            segment_ids_ = [0] * (len(q_tokens) + 2) + [1] * (len(d_tokens) + 1)
                            segment_ids_ = pad_segment_ids_with_zero(segment_ids_)

                            inputs_.append(input_)
                            d_ids.append(d_id)
                            segments_id_.append(segment_ids_)

                        it = ValidDataIterator(inputs_, segments_id_, batch_size=BATCH_SIZE)
                        scores = []
                        for (_inputs_, _segments_id_) in it:
                            inputs_2 = np.array(_inputs_)
                            segments_id_2 = np.array(_segments_id_)
                            out = net(torch.from_numpy(inputs_2).to(DEVICE),
                                        torch.from_numpy(segments_id_2).to(DEVICE))
                            out = out.detach().cpu()
                            scores += [float(out[i]) for i in range(len(_inputs_))]
                            # print(out)
                        # scores = out
                        tmp_list = [{"id": d_ids[i], "score": float(scores[i])} for i in range(len(d_ids))]
                        tmp_list.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                        if c_1 < 5:
                            print(tmp_list[:10])
                        c_1 += 1
                        # print(tmp_list)

                        valid_result[qid] = [a["id"] for a in tmp_list]

                    MAP = compute_MAP(valid_result)
                    NDCG_20 = compute_NDCG(valid_result)
                    NDCG_10 = compute_NDCG(valid_result, length=10)
                    P_20 = compute_precision(valid_result)
                    print(ep_idx, MAP)
                    print(f"MAP@1000: {MAP}")
                    print(f"NDCG@20: {NDCG_20}")
                    print(f"NDCG@10: {NDCG_10}")
                    print(f"P@20: {P_20}")
                    logger.info("Epoch :{epoch_id}, Step:{step}, MAP@1000: {MAP}, P@20: {P}, NDCG@20: {NDCG}".format(
                        epoch_id=ep_idx, MAP=MAP, NDCG=NDCG_20, P=P_20, step=now_step))

                    net.train()

    print_message('Finished')

