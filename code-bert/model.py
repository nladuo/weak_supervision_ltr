import torch.nn as nn
# from transformers import BertModel
from transformers import BertTokenizer, BertModel


class Bert(nn.Module):

    def __init__(self):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(768, 1)

    def forward(self, input_ids, segments_ids):
        attention_mask = (input_ids != 0)
        bert_out = self.bert(input_ids=input_ids,
                             token_type_ids=segments_ids,
                             attention_mask=attention_mask)[1]
        # out = self.fc1(bert_out)
        out = self.fc2(bert_out)

        return out

