import torch
from transformers import BertModel
class BertNerModel(torch.nn.Module):
    def __init__(self,myConfig):
        super().__init__() 
        self.bert = BertModel.from_pretrained(myConfig.bert_path)
        self.num_labels = myConfig.class_num
        self.linear = torch.nn.Linear(myConfig.hidden_size,self.num_labels)
        self.dropout = torch.nn.Dropout(myConfig.dropout)

        self.relu = torch.nn.ReLU()
    def forward(self,input_ids,attention_mask,labels = None):
        output = self.bert(input_ids = input_ids,attention_mask = attention_mask)
        sequence_out = self.dropout(output[0])
        logits = self.linear(sequence_out)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = None
        if labels is not None:
            loss = loss_fn(logits.view(-1,self.num_labels),labels.view(-1))
        return {
            'logits':logits,
            'loss':loss
        }