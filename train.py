import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification
from tqdm import tqdm
from data_pre_ import NERDataset

import swanlab
from metric import id2label_pred,compute,entity_level_f1
from myConfig import myconfig


def train(model):
    train_dataset = NERDataset(myconfig,mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size=myconfig.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    optimizer = torch.optim.Adam(model.parameters(),lr=myconfig.learning_rate)
 
    batch_counter = 1
    for epoch in range(myconfig.epochs):
        total_loss = 0.0
        model.train()
        print(f"Epoch {epoch + 1}/{myconfig.epochs}")

        for batch in tqdm(train_dataloader,desc = '------train------'):
            input_ids = batch['input_ids'].to(myconfig.device)
            attention_mask = batch["attention_mask"].to(myconfig.device)
            labels = batch["labels"].to(myconfig.device)
             #清零梯度(忘了)
            optimizer.zero_grad()
            outputs= model(input_ids = input_ids,attention_mask = attention_mask,labels = labels) 
            #训练需要给真实标签，但不需要预测标签，只需要loss，不计算指标
            train_loss = outputs.loss
           
            train_loss.backward()
            optimizer.step()
            total_loss += train_loss.item()
            swanlab_run.log({'batch_train_loss':train_loss.item()})
            batch_counter += 1

        avg_loss = total_loss / len(train_dataloader)
        
        res = evaluate(model)
        print('Loss:{:8.4f}|p:{:.2f}|R:{:.2f}|F1:{:.2f}'.format(avg_loss,res['p'],res['R'],res['f1']))
        swanlab_run.log({'train_avg_loss':avg_loss,'Dev-Precision':res['p'],'Dev-Recall':res['R'],'Dev-f1':res['f1']})
def evaluate(model):
    dev_dataset = NERDataset(config=myconfig,mode='dev')
    dev_dataloader = DataLoader(dev_dataset,batch_size=myconfig.batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn)
    model.eval()
    #累积标签
    all_true_labels = []
    all_pred_labels = []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader,desc = '------dev------'):
            input_ids = batch['input_ids'].to(myconfig.device)
            attention_mask = batch["attention_mask"].to(myconfig.device)
            labels = batch["labels"].to(myconfig.device)

            #验证的时候要真实标签也需要返回预测标签，要loss，要指标
            outputs = model(input_ids = input_ids,attention_mask = attention_mask,labels = labels) #感觉直接给batch是不是就可以？ 
            pred_ids = torch.argmax(outputs.logits,dim=-1)
            true_labels,pred_labels = id2label_pred(pred_ids=pred_ids,label_ids=labels,id2label=myconfig.id_2_tag)
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

    print("\n[DEBUG] Example of gold vs pred:")
    print("Gold:", len(all_true_labels[0]))
    print("Pred:", len(all_pred_labels[0]))

    #计算所有标签
    # precision,recall,f1 = compute(true_labels=all_true_labels,pred_labels=all_pred_labels)
    precision,recall,f1 = entity_level_f1(y_true=all_true_labels,y_pred=all_pred_labels)
    return {
        'p':precision ,
        'R':recall,
        'f1':f1
        }
    
def test(model):
    test_dataset = NERDataset(config=myconfig,mode='test')
    test_dataloader = DataLoader(test_dataset,batch_size=myconfig.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
    
    #累积标签
    all_true_labels = []
    all_pred_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader,desc = '------test------'):

            input_ids = batch['input_ids'].to(myconfig.device)
            attention_mask = batch["attention_mask"].to(myconfig.device)
            labels = batch["labels"].to(myconfig.device)
            #验证的时候要真实标签也需要返回预测标签，要loss，要指标
            outputs = model(input_ids = input_ids,attention_mask =attention_mask) 

            pred_ids = torch.argmax(outputs.logits,dim=-1)
            #调用评价指标
            true_labels,pred_labels = id2label_pred(pred_ids=pred_ids,label_ids=labels,id2label=myconfig.id_2_tag)
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
    print("\n[DEBUG] Example of gold vs pred:")
    print("Gold:", len(all_true_labels[0]))
    print("Pred:", len(all_pred_labels[0]))
    # precision,recall,f1 = compute(true_labels=all_true_labels,pred_labels=all_pred_labels)
    precision,recall,f1 = entity_level_f1(y_pred=all_pred_labels,y_true=all_true_labels)
    swanlab_run.log({'Test-Precision':precision,'Test-Recall':recall,'Test-f1':f1})
    print('P:{:.2f}|R:{:.2f}|F1:{:.2f}'.format(precision,recall,f1))
    


if __name__ == '__main__':
    model = BertForTokenClassification.from_pretrained(myconfig.bert_path,
                                                   num_labels=myconfig.class_num,
                                                   id2label=myconfig.id_2_tag,
                                                   label2id=myconfig.tag_2_id
                                                   ).to(myconfig.device)
    swanlab_run = swanlab.init(project="NER",experiment_name='命名实体识别')
    train(model)
    test(model)


