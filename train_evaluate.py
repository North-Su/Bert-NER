import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification,get_linear_schedule_with_warmup
from tqdm import tqdm
from data_pre_ import NERDataset
from model import BertNerModel
import swanlab
from metric import id2label_pred,compute,entity_level_f1
from myConfig import my_Config
import time
import os

def train(model):
    
    train_dataset = NERDataset(myconfig,mode='train')
    train_dataloader = DataLoader(train_dataset,batch_size=myconfig.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    optimizer = torch.optim.Adam(model.parameters(),lr=myconfig.learning_rate)
    #加warm_up 
    num_training_steps = len(train_dataloader) * myconfig.epochs #训练的所有步数
    # 选取所有步数的10%做warmup ,学习率会从0上升到2e-5，后面再慢慢下降直至为0
    num_warmup_steps = int(0.1 * num_training_steps)  
    #
    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
    )
    step = 1
    best_f1 = 0.0
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
            train_loss = outputs['loss']
           
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            swanlab_run.log({
                "train/learning_rate": current_lr
            })
            total_loss += train_loss.item()
            if step % 30 == 0:
                swanlab_run.log({'train/batch_loss':train_loss.item()})
            step += 1

        avg_loss = total_loss / len(train_dataloader)
        #验证部分
        res = evaluate(model)
        print('Loss:{:8.4f}|p:{:.2f}|R:{:.2f}|F1:{:.2f}'.format(avg_loss,res['p'],res['R'],res['f1']))
        swanlab_run.log({'train/avg_loss':avg_loss,'Dev/Precision':res['p'],'Dev/Recall':res['R'],'Dev/f1':res['f1']})
        #保留f1值最大时候的模型参数
        if res['f1'] > best_f1:
            best_f1 = res['f1']
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/'+myconfig.dataset_type+'_'+myconfig.model_type+'_best_model.pth')
            print(f"Saved new best model with F1={best_f1:.4f}")

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
            pred_ids = torch.argmax(outputs['logits'],dim=-1)
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
    checkpoints_path = 'checkpoints/'+myconfig.dataset_type+'_'+myconfig.model_type+'_best_model.pth'
    if os.path.exists(checkpoints_path):
        model.load_state_dict(torch.load(checkpoints_path))
        print(f"已加载保存权重")
    else:
        print(f"权重文件不存在")
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

            pred_ids = torch.argmax(outputs['logits'],dim=-1)
            #调用评价指标
            true_labels,pred_labels = id2label_pred(pred_ids=pred_ids,label_ids=labels,id2label=myconfig.id_2_tag)
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
    print("\n[DEBUG] Example of gold vs pred:")
    print("Gold:", len(all_true_labels[0]))
    print("Pred:", len(all_pred_labels[0]))
    # precision,recall,f1 = compute(true_labels=all_true_labels,pred_labels=all_pred_labels)
    precision,recall,f1 = entity_level_f1(y_pred=all_pred_labels,y_true=all_true_labels)
    swanlab_run.log({'Test/Precision':precision,'Test/Recall':recall,'Test/f1':f1})
    print('P:{:.2f}|R:{:.2f}|F1:{:.2f}'.format(precision,recall,f1))
    


if __name__ == '__main__':
    # model = BertForTokenClassification.from_pretrained(myconfig.bert_path,
    #                                                num_labels=myconfig.class_num,
    #                                                id2label=myconfig.id_2_tag,
    #                                                label2id=myconfig.tag_2_id
    #                                                ).to(myconfig.device)
    myconfig = my_Config()
    model = BertNerModel(myConfig=myconfig).to(myconfig.device)
    swanlab_run = swanlab.init(project="NER",
                               experiment_name='NER'+myconfig.dataset_type+myconfig.model_type
                               +time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    train(model)
    test(model)


