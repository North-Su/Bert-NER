
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.metrics.sequence_labeling import get_entities

def entity_level_f1(y_true, y_pred):
    TP= 0
    pred_entity_all = 0
    true_entity_all = 0
    for true_seq, pred_seq in zip(y_true, y_pred):
        pred_entity = set(get_entities(pred_seq))
        true_entity = set(get_entities(true_seq))
        TP += len(pred_entity & true_entity)
        # FN += len(true_entity - pred_entity)
        # FP += len(pred_entity - true_entity)
        
        pred_entity_all += len(pred_entity)
        true_entity_all += len(true_entity)
    
    P = TP / pred_entity_all + 1e-10
    R = TP / true_entity_all + 1e-10
    
    # print('预测实体',pred_entity)
    # print('真实实体',true_entity)
    # P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    # R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P,R,F1

def id2label_pred(pred_ids, label_ids, id2label):
    preds_labels, true_labels = [], []
    pred_list = pred_ids.detach().cpu().tolist()
    label_list = label_ids.detach().cpu().tolist()
    for pred, true in zip(pred_list, label_list):
        preds, labels = [], []
        for p, t in zip(pred, true):
            if t != -100:  
                preds.append(id2label[p])
                labels.append(id2label[t])
        preds_labels.append(preds)
        true_labels.append(labels)
    return true_labels,preds_labels        

#计算指标

def compute(pred_labels,true_labels):
    precision = precision_score(y_true=true_labels,y_pred=pred_labels)
    recall = recall_score(y_true=true_labels,y_pred=pred_labels)
    f1 = f1_score(y_true=true_labels,y_pred=pred_labels) 
    return precision,recall,f1
