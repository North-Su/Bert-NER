#分割数据集
from sklearn.model_selection import train_test_split
with open(r'/home/sulin/NER/data/msra/train.txt','r',encoding='UTF-8') as f:
    text = f.read().strip() 
sentance = text.split('\n\n')

train_dataset,dev_dataset = train_test_split(sentance,train_size=0.8,test_size=0.2,random_state=42)

#把重新分好的数据集写成.txt文件
with open('train.txt','w',encoding='utf-8') as f:
    f.write("\n\n".join(train_dataset))
with open('dev.txt','w',encoding='utf-8') as f:
    f.write("\n\n".join(dev_dataset))

