import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
class NERDataset(Dataset):
    #读文件并对特征和标签进行处理
    def __init__(self,config,mode):
        super().__init__()
        #读取文件
        self.tokenizer = BertTokenizerFast.from_pretrained(config.bert_path)
        self.mode = mode
        self.texts,self.labels = self.read_file(config.dataset_path+ config.dataset_name + self.mode + config.data_name)
        self.tag_to_id = config.tag_2_id
        self.max_length = config.max_length
        
    def read_file(self,path):
        texts, labels = [], []
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read().strip()  # 读出整个文件内容
            sentences = content.split("\n\n")  # 用空行分句
            for sent in sentences:
                words, tags = [], []
                for line in sent.split("\n"):
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    words.append(parts[0])
                    tags.append(parts[1])
                texts.append(words)
                labels.append(tags)
        return texts, labels

    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        texts = self.texts[index]
        labels = self.labels[index]
        return{
            'texts':texts,
            'labels':labels
        }
    
    def align_label(self,labels,word_ids):
        new_labels_id = []
       
        current_id = None
        for word_id in word_ids:
            if word_id != current_id:
                current_id = word_id
                label = -100 if word_id is None else self.tag_to_id[labels[word_id]]
                new_labels_id.append(label)    
            elif word_id is None:
                new_labels_id.append(-100)   
            else:
                label = labels[word_id]
                #如果是B-开始的，转换成I-
                if label.startswith('B-') and isinstance(label,str):
                    label = 'I-' + label[2:]     
                    new_labels_id.append(self.tag_to_id[label])    
                #如果是I-开始或者O，就直接添加这个标签
                else:
                    new_labels_id.append(self.tag_to_id[label])   
        return new_labels_id

   
    def collate_fn(self,batch):
        all_texts = []
        all_labels = []
        for i,element in enumerate(batch):
            all_texts.append(element['texts']) 
            all_labels.append(element['labels'])
        encoding = self.tokenizer(
            all_texts ,
            is_split_into_words=True,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_length,
           
            return_tensors='pt'
        )
        #把标签转成数字:同时bert做ner时标签输入的shape:(batch_size,seq_len)
        #词和token的映射，一个词如果被分成多个，word_ids会映射同一个词的索引,
        # word_ids只能对单条文本，所以得对批标签分别做对齐
        new_labels_id = []
        # new_labels_str = []
        for j,label in enumerate(all_labels):
            word_ids = encoding.word_ids(batch_index=j)
            new_label_id =self.align_label(label, word_ids)
            new_labels_id.append(new_label_id)

        # print('=====================================')
        # print('id:',new_labels_id)
        # print('str',new_labels_str)
        # print('=====================================')

        encoding['labels'] = torch.tensor(new_labels_id)
        return encoding

