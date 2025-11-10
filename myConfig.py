import torch
def tag2id(path):
        tag_to_id = {}
        tag_count = 0
        id_to_tag = {}  #预训练模型时需要传的都是字典
        with open(path,'r',encoding='UTF-8') as file:
            for line in file:
                data = line.strip()
                # print(data)
                tag_to_id[data] = tag_count
                id_to_tag[tag_count] = data
                tag_count += 1
        assert len(tag_to_id) == len(id_to_tag)
        return tag_to_id,tag_count,id_to_tag
tag_to_id,tag_count,id_to_tag = tag2id(r'/home/sulin/NER/data/class.txt')

class myConfig:
    """配置参数"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
myconfig = myConfig(
    # model_name = 'bert-base-chinese',
    dataset = r'/home/sulin/NER/data/',
    data_name = '.txt',
    bert_path = 'bert-base-chinese',
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  , # 设备
    class_num = 17 ,                                           # 类别数
    epochs = 3 ,                                            # epoch数
    batch_size = 16   ,                                        # mini-batch大小
    learning_rate = 2e-5  ,                                     # 学习率
    max_length = 128,
    tag_2_id = tag_to_id,
    id_2_tag = id_to_tag
    )

