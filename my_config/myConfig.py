import torch
import yaml
def tag2id(path):
        tag_to_id = {}
        tag_count = 0
        id_to_tag = {}  #预训练模型时需要传的都是字典
        with open(path,'r',encoding='UTF-8') as file:
            for line in file:
                data = line.strip()
                tag_to_id[data] = tag_count
                id_to_tag[tag_count] = data
                tag_count += 1
        assert len(tag_to_id) == len(id_to_tag)
        return tag_to_id,id_to_tag

class my_Config:
    def __init__(self, yaml_path="/home/sulin/code/my_config/config.yaml"):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 获取运行选项
        dataset_key = cfg["run"]["dataset"]
        model_key = cfg["run"]["model"]
        self.dataset_type = dataset_key
        self.model_type = model_key

        # === 加载数据集配置 ===
        dataset_cfg = cfg["datasets"][dataset_key]
        for k, v in dataset_cfg.items():
            setattr(self, k, v)

        # === 加载标签映射 ===
        self.tag_2_id, self.id_2_tag = tag2id(dataset_cfg["tag_file"])

        # === 加载模型配置 ===
        model_cfg = cfg["models"][model_key]
        for k, v in model_cfg.items():
            setattr(self, k, v)

        # === 加载训练超参 ===
        train_cfg = cfg["train"]
        for k, v in train_cfg.items():
            setattr(self, k, v)

        # === 自动设置设备 ===
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        return str(self.__dict__)

cfg = my_Config()
print(cfg)