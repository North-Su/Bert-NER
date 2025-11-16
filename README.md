# Named Entity Recognition (NER) Project

---

## 🚀 项目简介
本项目实现了基于 **BERT** 的命名实体识别（NER）模型，用于识别文本中的实体类别（如人名、地名、组织机构等）。  
主要特点：

- 支持BIO标记方法下的中文实体识别
- 提供训练、验证、测试与推理功能
- 提供实体级 F1、Precision、Recall 评估
- 支持模型保存与加载

---

## 🗂 项目结构

```
NER/
├── data/                  # 数据
├── checkpoints/           # 模型保存
├── pre_model/             # 预训练模型文件夹
├── my_config/             # 配置文件
├── data_pre_.py           # 数据预处理
├── metric.py.py           # P、R、F1指标评估函数
├── model.py               # 模型定义
├── pre_model_download.py  # 预训练模型权重下载
├── train_spilt.py         # 切分训练集、验证集文件
├── train_evaluate.py      # 训练、推理函数
├── README.md              # 说明文档
└── .gitignore             # 忽略文件

```

## 🏗 模型说明

本项目使用的Bert+MLP，模型可通过pre_model_download.py文件进行下载，只需修改本地存放路径即可。

Bert预训练模型及其来源如下：

```
1. bert-base-chinese ：https://huggingface.co/google-bert/bert-base-chinese
2. chinese-bert-wwm ：https://huggingface.co/hfl/chinese-bert-wwm  
```

模型目录结构：

```
pre_model/
├── bert-base-chinese
├── chinese-bert-wwm 
```

## 🗂 数据准备

数据集及其来源：

```
1.weibo：https://tianchi.aliyun.com/dataset/144312
	weibo数据集包括训练集（1350）、验证集（269）、测试集（270），实体类型包括地缘政治实体(GPE.NAM)、地名(LOC.NAM)、机构名(ORG.NAM)、人名(PER.NAM)及其对应的代指(以NOM为结尾）。
2.msra：https://tianchi.aliyun.com/dataset/144307?spm=a2c22.12282016.0.0.432a4f03K11Mhq
	MSRA数据集是面向新闻领域的中文命名实体识别数据集。本数据集包括训练集（46364）、测试集（4365），实体类型包括地名(LOC)、人名(PER)、组织名(ORG)。
将数据集下载到本地后，可用train_spilt.py函数对训练集进行划分，可将划分后的训练集、验证集重新写成文件。
标签文件可以自己按照简介中的标签种类生成class.txt文件，文件结构如下：
B-LOC
I-LOC
B-PER
I-PER
....
B-ORG
I-ORG
O
```

数据集目录结构：

```
data/
├── class.txt #标签信息
├── train.txt 
├── dev.txt
└── test.txt
```

## ⚙ 配置文件

本项目所有的配置文件均写于my_config文件夹中，配置文件目录结构：

```
my_config/
├── config.yaml  #所有参数写成.yaml文件
├── myConfig.py  #读取.yaml文件生成cofig参数类
```

特殊说明：

```
因为项目包含两组预训练权重和两个数据集，为了更改模型和数据集方便，设置了run参数整体调控，后面的datasets和models中会具体设置当前使用的模型、数据集的具体信息，train中设置了模型的训练参数，具体详见config.yaml文件。
run:
  dataset: weibo/msra
  model: bert-base/bert_wwm
datasets:
  ...
models:
  ...
train:
  ...
```

## ⚡ 快速开始

### 训练/评估

```
直接运行train_evaluate.py文件即可。
```

常用参数：

`--epochs`：训练轮数

`--batch_size`：批次大小

`--lr`：学习率

`--device`：设备，如 `cuda:0` 或 `cpu`

`...`

输出指标：

- `Precision（精确率）`
- `Recall（召回率）`
- `F1-score（F1 分数）`

训练结束后，最佳模型（取F1值最好的模型权重）会保存至 `checkpoints/`。

### 测试

加载已保存的权重，进行测试，返回`Precision（精确率）、Recall（召回率）、F1-score（F1 分数）。`
