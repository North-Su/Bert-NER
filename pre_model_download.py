#下载模型权重
from huggingface_hub import snapshot_download

# 下载模型到指定文件夹
model_path = snapshot_download(
    repo_id="hfl/chinese-bert-wwm",  # 模型名称
    local_dir="/home/sulin/NER/pre_model/chinese-bert-wwm",  
    local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
    revision="main" , # 指定分支或版本
    ignore_patterns=["*.h5", "*.ot", "tf_model*"] #只保留pytorch权重
)

print(f"模型已下载到: {model_path}")