import sys,os

import torch
from modelscope import snapshot_download
# 推理用的指定模型

model_id = 'Omelette/chat-daiyu-sovits'
mode_name_or_path = snapshot_download(model_id, revision='master')
model_file = "daiyu-voice_e15_s195.pth"   
full_path = mode_name_or_path + model_file  
sovits_path = full_path
gpt_path = "SoVits/GPT_weights/daiyu-voice-e12.ckpt"

# 一些默认参数
default_refer_path = "SoVits/daiyu_example.wav"
default_refer_text = "是单给我一个人的，还是别的姑娘都有？"
default_refer_language = "zh"

is_half_str = os.environ.get("is_half", "True")
is_half = True if is_half_str.lower() == 'true' else False
is_share_str = os.environ.get("is_share","False")
is_share= True if is_share_str.lower() == 'true' else False


model_id = 'AI-ModelScope/GPT-SoVITS'
mode_name_or_path = snapshot_download(model_id, revision='master')

model_file = "chinese-hubert-base"   
full_path = mode_name_or_path + model_file   

cnhubert_path = full_path

model_file = "chinese-roberta-wwm-ext-large"   
full_path = mode_name_or_path + model_file  
bert_path = full_path

model_file = "s2G488k.pth"   
full_path = mode_name_or_path + model_file  
pretrained_sovits_path = full_path

model_file = "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"   
full_path = mode_name_or_path + model_file  
pretrained_gpt_path = full_path

exp_root = "logs"
python_exec = sys.executable or "python"
if torch.cuda.is_available():
    infer_device = "cuda"
elif torch.backends.mps.is_available():
    infer_device = "mps"
else:
    infer_device = "cpu"



api_port = 9880

if infer_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half=False

if(infer_device=="cpu"):is_half=False

class Config:
    def __init__(self):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.default_refer_path = default_refer_path
        self.default_refer_text = default_refer_text
        self.default_refer_language = default_refer_language
        self.is_half = is_half

        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.pretrained_sovits_path = pretrained_sovits_path
        self.pretrained_gpt_path = pretrained_gpt_path

        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device


        self.api_port = api_port
