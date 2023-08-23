import os
import torch
import pickle
import numpy as np


from model import GPTConfig, GPT

from PIL import Image

device="cuda:6"
model_file="/out-cndict_novel_xl/hihglayer_slowlr/backup/ckpt.pt"
model_file_path=os.getcwd() + model_file
print ("model file:"+model_file_path)

checkpoint = torch.load(model_file_path,device)
model_args= checkpoint['model_args']
config = checkpoint['config']
print (model_args)
print (config)


gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']

# 提取现有的词嵌入权重
old_embeddings=model.transformer.wte.weight.data

# uniform to 0-1
tensor_normalized = (old_embeddings - old_embeddings.min()) / (old_embeddings.max() - old_embeddings.min())
# map to 0-65535
tensor_mapped = (tensor_normalized * 65535).numpy().astype(np.uint16)

print(old_embeddings.size())

image = Image.fromarray(tensor_mapped, mode='I;16')
image.save('tensor_image_slowlr.png')



