# This programe is used to analyze the structure of nanoGPT's model . And convert the weight into the 16bit 2-D image 

import os
import torch
import pickle
import numpy as np


from model import GPTConfig, GPT

from PIL import Image


def Tensor2Img(tensor,filname):
    #generate a 16bit image
    # uniform to 0-1
    tensor_normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # map to 0-65535
    tensor_mapped = (tensor_normalized.detach() * 65535).numpy().astype(np.uint16)
    tensor_transposed = np.transpose(tensor_mapped) 
    print(tensor_transposed.shape)

    image = Image.fromarray(tensor_transposed, mode='I;16')
    image.save(filname)


device="cuda:5"
model_file="/out-cndict_novel_xl/hihglayer/ckpt.pt"
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
Tensor2Img(old_embeddings,"wte.png")


print (model.transformer.wte.weight.data)
print (model.lm_head.weight)

# print all param 
for pn, p in model.named_parameters():
    print (pn)

# print a Layer
n=255

print ("ln_1")
print (model.transformer.h[n].ln_1.weight)
print (model.transformer.h[n].ln_1.weight.size())
print ("attn.c_attn WQ WK WV")
print (model.transformer.h[n].attn.c_attn.weight)
print (model.transformer.h[n].attn.c_attn.weight.size())
print ("attn.c_proj")
print (model.transformer.h[n].attn.c_proj.weight)
print (model.transformer.h[n].attn.c_proj.weight.size())
print ("ln_2")
print (model.transformer.h[n].ln_2.weight)
print (model.transformer.h[n].ln_2.weight.size())
print ("mlp_c_fc")
print (model.transformer.h[n].mlp.c_fc.weight)
print (model.transformer.h[n].mlp.c_fc.weight.size())
print ("mlp_c_proj")
print (model.transformer.h[n].mlp.c_proj.weight)
print (model.transformer.h[n].mlp.c_proj.weight.size())
print ("ln_f")
print (model.transformer.ln_f.weight.size())


Tensor2Img(model.transformer.h[n].attn.c_attn.weight,"attn_attn.png")
Tensor2Img(model.transformer.h[n].attn.c_proj.weight,"attn_proj.png")
Tensor2Img(model.transformer.h[n].mlp.c_fc.weight,"c_fc.png")
Tensor2Img(model.transformer.h[n].mlp.c_proj.weight,"c_proj.png")


