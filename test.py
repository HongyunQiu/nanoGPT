# AutoMatic Train with any text
# This code will read the text file and automaticly generate the train and val binary file and then start trainning.
# It will handle the DICT automaticly. If there is new token in the input text , it will append to the metal.pkl



import os
import torch
import pickle
import numpy as np

from model import GPTConfig, GPT






## print the token dict with list  
def print_dict(itos_dict):
    for index, token in itos_dict.items():
        print(f"{index}:{repr(token)}")
## print the token dict without index        
def print_itos_tokens(itos_dict):
    tokens = [repr(token)[1:-1] for token in itos_dict.values()]
    print(''.join(tokens))


# check if the text has the token that not exit in the token dict, if no , update the token dict and then save to meta.pkl
def update_meta_with_new_chars(text, itos, stoi, meta_path):
    # 将文本转换为一组唯一字符
    unique_chars = set(text)
    
    # 找到不在itos中的字符
    new_chars = unique_chars - set(itos.values())
    
    if new_chars:
        # 打印提示
        print(f"Found new characters: {', '.join(new_chars)}")
        
        # 更新itos和stoi
        start_index = max(itos.keys()) + 1
        for i, char in enumerate(new_chars, start=start_index):
            itos[i] = char
            stoi[char] = i
        
        # 保存更新后的meta到meta.pkl
        vocab_size = len(itos)

        meta = {'vocab_size': vocab_size,'itos': itos, 'stoi': stoi}
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"Updated itos and stoi with new characters and saved to {meta_path},vocab size = {vocab_size}")
    else:
        print("No new characters found in the text.")    



def updateModel(vocab_size):
    model_args['vocab_size'] = new_vocab_size
    # Step 2: 创建新的模型实例
    gptconf = GPTConfig(**model_args)
    new_model = GPT(gptconf)

    # Step 3: 使用原始checkpoint的state_dict加载模型的权重
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary as before
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    new_model.load_state_dict(state_dict)
    checkpoint['model_args'] = model_args  # 更新模型参数
    checkpoint['model'] = new_model.state_dict()  # 更新模型权重
    torch.save(checkpoint, model_file)





model_file="/out-cndict_novel_xl/hihglayer_slowlr/ckpt.pt"
model_file_path=os.getcwd() + model_file
print ("model file:"+model_file_path)


# need to put the meta.pkl and the input txt file to this folder for first
input_file_folder="data/cndict_novel_instruct"
input_file_path=os.getcwd() +"/" + input_file_folder +"/"+"novel_cn_token512_50k.json"
print ("input txt file:"+input_file_path)

# load the model and the token dict file

checkpoint = torch.load(model_file_path)
model_args= checkpoint['model_args']
config = checkpoint['config']
print (model_args)
print (config)


#meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
meta_path = os.getcwd() + "/" + input_file_folder + "/" + "meta.pkl"
load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

## update the token dict with the new token in the input txt file

print ("-------------Before------------------")
#print_itos_tokens(itos)


#data="new"
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")
update_meta_with_new_chars(data, itos, stoi, meta_path)

print ("-------------after------------------")
#print_itos_tokens(itos)


# generate train and val data set with the input txt file
# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

print (os.path.dirname(__file__))

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


new_vocab_size = meta['vocab_size']

print (f"vocab_size in final {new_vocab_size}")

