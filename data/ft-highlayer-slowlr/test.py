import torch
import os
import pickle





## print the token dict with list  
def print_dict(itos_dict):
    for index, token in itos_dict.items():
        print(f"{index}:{repr(token)}")
## print the token dict without index        
def print_itos_tokens(itos_dict):
    tokens = [repr(token)[1:-1] for token in itos_dict.values()]
    print(' '.join(tokens))


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
        meta = {'itos': itos, 'stoi': stoi}
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        print(f"Updated itos and stoi with new characters and saved to {meta_path}")
    else:
        print("No new characters found in the text.")    

dir='out-math'
modelfile=os.getcwd() +"/" + dir + "/12x12x768/ckpt.pt"
print ("model file:"+modelfile) 

checkpoint = torch.load(modelfile)
model_args= checkpoint['model_args']

config = checkpoint['config']
#print (model_args)
#print (config)


meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

print ("-------------Before------------------")
print_itos_tokens(itos)

text="new"
update_meta_with_new_chars(text, itos, stoi, meta_path)

print ("-------------after------------------")
print_itos_tokens(itos)


