"""
Prepare the novel xl dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.


This file is modified to process the seperated val.txt and train.txt.  The two file need to be preared manually. And this file will use the 
combined file (val.txt + train.txt) to generate the character table to make sure the table includes all the charcters in the two files. And 
convert the val.txt and train.txt to val.bin and train.bin based on the table.

"""



import os
import pickle
import requests
import numpy as np
import chardet


def convert_to_utf8(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        encoding = chardet.detect(data)['encoding']
        if encoding != 'utf-8':
            data = data.decode(encoding).encode('utf-8')
            with open(file_path, 'wb') as fw:
                fw.write(data)
            print("文件已成功转换为UTF-8编码")
        else:
            print("文件已经是UTF-8编码")




# download the tiny shakespeare dataset
input_file_path_train_val = os.path.join(os.path.dirname(__file__), 'train_val.txt')
input_file_path_train = os.path.join(os.path.dirname(__file__), 'train_1.txt')
input_file_path_val = os.path.join(os.path.dirname(__file__), 'val_1.txt')

#if not os.path.exists(input_file_path):
#   data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#    with open(input_file_path, 'w') as f:
#        f.write(requests.get(data_url).text)


# 使用示例

#convert_to_utf8(input_file_path_train_val)
#convert_to_utf8(input_file_path_train)
#convert_to_utf8(input_file_path_val)

with open(input_file_path_val, 'r') as f:
    data_val = f.read()
print(f"length of dataset in characters: {len(data_val):,}")

with open(input_file_path_train, 'r') as f:
    data_train = f.read()
print(f"length of dataset in characters: {len(data_train):,}")

with open(input_file_path_train_val, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")




# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
#n = len(data)
train_data = data_train
val_data = data_val


# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
