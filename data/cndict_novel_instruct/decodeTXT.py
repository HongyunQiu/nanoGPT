filename = "/home/ad/workspace/nanoGPT/data/cndict_novel_instruct/novel_cn_token512_50k.json"

# 打开文件
with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()

# 使用 decode 方法解码转义序列
decoded_text = bytes(text, "utf-8").decode("unicode_escape")

# 打印解码后的文本
# print(decoded_text)

# 如果你想保存解码后的文本，可以这样操作
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write(decoded_text)
