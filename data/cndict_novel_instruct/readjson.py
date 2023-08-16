import ijson

filename = "/home/ad/workspace/nanoGPT/data/cndict_novel_instruct/novel_cn_token512_50k.json"
n=100

# 使用文件对象进行迭代解析
with open(filename, 'r', encoding='utf-8') as file:
    objects = ijson.items(file, 'item')
    for idx,obj in enumerate(objects):
        # 这里处理每一个JSON对象
        if idx==n - 1:
          nth_object =obj
          break
        
#print(nth_object)

text_field0  = nth_object['instruction']     
text_field1  = nth_object['input']
text_field2  = nth_object['output'] 

print ("\ninstruction\n")
print (text_field0)
print ("\n")
# 使用换行符拆分文本

print ("\ninput\n")

lines = text_field1.split('\n')

# 逐行打印
for line in lines:
    print(line)        
        
        
print ("\noutput\n")        
        
# 使用换行符拆分文本
lines = text_field2.split('\n')

# 逐行打印
for line in lines:
    print(line)   
