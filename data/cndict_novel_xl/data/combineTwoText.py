# 合并两个文本文件
def merge_files(file1, file2, output_file):
    with open(file1, 'r',encoding='utf-8') as f1, open(file2, 'r',encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    with open(output_file, 'w',encoding='utf-8') as f3:
        for line in lines1:
            f3.write(line)
        for line in lines2:
            f3.write(line)

if __name__ == "__main__":
    file1 = 'train_1.txt' # 请替换为你的第一个文件名
    file2 = 'val_1.txt' # 请替换为你的第二个文件名
    output_file = 'train_val.txt' # 输出文件名

    merge_files(file1, file2, output_file)