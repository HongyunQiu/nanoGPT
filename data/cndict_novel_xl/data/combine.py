import os
import chardet
import codecs

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    return chardet.detect(rawdata)['encoding']

def convert_encoding_to_utf8(file_path, original_encoding):
    with codecs.open(file_path, 'r', original_encoding) as source_file:
        contents = source_file.read()
    with codecs.open(file_path, 'w', 'utf-8') as target_file:
        target_file.write(contents)

def concatenate_txt_files(directory, output_file):
    file_counter = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                original_encoding = detect_encoding(file_path)
                if original_encoding is None:
                    print(f"Could not detect the encoding of file {file_path}. Skipping this file.")
                    continue
                elif original_encoding.lower() != 'utf-8':
                    try:
                        convert_encoding_to_utf8(file_path, original_encoding)
                    except Exception as e:
                        print(f"Error converting file {file_path} to UTF-8. Skipping this file. Error: {str(e)}")
                        continue
                
                with open(file_path, 'r', encoding='utf-8') as source_file:
                    contents = source_file.read()
                with open(output_file, 'a', encoding='utf-8') as target_file:
                    target_file.write(contents + '\n')
                
                print(f"Processed file #{file_counter}: {file_path}")
                file_counter += 1

output_file = 'combined.txt'
concatenate_txt_files('.', output_file)