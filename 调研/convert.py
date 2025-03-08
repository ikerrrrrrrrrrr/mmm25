import json
import os

def convert_ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r',encoding='utf-8') as f:
        notebook = json.load(f)

    with open(py_file, 'w',encoding='utf-8') as f:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                f.write(''.join(cell['source']) + '\n\n')

def main():
    # 获取当前目录下的所有文件
    files = os.listdir('.')
    
    # 遍历文件，找到所有.ipynb文件
    for file in files:
        if file.endswith('.ipynb'):
            # 构造对应的.py文件名
            py_file = file[:-6] + '.py'  # 去掉.ipynb后缀，加上.py后缀
            # 调用convert_ipynb_to_py函数进行转换
            convert_ipynb_to_py(file, py_file)
            print(f'Converted {file} to {py_file}')

if __name__ == '__main__':
    main()