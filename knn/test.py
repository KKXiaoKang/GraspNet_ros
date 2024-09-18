#!/usr/bin/env python

import scipy.io as scio

# 读取 .mat 文件
file_path = 'path/to/your/file.mat'
data = scio.loadmat(file_path)

# 查看文件中的所有变量名
print("Variables in the .mat file:")
for key in data:
    if not key.startswith('__'):  # 排除 MATLAB 内部变量
        print(f"{key}: {type(data[key])}")

# 获取并打印特定变量的数据（示例）
variable_name = 'variable_name'  # 替换为你感兴趣的变量名
if variable_name in data:
    variable_data = data[variable_name]
    print(f"\nData for {variable_name}:")
    print(variable_data)
else:
    print(f"\n{variable_name} not found in the .mat file.")

