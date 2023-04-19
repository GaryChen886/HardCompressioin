import numpy as np
import os

# 讀取weights_hex.txt文件
filepath = "weights_hex.txt"
if not os.path.exists(filepath):
    raise FileNotFoundError("File not found: " + filepath)

with open(filepath, "r") as f:
    hex_str = f.read()

# hex字符串轉為Binary字符串
bin_str = bin(int(hex_str, 16))[2:]

# Binary字符串按照32位分割
bin_str_list = [bin_str[i:i+32] for i in range(0, len(bin_str), 32)]

# Binary轉為int
int_arr = np.array([int(x, 2) for x in bin_str_list], dtype=np.uint32)

# Euclidean Distance做dictionary based compression
unique_int_arr = np.unique(int_arr)
compressed_int_arr = np.zeros_like(int_arr, dtype=np.uint32)
lookup_table = dict(zip(unique_int_arr, range(len(unique_int_arr))))
for i in range(len(int_arr)):
    compressed_int_arr[i] = lookup_table[int_arr[i]]

# 結果寫入compressed_weight.txt
compressed_file_path = "compressed_weight.txt"
with open(compressed_file_path, "w") as f:
    for i in compressed_int_arr:
        f.write(str(i) + "\n")

# entry寫入entry.txt
entry_file_path = "entry.txt"
with open(entry_file_path, "w") as f:
    for i in lookup_table:
        f.write(str(i) + " " + str(lookup_table[i]) + "\n")
