import numpy as np

# 讀取weight.txt檔案
with open('weight.txt', 'r') as f:
    lines = f.readlines()

# 將每個數值轉換成float並存儲到numpy array中
weights = []
for line in lines:
    weights.append([float(val) for val in line.strip().split()])
weights = np.array(weights)

# 輸出讀取到的weight數值
print(weights)
