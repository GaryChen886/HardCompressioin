import numpy as np

def float_to_fixed_point(x, total_bits=16, decimal_bits=8):
    """將浮點數轉換為定點數"""
    # 計算總位數和小數位數的位元數
    integer_bits = total_bits - decimal_bits
    # 將浮點數乘上2的小數位數次方，變為整數
    x_fixed = int(x * 2**decimal_bits)
    # 將整數轉換為二進位字符串，補齊總位數
    x_fixed_str = np.binary_repr(x_fixed, width=total_bits)
    # 如果是負數，則進行二補數運算
    if x < 0:
        x_fixed_str = x_fixed_str.replace('0', '2').replace('1', '0').replace('2', '1')
        x_fixed = -int(x_fixed_str, 2) - 1
    else:
        x_fixed = int(x_fixed_str, 2)
    return x_fixed

# 將每個權重轉換為定點數
weight_fixed = []
for w in weight:
    w_fixed = [float_to_fixed_point(w_i) for w_i in w]
    weight_fixed.append(w_fixed)
weight_fixed = np.array(weight_fixed)

print(weight_fixed)
