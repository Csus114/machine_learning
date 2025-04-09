import scipy.io as sio
import numpy as np

"""标准化方法"""
# 指定范围标准化
def Spec_scope_standar(data, max, min):
    data_max = np.max(data)
    data_min = np.min(data)
    print(data_max, ' ', data_min)
    for i in range(len(data)):
        fai = (data[i] - data_min) / (data_max - data_min)
        data[i] = min + (fai / (max - min))
    return data

# 标准化还原
def Restore(data, max, min):
    data_max = 19.805500483174406
    data_min = 7.5892831232173155
    for i in range(len(data)):
        fai = (data[i] - min) * (max - min)
        data[i] = fai * (data_max - data_min) + data_min
    return data
