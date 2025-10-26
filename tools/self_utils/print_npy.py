import numpy as np

# 載入 .npy 檔案
data = np.load('/home/ys/MS3D/data/custom/citystreet_sunny_night_2024-12-11-19-22-40/VLS128_pcdnpy/000000.npy')

# 印出資料型態、形狀與內容
print("資料型態:", type(data))
print("資料形狀:", data.shape)
print("資料內容:\n", data)