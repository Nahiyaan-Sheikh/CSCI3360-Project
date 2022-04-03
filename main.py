import pandas as pd
import numpy as np
import matplotlib as mp

gpu_df = pd.read_csv('./GPU data/FACT_GPU_PRICE.csv')

# drop unused columns
to_drop = ['RegionId', 'MerchantId']
gpu_df.drop(to_drop, inplace=True, axis=1)


# drop unused rows
rx = range(893, 945)
gtx960 = range(1879, 1898)
gtx970_980 = range(1907, 1948)
gtx10 = range(995, 1108)
titan = range(1960, 1975)

datarx = gpu_df[gpu_df['ProdId'] >= 893]
datarx = datarx[datarx['ProdId'] <= 945]

data960 = gpu_df[gpu_df['ProdId'] >= 1879]
data960 = data960[data960['ProdId'] <= 1898]

data970_80 = gpu_df[gpu_df['ProdId'] >= 1907]
data970_80 = data970_80[data970_80['ProdId'] <= 1948]

data10 = gpu_df[gpu_df['ProdId'] >= 995]
data10 = data10[data10['ProdId'] <= 1108]

datatitan = gpu_df[gpu_df['ProdId'] >= 1960]
datatitan = datatitan[datatitan['ProdId'] <= 1975]

# combines all dataframes, gpu_data is the correct df with gpus we are using
frames = [datarx, data960, data970_80, data10, datatitan]
gpu_final = pd.concat(frames)

print(gpu_final)