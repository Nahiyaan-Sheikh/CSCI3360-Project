import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load grpu data
gpu_df = pd.read_csv('./gpu/FACT_GPU_PRICE.csv')

# drop unused columns
to_drop = ['RegionId', 'MerchantId']
gpu_df.drop(to_drop, inplace=True, axis=1)


# drop unused rows
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

# combines all dataframes, gpu_final is the correct df with gpus we are using
frames = [datarx, data960, data970_80, data10, datatitan]
gpu_final = pd.concat(frames)

#print(gpu_final)

# load crypto data
btc_df = pd.read_csv('./crypto/Bitcoin_Historical_Data.csv')
eth_df = pd.read_csv('./crypto/Ethereum_Historical_Data.csv')
ltc_df = pd.read_csv('./crypto/Litecoin_Historical_Data.csv')

# drop unused columns, convert datetimes and strings
drop = ['Open', 'High', 'Low', 'Vol.']
btc_df.drop(drop, inplace=True, axis=1)
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df['Price'] = btc_df['Price'].str.replace(',', '')
btc_df['Change %'] = btc_df['Change %'].str.replace('%', '')
eth_df.drop(drop, inplace=True, axis=1)
eth_df['Date'] = pd.to_datetime(eth_df['Date'])
eth_df['Price'] = eth_df['Price'].str.replace(',', '')
eth_df['Change %'] = eth_df['Change %'].str.replace('%', '')
ltc_df.drop(drop, inplace=True, axis=1)
ltc_df['Date'] = pd.to_datetime(ltc_df['Date'])
ltc_df['Change %'] = ltc_df['Change %'].str.replace('%', '')

print('')
print('GPU min date: ', gpu_final['TimeId'].min())
print('GPU max date: ', gpu_final['TimeId'].max())
print('Btc min date: ', btc_df['Date'].min())
print('Btc max date: ', btc_df['Date'].max())
print('Eth min date: ', eth_df['Date'].min())
print('Eth max date: ', eth_df['Date'].max())
print('Ltc min date: ', ltc_df['Date'].min())
print('Ltc max date: ', ltc_df['Date'].max())

# drop cpryto rows that are outside of gpu dates
dataBTC = btc_df[btc_df['Date'] >= 'Sep 19, 2014']
dataBTC = dataBTC[dataBTC['Date'] <= 'Mar 16, 2018']

dataETH = eth_df[eth_df['Date'] >= 'Sep 19, 2014']
dataETH = dataETH[dataETH['Date'] <= 'Mar 16, 2018']

dataLTC = ltc_df[ltc_df['Date'] >= 'Sep 19, 2014']
dataLTC = dataLTC[dataLTC['Date'] <= 'Mar 16, 2018']


# calculate means and correlations
arr_gpu = np.array(gpu_final)
arr_btc = np.array(dataBTC)
arr_eth = np.array(dataETH)
arr_ltc = np.array(dataLTC)

print('')
print(arr_gpu)
print('')
print(arr_btc)

#corr_gpu_btc = np.corrcoef(arr_gpu[:,2].astype('float'), arr_btc[:,1])
#print(corr_gpu_btc)




