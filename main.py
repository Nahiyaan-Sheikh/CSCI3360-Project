import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load grpu data
gpu_df = pd.read_csv('./gpu/FACT_GPU_PRICE.csv')

# drop unused columns
to_drop = ['RegionId', 'MerchantId']
gpu_df.drop(to_drop, inplace=True, axis=1)


# drop unused rows and round values
datarx = gpu_df[gpu_df['ProdId'] >= 893]
datarx = datarx[datarx['ProdId'] <= 945]
datarx = datarx.round(1)

data960 = gpu_df[gpu_df['ProdId'] >= 1879]
data960 = data960[data960['ProdId'] <= 1898]
data960 = data960.round(1)

data970_80 = gpu_df[gpu_df['ProdId'] >= 1907]
data970_80 = data970_80[data970_80['ProdId'] <= 1948]
data970_80 = data970_80.round(1)

data10 = gpu_df[gpu_df['ProdId'] >= 995]
data10 = data10[data10['ProdId'] <= 1108]
data10 = data10.round(1)

datatitan = gpu_df[gpu_df['ProdId'] >= 1960]
datatitan = datatitan[datatitan['ProdId'] <= 1975]
datatitan = datatitan.round(1)

# combines all dataframes, gpu_final is the correct df with gpus we are using
frames = [datarx, data960, data970_80, data10, datatitan]
gpu_final = pd.concat(frames)

# print(gpu_final)

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

# drop crypto rows that are outside gpu dates
dataBTC = btc_df[btc_df['Date'] >= 'Sep 19, 2014']
dataBTC = dataBTC[dataBTC['Date'] <= 'Mar 16, 2018']

dataETH = eth_df[eth_df['Date'] >= 'Sep 19, 2014']
dataETH = dataETH[dataETH['Date'] <= 'Mar 16, 2018']

dataLTC = ltc_df[ltc_df['Date'] >= 'Sep 19, 2014']
dataLTC = dataLTC[dataLTC['Date'] <= 'Mar 16, 2018']


# create new dataframes for linear regression
reg_btc_df = pd.DataFrame(columns=['date', 'rx_price', '960_price',
                                   '970_980_price', '10_price', 'titan_price', 'btc_price'])
reg_eth_df = pd.DataFrame(columns=['date', 'rx_price', '960_price',
                                   '970_980_price', '10_price', 'titan_price', 'eth_price'])
reg_ltc_df = pd.DataFrame(columns=['date', 'rx_price', '960_price',
                                   '970_980_price', '10_price', 'titan_price', 'ltc_price'])

reg_btc_df['date'] = dataBTC['Date']
reg_btc_df['btc_price'] = dataBTC['Price']
print(datarx)
#meanrx = datarx.sort_values('TimeId', ascending=False)
#reg_btc_df['rx_price'] = meanrx.groupby('TimeId').mean()

print(reg_btc_df)

reg_eth_df['date'] = dataETH['Date']
reg_eth_df['eth_price'] = dataETH['Price']

reg_ltc_df['date'] = dataLTC['Date']
reg_ltc_df['ltc_price'] = dataLTC['Price']


