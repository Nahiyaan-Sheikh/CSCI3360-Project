import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

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

print("rx min " + str(datarx['TimeId'].min()))
print("rx max " + str(datarx['TimeId'].max()))
print("960 min " + str(data960['TimeId'].min()))
print("960 max " + str(data960['TimeId'].max()))
print("970-980 min " + str(data970_80['TimeId'].min()))
print("970-980 max " + str(data970_80['TimeId'].max()))
print("10 min " + str(data10['TimeId'].min()))
print("10 max " + str(data10['TimeId'].max()))
print("titan min " + str(datatitan['TimeId'].min()))
print("titan max " + str(datatitan['TimeId'].max()))
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

meanrx = datarx.sort_values('TimeId', ascending=False)
meanrx = meanrx[meanrx['TimeId'] >= 20160824]
meanrx = meanrx[meanrx['TimeId'] <= 20180315]
rx560 = meanrx[meanrx['ProdId'] >= 893]
rx560 = meanrx[meanrx['ProdId'] <= 904]
rx560 = rx560.groupby('TimeId').mean()
print('rx560: ')
print(rx560)

rx570 = meanrx[meanrx['ProdId'] >= 905]
rx570 = meanrx[meanrx['ProdId'] <= 916]
rx570 = rx570.groupby('TimeId').mean()
print('rx570: ')
print(rx570)

rx580 = meanrx[meanrx['ProdId'] >= 917]
rx580 = meanrx[meanrx['ProdId'] <= 930]
rx580 = rx580.groupby('TimeId').mean()
print('rx580: ')
print(rx580)

vega56 = meanrx[meanrx['ProdId'] >= 931]
vega56 = meanrx[meanrx['ProdId'] <= 936]
vega56 = vega56.groupby('TimeId').mean()
print('vega56: ')
print(vega56)

vega64 = meanrx[meanrx['ProdId'] >= 937]
vega64 = meanrx[meanrx['ProdId'] <= 944]
vega64 = vega64.groupby('TimeId').mean()
print('vega64: ')
print(vega64)

mean960 = data960.sort_values('TimeId', ascending=False)
mean960 = mean960[mean960['TimeId'] >= 20160824]
mean960 = mean960[mean960['TimeId'] <= 20180315]
mean960 = mean960[mean960['ProdId'] >= 1879]
mean960 = mean960[mean960['ProdId'] <= 1897]
print('mean960: ')
print(mean960)

#date_fmt = '%Y%m%d'
#mean960['TimeId'] = dt.datetime.strptime(mean960['TimeId'], '%Y%m%d')
#dt_x = [dt.datetime.strptime(str(i), date_fmt) for i in mean960['TimeId']]
#mean960['TimeId'] = dt.datetime.strptime(mean960['TimeId'], '%Y%m%d')
mean960['TimeId'] = list([pd.to_datetime(x, format='%Y%m%d') for x in mean960['TimeId'].to_list()])
#mean960['TimeId'] = pd.to_datetime(mean960['TimeId'], format = '%Y%m%d')

fig, ax = plt.subplots(figsize = (15,7))
sns.lineplot(ax = ax, x=mean960['TimeId'], y = mean960['Price_USD'], data=mean960).set_title("GTX 960")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday = 1))

x_ticks = mean960['TimeId']
plt.tick_params(axis='x', which='major')

#plt.plot(mean960['TimeId'], mean960['Price_USD'])
#monthyearFmt = mdates.DateFormatter('%y%m%d')
#ax1.xaxis.set_major_formatter(monthyearFmt)
_ = plt.xticks(rotation=90)
plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)
ax.xaxis.set_minor_locator(mdates.DayLocator(interval = 1))

#meantitan.plot(kind='line', x = 'TimeId', y = 'Price_USD')
plt.show()

mean970_980 = data970_80.sort_values('TimeId', ascending=False)
mean970_980 = mean970_980[mean970_980['TimeId'] >= 20160824]
mean970_980 = mean970_980[mean970_980['TimeId'] <= 20180315]
mean970 = mean970_980[mean970_980['ProdId'] >= 1907]
mean970 = mean970_980[mean970_980['ProdId'] <= 1919]
mean970 = mean970.groupby('TimeId').mean()
print('mean970: ')
print(mean970)

mean980 = mean970_980[mean970_980['ProdId'] >= 1920]
mean980 = mean970_980[mean970_980['ProdId'] <= 1934]
mean980 = mean980.groupby('TimeId').mean()
print('mean980: ')
print(mean980)

mean980ti = mean970_980[mean970_980['ProdId'] >= 1935]
mean980ti = mean970_980[mean970_980['ProdId'] <= 1947]
mean980ti = mean980.groupby('TimeId').mean()
print('mean980ti: ')
print(mean980ti)

mean10 = data10.sort_values('TimeId', ascending=False)
mean10 = mean10[mean10['TimeId'] >= 20160824]
mean10 = mean10[mean10['TimeId'] <= 20180315]
mean1050 = mean10[mean10['ProdId'] >= 995]
mean1050 = mean10[mean10['ProdId'] <= 1006]
mean1050 = mean1050.groupby('TimeId').mean()
print('mean1050: ')
print(mean1050)

mean1050ti = mean10[mean10['ProdId'] >= 1007]
mean1050ti = mean10[mean10['ProdId'] <= 1018]
mean1050ti = mean1050ti.groupby('TimeId').mean()
print('mean1050ti: ')
print(mean1050ti)

mean1060 = mean10[mean10['ProdId'] >= 1019]
mean1060 = mean10[mean10['ProdId'] <= 1045]
mean1060 = mean1060.groupby('TimeId').mean()
print('mean1060: ')
print(mean1060)

mean1070 = mean10[mean10['ProdId'] >= 1046]
mean1070 = mean10[mean10['ProdId'] <= 1060]
mean1070 = mean1070.groupby('TimeId').mean()
print('mean1070: ')
print(mean1070)

mean1070ti = mean10[mean10['ProdId'] >= 1061]
mean1070ti = mean10[mean10['ProdId'] <= 1072]
mean1070ti = mean1070ti.groupby('TimeId').mean()
print('mean1070ti: ')
print(mean1070ti)

mean1080 = mean10[mean10['ProdId'] >= 1073]
mean1080 = mean10[mean10['ProdId'] <= 1087]
mean1080 = mean1080.groupby('TimeId').mean()
print('mean1080: ')
print(mean1080)

mean1080ti = mean10[mean10['ProdId'] >= 1088]
mean1080ti = mean10[mean10['ProdId'] <= 1107]
mean1080ti = mean1080ti.groupby('TimeId').mean()
print('mean1080ti: ')
print(mean1080ti)

meantitan = datatitan.sort_values('TimeId', ascending=False)
meantitan = meantitan[meantitan['TimeId'] >= 20160824]
meantitan = meantitan[meantitan['TimeId'] <= 20180315]
meantitan = meantitan.groupby('TimeId').mean()
print('meantitan: ')
print(meantitan)



#meantitan.columns = meantitan.columns.str.strip()
meantitan.plot(y = 'Price_USD', use_index=True)
#meantitan.plot(kind='line', x = 'TimeId', y = 'Price_USD')
plt.show()
#print(reg_btc_df)

reg_eth_df['date'] = dataETH['Date']
reg_eth_df['eth_price'] = dataETH['Price']

reg_ltc_df['date'] = dataLTC['Date']
reg_ltc_df['ltc_price'] = dataLTC['Price']


