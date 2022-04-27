import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

pd.options.mode.chained_assignment = None  # default='warn'

# load gpu data
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

#print("rx min " + str(datarx['TimeId'].min()))
#print("rx max " + str(datarx['TimeId'].max()))
#print("960 min " + str(data960['TimeId'].min()))
#print("960 max " + str(data960['TimeId'].max()))
#print("970-980 min " + str(data970_80['TimeId'].min()))
#print("970-980 max " + str(data970_80['TimeId'].max()))
#print("10 min " + str(data10['TimeId'].min()))
#print("10 max " + str(data10['TimeId'].max()))
#print("titan min " + str(datatitan['TimeId'].min()))
#print("titan max " + str(datatitan['TimeId'].max()))
#print('GPU min date: ', gpu_final['TimeId'].min())
#print('GPU max date: ', gpu_final['TimeId'].max())
#print('Btc min date: ', btc_df['Date'].min())
#print('Btc max date: ', btc_df['Date'].max())
#print('Eth min date: ', eth_df['Date'].min())
#print('Eth max date: ', eth_df['Date'].max())
#print('Ltc min date: ', ltc_df['Date'].min())
#print('Ltc max date: ', ltc_df['Date'].max())

# drop crypto rows that are outside gpu dates
dataBTC = btc_df[btc_df['Date'] >= 'Mar 31, 2017']
dataBTC = dataBTC[dataBTC['Date'] <= 'Mar 16, 2018']
dataBTC = dataBTC.sort_values(by=['Date'])
dataBTC = dataBTC.reset_index()

dataETH = eth_df[eth_df['Date'] >= 'Mar 31, 2017']
dataETH = dataETH[dataETH['Date'] <= 'Mar 16, 2018']
dataETH = dataETH.sort_values(by=['Date'])
dataETH = dataETH.reset_index()

dataLTC = ltc_df[ltc_df['Date'] >= 'Mar 31, 2017']
dataLTC = dataLTC[dataLTC['Date'] <= 'Mar 16, 2018']
dataLTC = dataLTC.sort_values(by=['Date'])
dataLTC = dataLTC.reset_index()

# method for making graphs of dataframes
def makegraph(dataframe, cardname):
    dataframe['TimeId'] = list([pd.to_datetime(x, format='%Y%m%d') for x in dataframe['TimeId'].to_list()])
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.lineplot(ax=ax, x=dataframe['TimeId'], y=dataframe['Price_USD'], data=dataframe).set_title(cardname)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1))
    x_ticks = dataframe['TimeId']
    plt.tick_params(axis='x', which='major')
    _ = plt.xticks(rotation=90)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.show()

# create dataframes of individual gpus and plot price over time

meanrx = datarx.sort_values('TimeId', ascending=False)
meanrx = meanrx[meanrx['TimeId'] >= 20160824]
meanrx = meanrx[meanrx['TimeId'] <= 20180315]
rx560 = meanrx[meanrx['ProdId'] >= 893]
rx560 = meanrx[meanrx['ProdId'] <= 904]
#makegraph(rx560, 'RX 560')
rx560 = rx560.groupby('TimeId').mean()
rx560 = rx560.reset_index()
rx560['TimeId'] = list([pd.to_datetime(x, format='%Y%m%d') for x in rx560['TimeId'].to_list()])
rx560 = rx560.set_index('TimeId').asfreq('d').reset_index()
#print('rx560: ')
#print(rx560)

rx570 = meanrx[meanrx['ProdId'] >= 905]
rx570 = meanrx[meanrx['ProdId'] <= 916]
#makegraph(rx570, 'RX 570')
rx570 = rx570.groupby('TimeId').mean()
rx570 = rx570.reset_index()
rx570['TimeId'] = list([pd.to_datetime(x, format='%Y%m%d') for x in rx570['TimeId'].to_list()])
rx570 = rx570.set_index('TimeId').asfreq('d').reset_index()
#print('rx570: ')
#print(rx570)

rx580 = meanrx[meanrx['ProdId'] >= 917]
rx580 = meanrx[meanrx['ProdId'] <= 930]
#makegraph(rx580, 'RX 580')
rx580 = rx580.groupby('TimeId').mean()
rx580 = rx580.reset_index()
rx580['TimeId'] = list([pd.to_datetime(x, format='%Y%m%d') for x in rx580['TimeId'].to_list()])
rx580 = rx580.set_index('TimeId').asfreq('d').reset_index()
#print('rx580: ')
#print(rx580)

vega56 = meanrx[meanrx['ProdId'] >= 931]
vega56 = meanrx[meanrx['ProdId'] <= 936]
#makegraph(vega56, 'vega 56')
vega56 = vega56.groupby('TimeId').mean()
vega56 = vega56.reset_index()
vega56['TimeId'] = list([pd.to_datetime(x, format='%Y%m%d') for x in vega56['TimeId'].to_list()])
vega56 = vega56.set_index('TimeId').asfreq('d').reset_index()
#print('vega56: ')
#print(vega56)

vega64 = meanrx[meanrx['ProdId'] >= 937]
vega64 = meanrx[meanrx['ProdId'] <= 944]
#makegraph(vega64, 'vega 64')
vega64 = vega64.groupby('TimeId').mean()
vega64 = vega64.reset_index()
vega64['TimeId'] = list([pd.to_datetime(x, format='%Y%m%d') for x in vega64['TimeId'].to_list()])
vega64 = vega64.set_index('TimeId').asfreq('d').reset_index()
#print('vega64: ')
#print(vega64)

mean960 = data960.sort_values('TimeId', ascending=False)
mean960 = mean960[mean960['TimeId'] >= 20160824]
mean960 = mean960[mean960['TimeId'] <= 20180315]
mean960 = mean960[mean960['ProdId'] >= 1879]
mean960 = mean960[mean960['ProdId'] <= 1897]
#makegraph(mean960, 'GTX 960')
mean960 = mean960.groupby('TimeId').mean()
mean960 = mean960.reset_index()
#print('mean960: ')
#print(mean960)

mean970_980 = data970_80.sort_values('TimeId', ascending=False)
mean970_980 = mean970_980[mean970_980['TimeId'] >= 20160824]
mean970_980 = mean970_980[mean970_980['TimeId'] <= 20180315]
mean970 = mean970_980[mean970_980['ProdId'] >= 1907]
mean970 = mean970_980[mean970_980['ProdId'] <= 1919]
#makegraph(mean970, 'GTX 970')
mean970 = mean970.groupby('TimeId').mean()
mean970 = mean970.reset_index()
#print('mean970: ')
#print(mean970)

mean980 = mean970_980[mean970_980['ProdId'] >= 1920]
mean980 = mean970_980[mean970_980['ProdId'] <= 1934]
#makegraph(mean980, 'GTX 980')
mean980 = mean980.groupby('TimeId').mean()
mean980 = mean980.reset_index()
#print('mean980: ')
#print(mean980)

mean980ti = mean970_980[mean970_980['ProdId'] >= 1935]
mean980ti = mean970_980[mean970_980['ProdId'] <= 1947]
#makegraph(mean980ti, 'GTX 980ti')
mean980ti = mean980ti.groupby('TimeId').mean()
mean980ti = mean980ti.reset_index()
#print('mean980ti: ')
#print(mean980ti)

mean10 = data10.sort_values('TimeId', ascending=False)
mean10 = mean10[mean10['TimeId'] >= 20160824]
mean10 = mean10[mean10['TimeId'] <= 20180315]
mean1050 = mean10[mean10['ProdId'] >= 995]
mean1050 = mean10[mean10['ProdId'] <= 1006]
#makegraph(mean1050, 'GTX 1050')
mean1050 = mean1050.groupby('TimeId').mean()
mean1050 = mean1050.reset_index()
#print('mean1050: ')
#print(mean1050)

mean1050ti = mean10[mean10['ProdId'] >= 1007]
mean1050ti = mean10[mean10['ProdId'] <= 1018]
#makegraph(mean1050ti, 'GTX 1050ti')
mean1050ti = mean1050ti.groupby('TimeId').mean()
mean1050ti = mean1050ti.reset_index()
#print('mean1050ti: ')
#print(mean1050ti)

mean1060 = mean10[mean10['ProdId'] >= 1019]
mean1060 = mean10[mean10['ProdId'] <= 1045]
#makegraph(mean1060, 'GTX 1060')
mean1060 = mean1060.groupby('TimeId').mean()
mean1060 = mean1060.reset_index()
#print('mean1060: ')
#print(mean1060)

mean1070 = mean10[mean10['ProdId'] >= 1046]
mean1070 = mean10[mean10['ProdId'] <= 1060]
#makegraph(mean1070, 'GTX 1070')
mean1070 = mean1070.groupby('TimeId').mean()
mean1070 = mean1070.reset_index()
#print('mean1070: ')
#print(mean1070)

mean1070ti = mean10[mean10['ProdId'] >= 1061]
mean1070ti = mean10[mean10['ProdId'] <= 1072]
#makegraph(mean1070ti, 'GTX 1070ti')
mean1070ti = mean1070ti.groupby('TimeId').mean()
mean1070ti = mean1070ti.reset_index()
#print('mean1070ti: ')
#print(mean1070ti)

mean1080 = mean10[mean10['ProdId'] >= 1073]
mean1080 = mean10[mean10['ProdId'] <= 1087]
#makegraph(mean1080, 'GTX 1080')
mean1080 = mean1080.groupby('TimeId').mean()
mean1080 = mean1080.reset_index()
#print('mean1080: ')
#print(mean1080)

mean1080ti = mean10[mean10['ProdId'] >= 1088]
mean1080ti = mean10[mean10['ProdId'] <= 1107]
# makegraph(mean1080ti, 'GTX 1080ti')
mean1080ti = mean1080ti.groupby('TimeId').mean()
mean1080ti = mean1080ti.reset_index()
#print('mean1080ti: ')
#print(mean1080ti)

meantitan = datatitan.sort_values('TimeId', ascending=False)
meantitan = meantitan[meantitan['TimeId'] >= 20160824]
meantitan = meantitan[meantitan['TimeId'] <= 20180315]
# makegraph(meantitan, 'GTX titan')
meantitan = meantitan.groupby('TimeId').mean()
meantitan = meantitan.reset_index()
# print('meantitan: ')
# print(meantitan)

# create dataframes for linear regression

reg_df = pd.DataFrame(columns=['date', 'rx_560_price', 'rx_570_price', 'rx_580_price', 'vega_56_price', 'vega_64_price',
                                   '960_price', '970_price', '980_price', '980ti_price', '1050_price', '1050ti_price',
                                    '1060_price', '1070_price', '1070ti_price', '1080_price', '1080ti_price','titan_price',
                                   'btc_price', 'eth_price', 'ltc_price'])

reg_df['date'] = dataBTC['Date']
reg_df = reg_df.sort_values(by=['date'])
reg_df = reg_df.reset_index()
reg_df = reg_df.reset_index()
reg_df['rx_560_price'] = rx560['Price_USD']
reg_df['rx_560_price'] = reg_df['rx_560_price'].fillna(rx560['Price_USD'].mean())
reg_df['rx_570_price'] = rx570['Price_USD']
reg_df['rx_570_price'] = reg_df['rx_570_price'].fillna(rx570['Price_USD'].mean())
reg_df['rx_580_price'] = rx580['Price_USD']
reg_df['rx_580_price'] = reg_df['rx_580_price'].fillna(rx580['Price_USD'].mean())
reg_df['vega_56_price'] = vega56['Price_USD']
reg_df['vega_56_price'] = reg_df['vega_56_price'].fillna(vega56['Price_USD'].mean())
reg_df['vega_64_price'] = vega64['Price_USD']
reg_df['vega_64_price'] = reg_df['vega_64_price'].fillna(vega64['Price_USD'].mean())
reg_df['960_price'] = mean960['Price_USD']
reg_df['970_price'] = mean970['Price_USD']
reg_df['980_price'] = mean980['Price_USD']
reg_df['980ti_price'] = mean980ti['Price_USD']
reg_df['1050_price'] = mean1050['Price_USD']
reg_df['1050ti_price'] = mean1050ti['Price_USD']
reg_df['1060_price'] = mean1060['Price_USD']
reg_df['1070_price'] = mean1070['Price_USD']
reg_df['1070ti_price'] = mean1070ti['Price_USD']
reg_df['1080_price'] = mean1080['Price_USD']
reg_df['1080ti_price'] = mean1080ti['Price_USD']
reg_df['titan_price'] = meantitan['Price_USD']
reg_df['titan_price'] = reg_df['titan_price'].fillna(meantitan['Price_USD'].mean())
reg_df.drop('index', inplace=True, axis=1)
reg_df['btc_price'] = dataBTC['Price']
reg_df['eth_price'] = dataETH['Price']
reg_df['ltc_price'] = dataLTC['Price']
print("Final dataframe")
print(reg_df)

# Visualization of Titan Card with ETH Price

def graphCardCrypto():
    fig, ax = plt.subplots()
    reg_df['eth_price'] = reg_df['eth_price'].astype(float)
    reg_df['btc_price'] = reg_df['btc_price'].astype(float)
    reg_df['ltc_price'] = reg_df['ltc_price'].astype(float)
    ax.set_ylabel('Crypto Prices')
    ax.plot(reg_df['date'], reg_df['btc_price'], color='blue', linewidth=1, label='BTC')
    ax.plot(reg_df['date'], reg_df['eth_price'], color='green', linewidth=1, label='ETH')
    ax.plot(reg_df['date'], reg_df['ltc_price'], color='red', linewidth=1, label='LTC')
    #ax.plot(reg_df['date'], reg_df['rx_560_price'], color='blue', linewidth=1, label='RX-560')
    #ax.plot(reg_df['date'], reg_df['rx_570_price'], color='green', linewidth=1, label='RX-570')
    #ax.plot(reg_df['date'], reg_df['rx_580_price'], color='red', linewidth=1, label='RX-580')
    #ax.plot(reg_df['date'], reg_df['vega_56_price'], color='cyan', linewidth=1, label='RX-Vega56')
    #x.plot(reg_df['date'], reg_df['vega_64_price'], color='magenta', linewidth=1, label='RX-Vega64')
    #ax.plot(reg_df['date'], reg_df['960_price'], color='yellow', linewidth=1, label='GTX-960')
    #ax.plot(reg_df['date'], reg_df['970_price'], color='purple', linewidth=1, label='GTX-970')
    #ax.plot(reg_df['date'], reg_df['980_price'], color='pink', linewidth=1, label='GTX-980')
    #ax.plot(reg_df['date'], reg_df['980ti_price'], color='orange', linewidth=1, label='GTX-980ti')
    #ax.plot(reg_df['date'], reg_df['1050_price'], color='brown', linewidth=1, label='GTX-1050')
    #ax.plot(reg_df['date'], reg_df['1050ti_price'], color='gray', linewidth=1, label='GTX-1050ti')
    #ax.plot(reg_df['date'], reg_df['1060_price'], color='olive', linewidth=1, label='GTX-1060')
    #ax.plot(reg_df['date'], reg_df['1070_price'], color='lime', linewidth=1, label='GTX-1070')
    #ax.plot(reg_df['date'], reg_df['1070ti_price'], color='maroon', linewidth=1, label='GTX-1070ti')
    #ax.plot(reg_df['date'], reg_df['1080_price'], color='navy', linewidth=1, label='GTX-1080')
    #ax.plot(reg_df['date'], reg_df['1080ti_price'], color='lightsteelblue', linewidth=1, label='GTX-1080ti')
    #ax.plot(reg_df['date'], reg_df['titan_price'], color='lightsalmon', linewidth=1, label='GTX-Titan')
    ax.legend()
    ax.tick_params(axis='y')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    plt.tight_layout()
    fig.tight_layout()
    plt.show()

# graphCardCrypto()


# data splitting
converted_data = reg_df[['rx_560_price', 'rx_570_price', 'rx_580_price', 'vega_56_price', 'vega_64_price',
                                   '960_price', '970_price', '980_price', '980ti_price', '1050_price', '1050ti_price',
                                    '1060_price', '1070_price', '1070ti_price', '1080_price', '1080ti_price',
                             'titan_price']].apply(lambda x: [x['rx_560_price'], x['rx_570_price'], x['rx_580_price'],
                                                              x['vega_56_price'], x['vega_64_price'], x['960_price'], x['970_price'],
                                                              x['980_price'], x['980ti_price'], x['1050_price'], x['1050ti_price'],
                                                              x['1060_price'], x['1070_price'], x['1070ti_price'], x['1080_price'],
                                                              x['1080ti_price'], x['titan_price']], axis=1)
gpuArray = list(converted_data)

# linear regression fitting

x_train, x_test, y_train, y_test = train_test_split(gpuArray, reg_df['btc_price'], test_size=0.1, random_state=0)
regBTC = LinearRegression()
regBTC.fit(x_train, y_train)
predBTC = regBTC.predict(x_test)
scoreBTC = regBTC.score(x_test, y_test)
x_trainArr = np.array(x_train).astype(float)
x_trainArr = np.arange(0, len(x_train), 1)
y_trainArr = np.array(y_train).astype(float)

x_train2, x_test2, y_train2, y_test2 = train_test_split(gpuArray, reg_df['eth_price'], test_size=0.1, random_state=0)
regETH = LinearRegression()
regETH.fit(x_train2, y_train2)
predETH = regETH.predict(x_test2)
scoreETH = regETH.score(x_test2, y_test2)
x_train2Arr = np.array(x_train2).astype(float)
x_train2Arr = np.arange(0, len(x_train2), 1)
y_train2Arr = np.array(y_train2).astype(float)

x_train3, x_test3, y_train3, y_test3 = train_test_split(gpuArray, reg_df['ltc_price'], test_size=0.1, random_state=0)
regLTC = LinearRegression()
regLTC.fit(x_train3, y_train3)
predLTC = regLTC.predict(x_test3)
scoreLTC = regLTC.score(x_test3, y_test3)
x_train3Arr = np.array(x_train3).astype(float)
x_train3Arr = np.arange(0, len(x_train3), 1)
y_train3Arr = np.array(y_train3).astype(float)


print('')
print('Linear regression score of BTC: ' + str((scoreBTC * 100)))
print("Mean squared error: %.2f" % mean_squared_error(y_test, predBTC))
plt.scatter(x_trainArr, y_trainArr, color='black', alpha=0.5)
plt.plot(x_trainArr, regBTC.predict(x_train), color='blue', linewidth=2)
plt.show()

print('Linear regression score of ETH: ' + str((scoreETH * 100)))
print("Mean squared error: %.2f" % mean_squared_error(y_test2, predETH))
plt.scatter(x_train2Arr, y_train2Arr, color='black', alpha=0.5)
plt.plot(x_train2Arr, regETH.predict(x_train2), color='blue', linewidth=2)
#plt.show()

print('Linear regression score of LTC: ' + str((scoreLTC * 100)))
print("Mean squared error: %.2f" % mean_squared_error(y_test3, predLTC))
plt.scatter(x_train3Arr, y_train3Arr, color='black', alpha=0.5)
plt.plot(x_train3Arr, regLTC.predict(x_train3), color='blue', linewidth=2)
#plt.show()


# ridge regression fitting

x_train4, x_test4, y_train4, y_test4 = train_test_split(gpuArray, reg_df['btc_price'], test_size=0.1, random_state=0)
ridgeBTC = linear_model.Ridge(alpha=.5)
ridgeBTC.fit(x_train4, y_train4)
ridgepredBTC = ridgeBTC.predict(x_test4)
ridgeScoreBTC = ridgeBTC.score(x_test4, y_test4)
x_train4Arr = np.array(x_test4).astype(float)
x_train4Arr = np.arange(0, len(x_test4), 1)
y_train4Arr = np.array(y_test4).astype(float)

x_train5, x_test5, y_train5, y_test5 = train_test_split(gpuArray, reg_df['btc_price'], test_size=0.1, random_state=0)
ridgeETH = linear_model.Ridge(alpha=.5)
ridgeETH.fit(x_train5, y_train5)
ridgepredETH = ridgeETH.predict(x_test5)
ridgeScoreETH = ridgeETH.score(x_test5, y_test5)
x_train5Arr = np.array(x_test5).astype(float)
x_train5Arr = np.arange(0, len(x_test5), 1)
y_train5Arr = np.array(y_test5).astype(float)

x_train6, x_test6, y_train6, y_test6 = train_test_split(gpuArray, reg_df['btc_price'], test_size=0.1, random_state=0)
ridgeLTC = linear_model.Ridge(alpha=.5)
ridgeLTC.fit(x_train6, y_train6)
ridgepredLTC = ridgeLTC.predict(x_test6)
ridgeScoreLTC = ridgeLTC.score(x_test6, y_test6)
x_train6Arr = np.array(x_train6).astype(float)
x_train6Arr = np.arange(0, len(x_train6), 1)
y_train6Arr = np.array(y_train6).astype(float)

print('')
print('Ridge regression score of BTC: ' + str((ridgeScoreBTC * 100)))
plt.scatter(x_train4Arr, y_train4Arr, color='black', alpha=0.5)
plt.plot(x_train4Arr, ridgeBTC.predict(x_test4), color='blue', linewidth=2)
#plt.show()

print('Ridge regression score of ETH: ' + str((ridgeScoreETH * 100)))
plt.scatter(x_train5Arr, y_train5Arr, color='black', alpha=0.5)
plt.plot(x_train5Arr, ridgeETH.predict(x_test5), color='blue', linewidth=2)
#plt.show()

print('Ridge regression score of LTC: ' + str((ridgeScoreLTC * 100)))
plt.scatter(x_train6Arr, y_train6Arr, color='black', alpha=0.5)
plt.plot(x_train6Arr, ridgeLTC.predict(x_train6), color='blue', linewidth=2)
#plt.show()

# decision tree regressor model fitting (higher score than other models but inconsistent)

x_train7, x_test7, y_train7, y_test7 = train_test_split(gpuArray, reg_df['btc_price'], test_size=0.2, random_state=0)
clfBTC = tree.DecisionTreeRegressor()
clfBTC.fit(x_train7, y_train7)
clfpredBTC = clfBTC.predict(x_test7)
clfScoreBTC = clfBTC.score(x_test7, y_test7)
x_train7Arr = np.array(x_train7).astype(float)
x_train7Arr = np.arange(0, len(x_train7), 1)
y_train7Arr = np.array(y_train7).astype(float)

x_train8, x_test8, y_train8, y_test8 = train_test_split(gpuArray, reg_df['btc_price'], test_size=0.2, random_state=0)
clfETH = tree.DecisionTreeRegressor()
clfETH.fit(x_train8, y_train8)
clfpredETH = clfETH.predict(x_test8)
clfScoreETH = clfETH.score(x_test8, y_test8)

x_train9, x_test9, y_train9, y_test9 = train_test_split(gpuArray, reg_df['btc_price'], test_size=0.2, random_state=0)
clfLTC = tree.DecisionTreeRegressor()
clfLTC.fit(x_train9, y_train9)
clfpredLTC = clfLTC.predict(x_test9)
clfScoreLTC = clfLTC.score(x_test9, y_test9)

print('')
print('Decision tree regressor score of BTC: ' + str((clfScoreBTC * 100)))
plt.scatter(x_train7Arr, y_train7Arr, color='black', alpha=0.5)
plt.plot(x_train7Arr, ridgeBTC.predict(x_train7), color='blue', linewidth=2)
plt.show()
print('Decision tree regressor score of ETH: ' + str((clfScoreETH * 100)))
print('Decision tree regressor score of LTC: ' + str((clfScoreLTC * 100)))



