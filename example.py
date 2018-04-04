import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import math

#解析器  將民國年分對應到西元年分  以datetime的形式呈現
def parser_data(x):
    y=int(x)+1911
    return pd.datetime.strptime(str(y), '%Y')
#解析器  將'1-01'  對應為 1991 1月 以datetime的形式呈現
def parser(x):
    return pd.datetime.strptime('199'+str(x), '%Y-%m')


#取得MA後的結果(前後有幾個NA版)
def get_ma_list1(q,list):
    #前後有幾個NA
    na_num=int((q-1)/2)

    #實際ma_list有值的長度
    ma_num=len(list)-2*na_num    
    ma_list = []
    #計算移動平均
    for i in range(0,ma_num):
        ma_list.append(np.mean(list[i:i+q]))
    #前後補上NA
    while na_num >0:
        ma_list.insert(0,np.nan)
        ma_list.append(np.nan)
        na_num-=1
    return ma_list

    
#取得MA後的結果(前面有幾個NA版)
def get_ma_list(q,list):
    #前有幾個NA
    na_num=q
    #實際ma_list有值的長度
    
    ma_num=len(list)-na_num
    ma_list = []
    #計算移動平均
    for i in range(0,ma_num):
        ma_list.append(np.mean(list[i:i+q]))
    #前後補上NA
    while na_num >0:
        ma_list.insert(0,np.nan)
        na_num-=1
        
    return ma_list
    
    
    
#找到最適合差分的階數 (只找前8階)  階數最小且滿足ADF檢定p -value 小於0.05的
def best_diff(df, maxdiff = 8):
    p_set = []
    bestdiff = 0
    for i in range(0, maxdiff):
        temp=df.copy()
        if i>0:
            temp = temp.diff(i).dropna()
        p_set.append(ADF(temp.values)[1])
    i=0
    while i < len(p_set):
        if p_set[i] < 0.05:
            bestdiff = i
            break
        i += 1
    return bestdiff

def best_p_q(d, series):
    #一般而言階數不會超過 資料長度的 1/10
    pmax = int(len(series)/10)
    qmax = int(len(series)/10)
    #bic矩陣
    bic_matrix = []
    for p in range(pmax):
        tmp = []
        for q in range(qmax):
            try:
                tmp.append(ARIMA(series, order=(p,d,q)).fit(disp=0).bic)
            except:
                tmp.append(None)
            bic_matrix.append(tmp)
    
    #找出bic最小的 p,q參數
    bic_matrix = pd.DataFrame(bic_matrix) 
    p,q = bic_matrix.stack().idxmin() 
    return p,q
    

#資料來源路徑
data='data.csv'
#讀取資料(利用pandas的 read_csv函式)
data_df=pd.read_csv(data, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser_data)

#顯示讀取的資料
print(data_df)

#畫線
plt.plot(data_df.index,data_df.values, label='real')
#圖的x軸名稱
plt.xlabel('year')
#圖的y軸名稱
plt.ylabel('unit')
#圖的標題
plt.title('source data')
#顯示圖例的位置
plt.legend(loc="best")
#顯示圖
plt.show()

#Moving average fitting result (ref1:http://www.mcu.edu.tw/department/management/stat/ch_web/etea/Statistics-3-net/chap32time2.pdf)

#移動平均的參數q
ma_q=2
#利用自定義函式 取得移動平均的預測線
ma_list = get_ma_list(ma_q,data_df.values)
#作圖
plt.plot(data_df.index,data_df.values, label='real')
plt.plot(data_df.index,ma_list, color='red', linestyle='--', label='MA(%d)'%(ma_q))
plt.xlabel('year')
plt.ylabel('unit')
plt.title('MA fitting result')
plt.legend(loc="best")
plt.show()


#Moving average fitting result (ref2:https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)
ma_q=3
ma_list = pd.rolling_mean(data_df.values,ma_q)
plt.plot(data_df.index,data_df.values, label='real')
plt.plot(data_df.index,ma_list, color='red', linestyle='--', label='MA(%d)'%(ma_q))
plt.xlabel('year')
plt.ylabel('unit')
plt.title('MA fitting result')
plt.legend(loc="best")
plt.show()

#AR fitting result

#自迴歸的的參數p
ar_p=2
#取得以AR(2)fit過的模型
model_fit=AR(data_df).fit(ar_p)
#印出AR(2)的迴歸係數
print('AR(%d)迴歸係數:'%(ar_p),model_fit.params)

#利用AR(2)fit過的模型取得 時間段到預測期101年的預測結果
ar_result=model_fit.predict(end=pd.datetime.strptime(str(101+1911), '%Y')).tolist()
#由於使用AR(p)模型  因此 前p年沒有預測結果 補上NA
for i in range(0,ar_p):
    ar_result.insert(0,np.nan)
#印出101年的預測結果 在ar_result  index為最後的數值
print("民101年的預測結果:",ar_result[-1])

#作圖
plt.plot(data_df.index,data_df.values, label='real')
#x軸加上預測期(即民101年)
xlist=data_df.index.tolist()
xlist.append(str(101+1911))

plt.plot(xlist,ar_result, color='red', linestyle='--', label='AR(%d)'%(ar_p))
plt.xlabel('year')
plt.ylabel('unit')
plt.title('AR fitting result')
plt.legend(loc="best")
plt.show()



#Dickey-Fuller test(檢定是否為平穩序列)
#H0:存在unit root (非平穩序列)  H1:不存在unit root (平穩序列)
dftest = ADF(data_df.values, 1)

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


#一階差分
print(data_df.diff(1))
#由於一階差分後第一年的預測值會是na 因此透過dropna()將第一年排除
diff_df= data_df.diff(1).dropna()

#檢定差分後的序列是否滿足  Dickey-Fuller test檢定
dftest = ADF(diff_df.values, 1)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


#發現該自行建立的資料集數據太少難以看出平穩趨勢  因此改用另一範例資料集

#讀取時序資料(csv相關資料來源:https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period)
series = pd.read_csv('shampoo-sales.csv', parse_dates=[0], index_col=0,squeeze=True, date_parser=parser)
#顯示前5筆資料
print(series.head())

#作圖
plt.plot(series.index, series.values, label='real')
plt.xlabel('Month')
plt.ylabel('Sales')
#將x軸顯示的時間做格式轉換  並每隔6個挑一次顯示在 ticks
plt.xticks(series.index[::6], series.index.strftime('%Y-%b')[::6] )
plt.title('source data')
plt.show()


#一階差分後的series
diff_series=series.diff(1).dropna()
#作圖
plt.plot(diff_series.index, diff_series.values, label='real-diff')
plt.xlabel('Month')
plt.xticks(diff_series.index[::6], diff_series.index.strftime('%Y-%b')[::6] )
plt.ylabel('Sales-diff')
plt.title('source data-diff')
plt.show()

#檢定差分後的序列是否滿足  Dickey-Fuller test檢定
dftest = ADF(diff_series)
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


#如何挑選pq參數 
#1. ACF  (q) 與 PACF (p)   (參考:https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)
#2. AIC  或  BIC      實作在best_p_q function中

#一般而言階數不會超過 資料長度的 1/10
lag_acf =acf(diff_series, nlags=int(len(diff_series)/10))
lag_pacf =pacf(diff_series, nlags=int(len(diff_series)/10))

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(diff_series)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff_series)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(diff_series)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff_series)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()


#利用自訂函式找到最佳的差分階數
d = best_diff(series)
print("最佳差分階數:%d階" % d)

diff_series=series.diff(1).dropna()
#利用自訂函式取得bic最佳的pq參數
p, q = best_p_q(d, series)
#印出最佳的ARIMA參數
print("最佳參數組合ARIMA(%d,%d,%d)" %(p,d,q))

#使用最佳參數組合fit 
model = ARIMA(series, order=(p,d,q))

#由於在fit的過程中有很多線性迴歸的debug訊息出現 因此設定disp =0 關閉
model_fit = model.fit(disp=0)
#顯示ARIMA模型的摘要
print(model_fit.summary())

#取得ARIMA fit過的值  (因fit的對象是差分過的  因此比較對象應為差分過的資料集)
arima_predict_diff=model_fit.fittedvalues
#作圖
plt.plot(diff_series, label='real-diff')
plt.plot(arima_predict_diff, color='red', linestyle='--', label='ARIMA(%d,%d,%d)'%(p,d,q))
plt.xlabel('Month')
plt.xticks(diff_series.index[::6], diff_series.index.strftime('%Y-%b')[::6] )
plt.ylabel('Sales-diff')
plt.title('ARIMA fitting result')
plt.legend(loc="best")
plt.show()

#取得ARIMA 符合原數據(差分前的scale)的預測結果
r=model_fit.predict(typ='levels')
#作圖
plt.plot(series)
plt.plot(series.index[1:],r, color='red', linestyle='--', label='ARIMA(%d,%d,%d)'%(p,d,q))
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(series.index[::6], series.index.strftime('%Y-%b')[::6] )
plt.title('RMSE: %.4f'%math.sqrt(mean_squared_error(r,series[1:])))
plt.legend(loc="best")
plt.show()