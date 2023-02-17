import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error

import pmdarima as pm
from pmdarima.arima import StepwiseContext
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")




def Tranformation(tseries):
    from scipy import stats
    tseries_transf, lam = stats.boxcox(tseries)
    tseries_transf = pd.Series(tseries_transf,index = tseries.index)
    return tseries_transf, lam

def Inv_Tranformation(tseries, lam):
    from scipy.special import inv_boxcox
    tseries_transf = inv_boxcox(tseries,lam)
    return tseries_transf


def Outliers(tseries):
    from PyAstronomy import pyasl
    # Apply the generalized ESD
    Q1 = tseries.quantile(0.25)
    Q3 = tseries.quantile(0.75)

    IQR = Q3-Q1

    tope_max = Q3+1.5*IQR
    tope_min = Q1-1.5*IQR
    
    r = pyasl.generalizedESD(tseries, 10, 0.05, fullOutput=True)
    if r[0] == 0:
        return(tseries)
    else:
        tseries_prep = tseries.copy()
        tseries_prep.loc[tseries_prep>tope_max] = tope_max
        tseries_prep.loc[tseries_prep<tope_min] = tope_min
        return(tseries_prep)




def get_lags(tseries , nlags = 12, alpha = 0.05):
    
    
    acf, ci = sm.tsa.acf(tseries ,nlags = nlags, alpha=alpha)
    
    #Falta implementar pacg parcial
    #pacf, cp = sm.tsa.pacf(tseries ,nlags = nlags, alpha=alpha)
    
    lags,lpags = [], []
    
    for i in range(1,ci.shape[0]):
        
        if (np.sum(ci[i][:] > 0 ) == np.sum(ci[i][:] < 0 )) == False:
            lags.append(i)
            lpags.append(i)
        
    
    myset = set(lags).union(lpags)
    
    mylist = [x for x in myset]
    
    return mylist

def strength_trend(tseries, period = 12):
    
    
    res = STL(tseries,period=period).fit()

    Ft = 1- np.var(res.resid)/np.var(res.trend +res.resid)

    return Ft

def strength_seasonal(tseries, period = 12):
    
    
    res = STL(tseries,period=period).fit()
    
    Fs = 1- np.var(res.resid)/np.var(res.seasonal+res.resid)

    return Fs
    
def strength_residual(tseries, period = 12):
    
    
    res = STL(tseries,period=period).fit()
    
    Fr = get_lags(res.resid, nlags = 3)
    
    if len(Fr) == 0:
        
        Fr_b = False
    else:
        Fr_b = True

    return Fr_b


def trend_regression(tseries, steps = 3):
    
    x_train = np.array(range(0,tseries.size)).reshape(-1, 1)
    
    y_train = np.array(tseries).reshape(-1, 1)
    
    x_test  = np.array(range(0,tseries.size+steps)).reshape(-1, 1)
    
    #Linear Regression
    
    reg = linear_model.LinearRegression()
    
    reg.fit(x_train, y_train)
    reg.score(x_train, y_train)
    
    #Polynomial Regression

    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('linear', LinearRegression(fit_intercept=False))])

    model =  model.fit(x_train, y_train)
    model.score(x_train, y_train)
    
    if reg.score(x_train, y_train) > model.score(x_train, y_train):
        prediction = reg.predict(x_test[-steps:])
    else:
        prediction = model.predict(x_test[-steps:])
    
    return prediction


def XGBoost_timeseries(tseries, seasonal, error, steps = 3):
    res = STL(tseries,period=12).fit()
    

    
    if (seasonal and error) == True:
        
        datos_train = res.resid + res.seasonal
        
        if len(get_lags(datos_train , nlags = 12, alpha = 0.1)) == 0:
            xlags = [1,2,12]
        else:
            xlags = get_lags(datos_train , nlags = 12, alpha = 0.1)
            
        
        forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(random_state=123),
                lags = xlags
             )
        forecaster.fit(y=datos_train)
        prediction =forecaster.predict(steps=steps)
    
    elif seasonal == True:
        
        datos_train = res.seasonal
        
        if len(get_lags(datos_train , nlags = 12, alpha = 0.1)) == 0:
            xlags = [12]
        else:
            xlags = get_lags(datos_train , nlags = 12, alpha = 0.1)
        
        forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(random_state=123),
                lags = xlags
             )
        forecaster.fit(y=datos_train)
        prediction =forecaster.predict(steps=steps)
    
    else: 
        datos_train = res.resid
        
        forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(random_state=123),
                lags = get_lags(datos_train , nlags = 3)
             )
        forecaster.fit(y=datos_train)
        prediction =forecaster.predict(steps=steps)
    
    return(prediction)


def model_selection (df1):
    
    res = STL(df1,period=12).fit()
    
    first_seasonal = (strength_seasonal(df1,12) > 0.4)
    second_seasonal = (df1.shape[0])>23
    
    if first_seasonal & second_seasonal:
        seasonal = True
        
        # Parametros ETS
        seasonal_ets = 'add'
        
        # Parametros ARIMA
        if (df1.shape[0]-1)>13:
            D_sarima = 1
            seasonal_sarima = True
            m_sarima = 12
            P_arima = Q_arima = 2
        else:
            seasonal_sarima = False
            D_sarima = None
            m_sarima = 1
            P_arima = Q_arima = 1
        
    else:
        seasonal = False
        
        # Parametros ETS
        seasonal_ets = None
        
        # Parametros ARIMA
        seasonal_sarima = False
        D_sarima = None
        m_sarima = 1
        P_arima = Q_arima = 1
    
    if strength_trend(df1,12) > 0.4:
        trend = True
        
        # Parametros ETS
        trend_ets = 'add'
        
        # Parametros ARIMA
        d_sarima = 1
        
    else:
        trend = False
        
        # Parametros ETS
        trend_ets = None
        
        # Parametros ARIMA
        d_sarima = None
        
    if strength_residual(df1) == True:
        error = True
        
        # Parametros ETS
        error_ets = 'add'
        
        # Parametros ARIMA
        p_arima = q_arima = max(get_lags(res.resid, nlags = 3))
    
    else:
        error = False
        error_ets = 'add'
        p_arima = q_arima = 0
    
    
    
    return seasonal, trend, error,  seasonal_ets, D_sarima, seasonal_sarima, m_sarima, P_arima , Q_arima, trend_ets, d_sarima, error_ets, p_arima , q_arima
    


def modelado(df1,steps = 3, best = ['Sarima','ETS', 'XGB']):
    
    seasonal, trend, error, seasonal_ets, D_sarima, seasonal_sarima, m_sarima, P_arima , Q_arima, trend_ets,d_sarima, error_ets, p_arima , q_arima = model_selection(df1)
    
    
    best = np.array(best)
    
    if best.size>1 or ((best == 'Sarima') or (best == "MIX")):
        ## Arima Model
        with StepwiseContext(max_steps=15,max_dur=1):
            stepwise_fit = pm.auto_arima(df1, start_p = 0, start_q = 0, 
                                         max_p = p_arima, max_q = q_arima, m = m_sarima, 
                                         start_P = 0,star_Q = 0, max_P=P_arima, max_Q=Q_arima ,seasonal = seasonal_sarima, 
                                         d = d_sarima, D = D_sarima, 
                                         error_action ='ignore',   
                                         suppress_warnings = True,  
                                         stepwise = True) 
        
    
    if best.size>1 or ((best == 'ETS') or (best == "MIX")):
        ## ETS Model
        model = ETSModel(df1, error= error_ets, trend = trend_ets, seasonal= seasonal_ets ,damped_trend=False,
                         seasonal_periods=12).fit()
    
    
    if best.size>1 or ((best == 'XGB') or (best == "MIX")):
        ## XGBoost and Linear Model
        if (seasonal or error) == True:
            res = STL(df1,period=12).fit()
            res.trend
            Ln = trend_regression(res.trend, steps = steps)
            Ln = Ln.flat[:] 
            XG = XGBoost_timeseries(df1, seasonal, error, steps = steps)
            modelXGB = np.add(XG,Ln)
    
        else:
            res = STL(df1,period=12).fit()
            res.trend
            Ln = trend_regression(res.trend, steps = steps)
            modelXGB = Ln.flat[:]
            
    if best.size>1:
        
        print()
        
        dfmodel= pd.DataFrame({'Sarima':stepwise_fit.predict(steps),'ETS': model.forecast(steps).values
                             , 'XGB': modelXGB.values})
        
        dfmodel['MIX'] = (dfmodel.Sarima + dfmodel.ETS + dfmodel.XGB)/3
        
        
        if df1.index.inferred_type == 'datetime64':
            start = df1.index.max() + pd.DateOffset(months=1)
            #dfmodel.index = pd.date_range(start=start, periods = steps,freq='M' )
            dfmodel.index = pd.period_range(start = start , periods = steps, freq='M').astype(str)+'-'+'01'
            dfmodel.index = pd.to_datetime(dfmodel.index)
        
    else:
        
        best = best.item()
        
        if best == 'Sarima':
            prediction = stepwise_fit.predict(steps)
        
        elif best =='MIX':
            prediction =(stepwise_fit.predict(steps) + model.forecast(steps) + modelXGB)/3
        
        elif best == 'ETS':
            prediction = model.forecast(steps)
        
        else:
            prediction = modelXGB
            
        dfmodel = pd.DataFrame({ best:prediction})
        
        if df1.index.inferred_type == 'datetime64':
            start = df1.index.max() + pd.DateOffset(months=1)
            #dfmodel.index = pd.date_range(start=start, periods = steps,freq='M' )
            dfmodel.index = pd.period_range(start = start , periods = steps, freq='M').astype(str)+'-'+'01'
            dfmodel.index = pd.to_datetime(dfmodel.index)
        
    
    return dfmodel


def perfomance(y_true, df_model):
    # Arima
    mae_arima = mean_absolute_error(y_true, df_model.Sarima)
    
    #ETS
    mae_ets = mean_absolute_error(y_true, df_model.ETS)
    

    #XGBOOST
    mae_xgb = mean_absolute_error(y_true, df_model.XGB)
    
    #Mixed
    mae_mix = mean_absolute_error(y_true, (df_model.XGB+df_model.ETS+df_model.Sarima)/3)

    
    best_mae = pd.DataFrame({'metric': [mae_arima,mae_ets, mae_xgb, mae_mix ]},
                             index = ['Sarima','ETS', 'XGB',"MIX"])
    
    best_std = best_mae.min().values[0]
    
    best_mae = best_mae.loc[ best_mae.min().values[0] == best_mae.metric].index
    
    return best_mae, best_std


def best_model_prediction(df1, steps= 3, prediction = 3, anomalies = False, box_cox = False):
    
    if anomalies == True:
        df1 = Outliers(df1)
    
    if box_cox == True:
        df1, lam = Tranformation(df1)
    
    df1 = df1.astype('float64')
    df1_train = df1[:-steps]
    df1_test  = df1[-steps:]

    df_model = modelado(df1_train,steps = steps)

    best_model = perfomance(df1_test, df_model)
    
    best_model_std = best_model[1]
    
    best_model = best_model[0]
    
    df_validation = modelado(df1, steps = prediction ,best = best_model)
    
    df_validation['sample'] = 'validation'
    
    model_test= df_model[[best_model.values.item()]]
    model_test['sample'] = 'testing'
    
    df_best_model = pd.concat([model_test,df_validation])
    df_best_model['model'] = best_model.values.item()
    df_best_model['std'] = best_model_std
    
    if box_cox == True:
        dff = Inv_Tranformation(df_best_model[df_best_model.columns[0]],lam)
        df_best_model[df_best_model.columns[0]] = dff
    
    return df_best_model
    
