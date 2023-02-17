

Best_model_Forecasting


# Best Time Series Forecasting

Actualmente en cualquie área cuando se requiere
realizar un modelo de serie de tiempo se buscan un ETS o los modelos Arima.
Sin embargo no se consideran otros por cuales 
puede ser pueden ser utiles para generar 
series de tiempo.
La idea de este codigo es competir diferentes 
modelos y posteriormente obtener el mejor de 
ellos para el mejor pronostico.

Para la realización de la serie de tiempo es el siguiente:

Entrada(dataset) --> STL (Measuring strength
) --> Selección del Modelo ---> Resultado

* STL =Tt+St+Rt

  Fuerza de Tendencia
	Ft=max(0,1−Var(Rt)Var(Tt+Rt))

  Fuerza de Estacional
	Fs=max(0,1−Var(Rt)Var(St+Rt))

  Fuerza de Residual (remanente)
    Fr= ACF y PACF (Rezagos)

Donde eligira el mejor modelo de un Arima , ETS o XGBoost