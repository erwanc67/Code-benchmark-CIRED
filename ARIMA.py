# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:11:40 2025

@author: echibout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
# 1. Générons des données d'exemple (remplacez ceci par vos propres données)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset_path = f"C:/Users/echibout/Desktop/csvfiles.evora.2016/evora.measured.standard.PT0002000000000025AZ.201601010000.201612312359.csv"
timestamp_column = "TimeMeas_UTC"
data2 = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
    delimiter=';'
)



np.random.seed(42)
dates = data2['TimeMeas_UTC'].values
n_points = len(dates)
consumption = data2['PowerMeas_EAI'].values
data = pd.DataFrame({"date": dates, "consumption": consumption})
data.set_index("date", inplace=True)
df = pd.DataFrame({'date': dates, 'consumption': consumption})
df.set_index('date', inplace=True)


train_size = int(len(df) * 0.8)
train_data = df['consumption'][:train_size]
test_data = df['consumption'][train_size:]


def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('Statistique ADF :', result[0])
    print('p-valeur :', result[1])
    return result[1] <= 0.05

print("Stationnarité des données d'entraînement :")
if not check_stationarity(train_data):
    print("Les données ne sont pas stationnaires. Différenciation recommandée.")


model = auto_arima(train_data, seasonal=True, m=24,  # m=24 pour une saisonnalité horaire quotidienne (24 heures)
                   start_p=0, start_q=0, max_p=3, max_q=3, d=None,
                   trace=True, error_action='ignore', suppress_warnings=True,
                   stepwise=True)

print(f"Meilleurs paramètres ARIMA : {model.order}")
print(f"Paramètres saisonniers : {model.seasonal_order}")

# 5. Ajustement du modèle SARIMA
sarima_model = SARIMAX(train_data, order=model.order, seasonal_order=model.seasonal_order)
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

# 6. Prédiction sur toute la période de test
forecast = sarima_fit.forecast(steps=len(test_data))

# 7. Visualisation (uniquement période de test et prédictions)
plt.figure(figsize=(12,6))
plt.plot(test_data.index, test_data, label='Valeurs réelles (période de test)', color='green')
plt.plot(test_data.index, forecast, label='Prédictions', color='red', linestyle='--')
plt.title('Comparaison des valeurs réelles et prédites pour la période de test (SARIMA)')
plt.xlabel('Date')
plt.ylabel('Consommation (kWh)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 8. Évaluation : Calcul de l'erreur RMSE
rmse = np.sqrt(mean_squared_error(test_data, forecast))
print(f'RMSE sur la période de test : {rmse:.2f}')