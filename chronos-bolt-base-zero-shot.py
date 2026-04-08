 # -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:30:34 2025

@author: echibout
"""

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Charger le fichier CSV avec délimiteur ;
df = pd.read_csv("C:/Users/echibout/Desktop/csvfiles.evora.2016/evora.measured.standard.PT0002000000000046BU.201601010000.201612312359.csv", sep=";")
print(f"Nombre de lignes dans le CSV : {len(df)}")

maison_testée=46

# Convertir la colonne TimeMeas_UTC en format datetime
df["TimeMeas_UTC"] = pd.to_datetime(df["TimeMeas_UTC"], format="%Y%m%d%H%M")
print("Premiers timestamps après conversion :")
print(df["TimeMeas_UTC"].head())
print("Derniers timestamps après conversion :")
print(df["TimeMeas_UTC"].tail())

# Ajouter une colonne item_id
df["item_id"] = "electricity"

# Trier par timestamp
df = df.sort_values("TimeMeas_UTC")

# Calculer la taille du dataset d'entraînement (80%)
train_size = int(0.8 * len(df))
print(f"Taille entraînement : {train_size}, Taille test : {len(df) - train_size}")

# Vérifier que le dataset n'est pas trop petit
if train_size == 0 or len(df) - train_size == 0:
    raise ValueError("Le dataset est trop petit pour être découpé en 80%/20%")

# Découper en entraînement (80%) et test (20%)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
print(f"Train_df : {len(train_df)} lignes, Test_df : {len(test_df)} lignes")

# Convertir en TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(
    train_df,
    id_column="item_id",
    timestamp_column="TimeMeas_UTC"
)
test_data = TimeSeriesDataFrame.from_data_frame(
    test_df,
    id_column="item_id",
    timestamp_column="TimeMeas_UTC"
)

# Définir la fréquence (15 minutes)
freq = "15T"
prediction_length = 96  # 96 intervalles de 15 min = 24h

# Initialiser le prédicteur
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="PowerMeas_EAI",
    eval_metric="MASE",
    freq=freq
)

# Configurer Chronos-Bolt en mode zero-shot
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": {
            "model_path": "bolt_base",
            "ag_args": {"name_suffix": "ZeroShot"}
        }
    },
    skip_model_selection=True
)

# Trouver les points à midi dans test_data
test_timestamps = test_data.index.get_level_values("timestamp")
test_midday = test_data[
    (test_timestamps.hour == 12) & (test_timestamps.minute == 0)
]

if len(test_midday) == 0:
    print("Aucun timestamp exact à 12h00. Recherche du point le plus proche de midi par jour...")
    test_df["date"] = test_df["TimeMeas_UTC"].dt.date
    midday_approx = test_df.groupby("date").apply(
        lambda x: x.iloc[(x["TimeMeas_UTC"] - pd.to_datetime(x["date"].iloc[0].strftime("%Y-%m-%d 12:00"))).abs().argmin()]
    )
    test_midday = TimeSeriesDataFrame.from_data_frame(
        midday_approx,
        id_column="item_id",
        timestamp_column="TimeMeas_UTC"
    )
print(f"Points à midi dans test_data : {len(test_midday)}")
print("Timestamps à midi :")
print(test_midday.index.get_level_values("timestamp"))

# Générer des prédictions pour chaque point de test_midday
all_pred_timestamps = []
all_pred_values = []
midday_timestamps = test_midday.index.get_level_values("timestamp")

for i, midday in enumerate(midday_timestamps):
    # Créer un sous-ensemble de données jusqu'à ce midi pour la prédiction
    data_up_to_midday = test_data[test_data.index.get_level_values("timestamp") <= midday]
    if len(data_up_to_midday) == 0:
        continue
    # Faire la prédiction pour les 24h suivantes
    pred = predictor.predict(data_up_to_midday)
    pred_timestamps = pd.date_range(start=midday, periods=prediction_length, freq=freq)
    pred_values = pred["mean"].iloc[-prediction_length:]  # Prendre les dernières 96 valeurs
    all_pred_timestamps.extend(pred_timestamps)
    all_pred_values.extend(pred_values)

# Créer un DataFrame pour les prédictions
pred_df = pd.DataFrame({"TimeMeas_UTC": all_pred_timestamps, "Predictions": all_pred_values})
print(f"Nombre total de prédictions : {len(pred_df)}")
print("Période couverte par les prédictions :")
print(f"Du {pred_df['TimeMeas_UTC'].min()} au {pred_df['TimeMeas_UTC'].max()}")

# Visualisation avec Plotly : Graphique 1 (données brutes)
fig = go.Figure()

# Ajouter les données réelles
fig.add_trace(go.Scatter(
    x=test_df["TimeMeas_UTC"],
    y=test_df["PowerMeas_EAI"],
    mode="lines",
    name="Test réel",
    line=dict(color="blue")
))

# Ajouter les prédictions
fig.add_trace(go.Scatter(
    x=pred_df["TimeMeas_UTC"],
    y=pred_df["Predictions"],
    mode="lines",
    name="Prédictions Zero-Shot",
    line=dict(color="orange")
))

# Ajouter les lignes verticales pour les midis
#for midday in midday_timestamps:
    #fig.add_shape(
    #    type="line",
    #    x0=midday,
    #    x1=midday,
    #    y0=test_df["PowerMeas_EAI"].min(),
    #    y1=test_df["PowerMeas_EAI"].max(),
    #    line=dict(color="red", dash="dash", width=1)
    #)

#Layout update or 
fig.update_layout(
title="Prédictions Zero-Shot pour toute la période de test",
xaxis_title="Temps",
yaxis_title="Consommation électrique",
hovermode="x unified",
showlegend=True,
template="plotly_white")

# Afficher le graphique
fig.show()

# Exporter le graphique en HTML
fig.write_html("zero_shot_predictions_raw_"+str(maison_testée)+".html")
print("Graphique brut exporté sous 'zero_shot_predictions_raw.html'")