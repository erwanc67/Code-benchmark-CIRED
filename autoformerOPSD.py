import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
import plotly.graph_objects as go
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dataset_path = "C:/Users/echibout/Downloads/household_data_15min_singleindex.csv"
timestamp_column = "utc_timestamp"
data2 = pd.read_csv(
    dataset_path,
    parse_dates=[timestamp_column],
    delimiter=','
)

# Filtrer pour 2016
data2['utc_timestamp'] = pd.to_datetime(data2['utc_timestamp'])
data2 = data2[data2['utc_timestamp'].dt.year == 2016]

# Sum residential consumption (cumulative)
residential_cols = [col for col in data2.columns if 'residential' in col]
cumulative_consumption = data2[residential_cols].sum(axis=1)

# Calculate actual consumption by subtracting consecutive timesteps
consumption = cumulative_consumption.diff().fillna(0)  # First value will be 0 due to no previous timestep
consumption[consumption < 0] = 0  # Ensure no negative values due to possible resets in cumulative data
data2["PowerMeas_EAI"] = consumption

np.random.seed(42)
dates = data2['utc_timestamp'].values
n_points = len(dates)
consumption = data2['PowerMeas_EAI'].values
data = pd.DataFrame({"date": dates, "consumption": consumption})
data.set_index("date", inplace=True)


def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])  # Fenêtre d’entrée
    return np.array(X)

seq_length = 672  # 1 semaine (672 points à 15 min d’intervalle)
X = create_sequences(data["consumption"].values, seq_length)

# Séparer entraînement (80%) et test (20%)
split = int(0.8 * len(X))
X_train = X[:split]
X_test = X[split:]
y_train = data["consumption"].values[seq_length:split + seq_length]  # dataset train
y_test = data["consumption"].values[split + seq_length:]  # dataset test

# Conversion en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Modèle AutoInformer simplifié
class AutoInformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, seq_length, dropout=0.1):
        super(AutoInformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = nn.Linear(d_model, 1)  # Prédiction point par point
    
    def forward(self, x):
        x = self.input_embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

#Format des vecteurs
#Entrée : (32, 672, 1) (32 séquences de 672 scalaires).
#Embedding + Pos : (32, 672, 64) (vecteurs enrichis avec position).
#Encodeur : (32, 672, 64) (vecteurs contextualisés).
#Décodeur : (32, 1) (prédictions scalaires)

# Paramètres du modèle
input_dim = 1
d_model = 64
n_heads = 8
lr=0.001
model = AutoInformer(input_dim, d_model, n_heads, seq_length)

# Entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = nn.MSELoss()  #Fonction de perte MSE

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


model.eval()
predictions = []
with torch.no_grad():
    for i in range(len(X_test) - 1):
        X_batch = X_test[i:i+1].to(device)
        pred = model(X_batch).cpu().numpy()
        predictions.append(pred.item())

# Conversion en tableau numpy et ajuster la longueur
predictions = np.array(predictions)
if len(predictions) > len(y_test):
    predictions = predictions[:len(y_test)]
elif len(predictions) < len(y_test):
    predictions = np.pad(predictions, (0, len(y_test) - len(predictions)), mode='edge')


dates = data.index[-len(y_test):]  # Les dates correspondant au datastet de test
df_plot = pd.DataFrame({
    "Date": dates,
    "Vrai (20% test)": y_test.flatten(),
    "Prédictions": predictions
})


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_plot["Date"],
    y=df_plot["Vrai (20% test)"],
    mode='lines',
    name='Vrai (20% test)',
    line=dict(color='blue', width=1),
    opacity=0.7
))

# Ajouter la courbe des prédictions
fig.add_trace(go.Scatter(
    x=df_plot["Date"],
    y=df_plot["Prédictions"],
    mode='lines',
    name='Prédictions',
    line=dict(color='orange', width=1),
    opacity=0.7
))
rmse = np.sqrt(np.mean((y_test.flatten() - predictions) ** 2))

fig.update_layout(
    title="Prédiction autoformer 2016 Résidentiel (epochs=5), nrmse ="+str(rmse/max(consumption)),
    xaxis_title="Date",
    yaxis_title="Consommation (kWh)",
    legend_title="Légende",
    hovermode="x unified",  # Afficher les valeurs au survol
    template="plotly_white"  # Thème clair
)

output_file = "prediction_day-ahead_electricite_autoformer_2016_residential.html"
fig.write_html(output_file, auto_open=True)
print(f"Graphique exporté avec succès dans : {output_file}")

fig.show()

# Calcul de la RMSE
rmse = np.sqrt(np.mean((y_test.flatten() - predictions) ** 2))
print(f"RMSE sur la période de test : {rmse:.4f}")