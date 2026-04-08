# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 00:10:11 2026

@author: echibout
"""



import argparse
import warnings
import sys
from pathlib import Path
from typing import Dict, Tuple
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Plotly pour visualisations interactives
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

FORECAST_HORIZON = 96  # 24h à 15min
LAG_FEATURES = [1, 2, 4, 96, 96*2, 96*7]  # 15min, 30min, 1h, 1j, 2j, 1sem

RESIDENTIAL_HOUSEHOLDS = {
    1: 'DE_KN_residential1_grid_import',
    2: 'DE_KN_residential2_grid_import', 
    3: 'DE_KN_residential3_grid_import',
    4: 'DE_KN_residential4_grid_import',
    5: 'DE_KN_residential5_grid_import',
    6: 'DE_KN_residential6_grid_import',
}

DATA_URL_15MIN = "https://data.open-power-system-data.org/household_data/2020-04-15/household_data_15min_singleindex.csv"


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def download_data(url: str, cache_dir: str = "./data") -> Path:
    """Télécharge le dataset."""
    import urllib.request
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    filename = url.split('/')[-1]
    filepath = cache_path / filename
    
    if filepath.exists():
        print(f"✓ Dataset trouvé en cache: {filepath}")
        return filepath
    
    print(f"⬇ Téléchargement du dataset...")
    urllib.request.urlretrieve(url, filepath)
    print(f"✓ Dataset téléchargé: {filepath}")
    return filepath


def load_and_prepare_data(filepath: Path, household_col: str) -> pd.DataFrame:
    """Charge et prépare les données avec traitement cumulatives."""
    print("\n📊 Chargement des données...")
    
    df = pd.read_csv(
        filepath,
        usecols=['utc_timestamp', household_col],
        parse_dates=['utc_timestamp'],
        index_col='utc_timestamp'
    )
    
    df.columns = ['cumulative']
    
    missing_pct = df['cumulative'].isnull().sum() / len(df) * 100
    if missing_pct > 0:
        print(f"  • Valeurs manquantes: {missing_pct:.2f}% → Interpolation")
        df['cumulative'] = df['cumulative'].interpolate(method='linear', limit_direction='both')
    
    print(f"  ⚙️  Traitement données cumulatives...")
    df['consumption'] = df['cumulative'].diff()
    df = df.iloc[1:].copy()
    
    negative_count = (df['consumption'] < 0).sum()
    if negative_count > 0:
        median_val = df[df['consumption'] > 0]['consumption'].median()
        df.loc[df['consumption'] < 0, 'consumption'] = median_val
    
    q99 = df['consumption'].quantile(0.99)
    df['consumption'] = df['consumption'].clip(upper=q99 * 3)
    
    print(f"  • Consommation moyenne: {df['consumption'].mean():.4f} kWh/15min")
    
    return df[['consumption']]


def split_data(df: pd.DataFrame, test_year: int = 2019) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pour XGBoost.
    
    Train: Toutes les années avant test_year (2015-2018)
    Test: TOUTE l'année test_year (janvier-décembre 2019)
    """
    df['year'] = df.index.year
    
    train_df = df[df['year'] < test_year].copy()
    test_df = df[df['year'] == test_year].copy()
    
    train_df = train_df.drop(['year'], axis=1)
    test_df = test_df.drop(['year'], axis=1)
    
    print(f"\n📂 Découpage des données:")
    print(f"  • Train: {len(train_df):,} timesteps (2015-{test_year-1}: {train_df.index[0].date()} → {train_df.index[-1].date()})")
    print(f"  • Test:  {len(test_df):,} timesteps (TOUTE l'année {test_year}: {test_df.index[0].date()} → {test_df.index[-1].date()})")
    
    return train_df, test_df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features pour XGBoost."""
    print("\n🔧 Création des features...")
    
    df = df.copy()
    
    # Features temporelles
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Features cycliques (heure)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Features cycliques (jour de la semaine)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Features cycliques (mois)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lags
    for lag in LAG_FEATURES:
        df[f'lag_{lag}'] = df['consumption'].shift(lag)
    
    # Rolling statistics (7 jours)
    df['rolling_mean_7d'] = df['consumption'].shift(1).rolling(window=96*7, min_periods=1).mean()
    df['rolling_std_7d'] = df['consumption'].shift(1).rolling(window=96*7, min_periods=1).std()
    
    # Rolling statistics (1 jour)
    df['rolling_mean_1d'] = df['consumption'].shift(1).rolling(window=96, min_periods=1).mean()
    df['rolling_std_1d'] = df['consumption'].shift(1).rolling(window=96, min_periods=1).std()
    
    print(f"  • Features créées: {len([c for c in df.columns if c != 'consumption'])}")
    
    return df


def train_xgboost(train_df: pd.DataFrame) -> object:
    """Entraîne le modèle XGBoost."""
    try:
        import xgboost as xgb
    except ImportError:
        print("✗ XGBoost non installé")
        print("  pip install xgboost")
        sys.exit(1)
    
    print("\n🎓 Entraînement XGBoost...")
    
    # Préparer données
    train_clean = train_df.dropna()
    
    feature_cols = [c for c in train_clean.columns if c != 'consumption']
    X_train = train_clean[feature_cols]
    y_train = train_clean['consumption']
    
    print(f"  • Échantillons entraînement: {len(X_train):,}")
    print(f"  • Features: {len(feature_cols)}")
    
    # Modèle XGBoost optimisé
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    # Entraînement
    start_time = time.time()
    model.fit(X_train, y_train, verbose=False)
    elapsed = time.time() - start_time
    
    print(f"  ✓ Entraînement terminé en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    return model


def generate_forecasts(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Génère les prévisions day-ahead."""
    print("\n🔮 Génération des prévisions...")
    
    # Combiner pour avoir accès aux lags
    full_df = pd.concat([train_df, test_df])
    full_df = create_features(full_df)
    
    # Index de départ du test
    test_start_idx = len(train_df)
    
    predictions = []
    actuals = []
    timestamps = []
    
    n_days = (len(test_df) - FORECAST_HORIZON) // FORECAST_HORIZON
    print(f"  • Nombre de jours: {n_days}")
    
    start_time = time.time()
    
    for day in tqdm(range(n_days), desc="  Progression"):
        day_start = test_start_idx + (day * FORECAST_HORIZON)
        day_end = day_start + FORECAST_HORIZON
        
        # Données du jour
        day_data = full_df.iloc[day_start:day_end].copy()
        
        if day_data.isnull().any().any():
            continue
        
        # Features
        feature_cols = [c for c in day_data.columns if c != 'consumption']
        X_day = day_data[feature_cols]
        y_day = day_data['consumption'].values
        
        # Prévision
        y_pred = model.predict(X_day)
        
        predictions.append(y_pred)
        actuals.append(y_day)
        timestamps.append(day_data.index[0])
    
    elapsed = time.time() - start_time
    
    if len(predictions) == 0:
        print("❌ Aucune prévision générée")
        sys.exit(1)
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    # ── Métriques ─────────────────────────────────────────────────────────────
    mae  = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

    # MAPE avec seuil de consommation minimale
    # -----------------------------------------------------------------------
    # La MAPE est définie comme mean(|y - ŷ| / y). Quand y ≈ 0 (foyer absent,
    # vacances, nuit profonde), le dénominateur est quasi-nul et le ratio
    # explose, donnant une MAPE artificiellement élevée et non représentative
    # de la qualité réelle du modèle sur les périodes d'occupation.
    #
    # Solution : on exclut les pas de temps où y < MAPE_THRESHOLD du calcul
    # de la MAPE uniquement. Les autres métriques (RMSE, MAE, NRMSE) restent
    # calculées sur l'ensemble complet pour ne pas biaiser la comparaison.
    #
    # Choix du seuil : 10% de la consommation moyenne, soit environ 5-6 Wh
    # par quart d'heure — en dessous, le foyer est considéré "absent".
    # Ce seuil et le nombre de pas filtrés sont reportés explicitement
    # pour transparence dans l'article.
    # -----------------------------------------------------------------------
    mean_actual     = np.mean(actuals)
    mape_threshold  = 0.10 * mean_actual          # 10% de la moyenne
    mask_active     = actuals >= mape_threshold    # True = pas de temps "actif"
    n_filtered      = np.sum(~mask_active)
    pct_filtered    = n_filtered / len(actuals) * 100

    if mask_active.sum() > 0:
        mape = np.mean(
            np.abs((actuals[mask_active] - predictions[mask_active])
                   / actuals[mask_active])
        ) * 100
    else:
        mape = np.inf

    # NRMSE normalisé par la moyenne
    # -----------------------------------------------------------------------
    # Choix justifié par la distribution de residential5 : la consommation
    # résidentielle à 15 min est fortement asymétrique (beaucoup de zéros,
    # pics ponctuels élevés). Dans ce cas, l'IQR est pathologiquement petit
    # (≈ 0.03 kWh) car Q25 et Q75 sont tous deux proches de zéro, ce qui
    # fait exploser le NRMSE au-delà de 1 même pour un modèle visuellement
    # performant.
    # La normalisation par la moyenne est plus représentative pour ce profil :
    #   NRMSE_mean = RMSE / mean(y)
    # Référence : Hyndman & Koehler (2006), IJF — recommandent la moyenne
    # pour les séries avec beaucoup de zéros ou distribution très asymétrique.
    # -----------------------------------------------------------------------
    mean_actual = np.mean(actuals)
    nrmse = rmse / mean_actual if mean_actual > 0 else np.inf

    # On conserve aussi l'IQR pour information (comparabilité littérature)
    q75, q25 = np.percentile(actuals, [75, 25])
    iqr   = q75 - q25
    nrmse_iqr = rmse / iqr if iqr > 0 else np.inf

    print(f"\n📊 Résultats:")
    print(f"  • Temps              : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  • NRMSE (moyenne)    : {nrmse:.4f}     ← métrique principale")
    print(f"  • NRMSE (IQR)        : {nrmse_iqr:.4f}     ← pour comparabilité littérature")
    print(f"  • RMSE               : {rmse:.4f} kWh  (sur {len(actuals):,} pas)")
    print(f"  • MAE                : {mae:.4f} kWh  (sur {len(actuals):,} pas)")
    print(f"  • MAPE               : {mape:.2f}%     ← sur pas actifs uniquement")
    print(f"  • Seuil MAPE         : {mape_threshold:.4f} kWh (= 10% × moyenne)")
    print(f"  • Pas filtrés (MAPE) : {n_filtered:,} / {len(actuals):,} ({pct_filtered:.1f}%)")
    print(f"  • Moyenne réelle     : {mean_actual:.4f} kWh")
    print(f"  • IQR réel           : {iqr:.4f} kWh")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'timestamps': timestamps,
        'metrics': {
            'NRMSE':          nrmse,
            'NRMSE_IQR':      nrmse_iqr,
            'RMSE':           rmse,
            'MAE':            mae,
            'MAPE':           mape,
            'mape_threshold': mape_threshold,
            'n_filtered':     int(n_filtered),
            'pct_filtered':   round(pct_filtered, 1),
        }
    }


def create_interactive_viz(results: Dict, household: str, test_index: pd.DatetimeIndex, output_dir: str = "./visualizations"):
    """Crée visualisations interactives Plotly."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    predictions = results['predictions']
    actuals = results['actuals']
    timestamps = results['timestamps']
    
    print(f"\n📈 Création visualisations interactives...")
    
    # Créer index temporel complet
    full_dates = []
    full_actuals = []
    full_predictions = []
    
    for i, ts in enumerate(timestamps):
        start_idx = i * FORECAST_HORIZON
        end_idx = start_idx + FORECAST_HORIZON
        
        day_dates = test_index[start_idx:end_idx]
        full_dates.extend(day_dates)
        full_actuals.extend(actuals[start_idx:end_idx])
        full_predictions.extend(predictions[start_idx:end_idx])
    
    full_dates = pd.DatetimeIndex(full_dates)
    full_actuals = np.array(full_actuals)
    full_predictions = np.array(full_predictions)
    
    # Graphique principal
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=full_dates,
        y=full_actuals,
        mode='lines',
        name='Consommation Réelle',
        line=dict(color='#2E86AB', width=1.5),
        hovertemplate='<b>%{x|%d/%m/%Y %H:%M}</b><br>Réel: %{y:.4f} kWh<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=full_dates,
        y=full_predictions,
        mode='lines',
        name='Prévision XGBoost',
        line=dict(color='#E63946', width=1.5, dash='dash'),
        hovertemplate='<b>%{x|%d/%m/%Y %H:%M}</b><br>Prédit: %{y:.4f} kWh<extra></extra>'
    ))

    m = results['metrics']
    annotation_text = (
        f"NRMSE (moyenne) = {m['NRMSE']:.4f} | "
        f"NRMSE (IQR) = {m['NRMSE_IQR']:.4f} | "
        f"MAE = {m['MAE']:.4f} kWh | "
        f"MAPE = {m['MAPE']:.2f}% (hors {m['pct_filtered']:.1f}% pas inactifs, "
        f"seuil = {m['mape_threshold']:.4f} kWh)"
    )
    
    fig.update_layout(
        title=dict(
            text=f'Prévisions Interactive - {household} - XGBoost (Toute l\'année 2019)',
            font=dict(size=18, family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        annotations=[dict(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            showarrow=False,
            font=dict(size=12),
            align="center",
        )],
        xaxis=dict(
            title='Date',
            titlefont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1 jour", step="day", stepmode="backward"),
                    dict(count=7, label="1 semaine", step="day", stepmode="backward"),
                    dict(count=1, label="1 mois", step="month", stepmode="backward"),
                    dict(step="all", label="Tout")
                ]),
                x=0.0,
                y=1.15
            ),
            type='date'
        ),
        yaxis=dict(
            title='Consommation (kWh/15min)',
            titlefont=dict(size=14),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=700,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(t=100, b=80)
    )
    
    html_file = output_path / f"{household}_xgboost_interactive.html"
    fig.write_html(str(html_file))
    print(f"  ✓ Graphique: {html_file}")
    
    # Profil journalier
    df_plot = pd.DataFrame({
        'datetime': full_dates,
        'actual': full_actuals,
        'prediction': full_predictions
    })
    
    df_plot['hour'] = df_plot['datetime'].dt.hour
    df_plot['minute'] = df_plot['datetime'].dt.minute
    df_plot['time_of_day'] = df_plot['hour'] + df_plot['minute'] / 60
    
    hourly_actual = df_plot.groupby('time_of_day')['actual'].mean()
    hourly_pred = df_plot.groupby('time_of_day')['prediction'].mean()
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=hourly_actual.index,
        y=hourly_actual.values,
        mode='lines+markers',
        name='Consommation Réelle',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)'
    ))
    
    fig2.add_trace(go.Scatter(
        x=hourly_pred.index,
        y=hourly_pred.values,
        mode='lines+markers',
        name='Prévision XGBoost',
        line=dict(color='#E63946', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig2.add_vrect(x0=6, x1=9, fillcolor="orange", opacity=0.1, annotation_text="Pic Matin")
    fig2.add_vrect(x0=18, x1=22, fillcolor="red", opacity=0.1, annotation_text="Pic Soir")
    
    fig2.update_layout(
        title=f'Profil Journalier Moyen - {household} - XGBoost',
        xaxis=dict(
            title='Heure du Jour',
            tickmode='array',
            tickvals=list(range(0, 25, 2)),
            ticktext=[f'{h:02d}h' for h in range(0, 25, 2)]
        ),
        yaxis=dict(title='Consommation Moyenne (kWh/15min)'),
        template='plotly_white',
        height=600
    )
    
    html_file2 = output_path / f"{household}_xgboost_daily_profile.html"
    fig2.write_html(str(html_file2))
    print(f"  ✓ Profil journalier: {html_file2}")
    
    # CSV résultats
    csv_file = output_path / f"{household}_xgboost_results.csv"
    results_df = pd.DataFrame({
        'Date':            timestamps,
        'NRMSE':           [m['NRMSE']]           * len(timestamps),
        'NRMSE_IQR':       [m['NRMSE_IQR']]       * len(timestamps),
        'RMSE':            [m['RMSE']]             * len(timestamps),
        'MAE':             [m['MAE']]              * len(timestamps),
        'MAPE':            [m['MAPE']]             * len(timestamps),
        'MAPE_threshold':  [m['mape_threshold']]   * len(timestamps),
        'pct_filtered':    [m['pct_filtered']]     * len(timestamps),
    })
    results_df.to_csv(csv_file, index=False)
    print(f"  ✓ Résultats CSV: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='XGBoost avec Visualisations - Année Complète 2019')
    parser.add_argument('--household', type=int, default=5, choices=[1,2,3,4,5,6])
    parser.add_argument('--test-year', type=int, default=2019)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./visualizations')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" XGBOOST - TOUTE L'ANNÉE 2019 ".center(80))
    print("=" * 80)
    
    household_col = RESIDENTIAL_HOUSEHOLDS[args.household]
    household_label = household_col.replace('DE_KN_', '').replace('_grid_import', '')
    
    print(f"\n📍 Foyer: {household_label} (ID: {args.household})")
    
    # Charger données
    data_file = download_data(DATA_URL_15MIN, cache_dir=args.data_dir)
    data = load_and_prepare_data(data_file, household_col)
    
    # Split
    train_df, test_df = split_data(data, test_year=args.test_year)
    
    # Features
    train_df = create_features(train_df)
    
    # Entraîner
    model = train_xgboost(train_df)
    
    # Prévisions
    results = generate_forecasts(model, train_df, test_df)
    
    # Visualisations
    create_interactive_viz(results, household_label, test_df.index, output_dir=args.output_dir)
    
    print("\n" + "=" * 80)
    print(" ✓ TERMINÉ ".center(80))
    print("=" * 80)
    
    print(f"\n💡 Ouvre les fichiers HTML dans ton navigateur:")
    print(f"  {Path(args.output_dir) / f'{household_label}_xgboost_interactive.html'}")


if __name__ == "__main__":
    main()