import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import requests
import zipfile
import io
import os
import shutil

print("🏎️ INIZIO PROCESSO DI RIPARAZIONE MODELLO LOCALE (V4 - ZIP METHOD)...")

# --- 1. SCARICAMENTO ROBUSTO (ZIP) ---
# Usiamo il download diretto dello ZIP che è molto più stabile dei singoli CSV raw
ZIP_URL_PRIMARY = "http://ergast.com/downloads/f1db_csv.zip"  # Fonte Ufficiale
ZIP_URL_BACKUP = "https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip" # Fonte Backup (F1DB Project)
TEMP_DIR = "f1_data_temp"

def download_and_extract():
    # Pulizia cartella temporanea se esiste
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print("📥 Tentativo download Database F1 (Fonte Primaria)...")
    try:
        r = requests.get(ZIP_URL_PRIMARY, timeout=15)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(TEMP_DIR)
            print("✅ Download Ergast completato.")
            return True
    except Exception as e:
        print(f"⚠️ Fonte primaria fallita: {e}")

    print("📥 Tentativo download Database F1 (Fonte Backup)...")
    try:
        r = requests.get(ZIP_URL_BACKUP, timeout=30)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(TEMP_DIR)
            print("✅ Download F1DB completato.")
            return True
    except Exception as e:
        print(f"❌ ERRORE CRITICO: Impossibile scaricare il database da nessuna fonte: {e}")
        return False

if not download_and_extract():
    exit()

# --- 2. CARICAMENTO E NORMALIZZAZIONE COLONNE ---
# Questa funzione risolve il problema dei nomi diversi (raceId vs race_id)
def load_clean_csv(filename):
    path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(path):
        # A volte i backup usano nomi leggermente diversi, cerchiamo file simili
        files = os.listdir(TEMP_DIR)
        candidates = [f for f in files if filename.replace('.csv','') in f]
        if candidates:
            path = os.path.join(TEMP_DIR, candidates[0])
        else:
            print(f"⚠️ File {filename} non trovato. Salto.")
            return None
            
    df = pd.read_csv(path)
    
    # NORMALIZZAZIONE NOMI COLONNE (CAMELCASE)
    # Trasforma race_id -> raceId, driver_id -> driverId
    new_cols = []
    for c in df.columns:
        if '_id' in c:
            new_c = c.replace('_id', 'Id')
        elif c == 'year': # Standardizziamo season -> year
            new_c = 'year'
        else:
            new_c = c
        new_cols.append(new_c)
    df.columns = new_cols
    
    # Fix specifici
    rename_map = {
        'season': 'year',
        'date': 'race_date',
        'name': 'race_name' if 'race' in filename else 'name', # race_name solo in races.csv
        'grid': 'grid'
    }
    # Applica rename solo se la colonna esiste
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    # Fix specifico per races.csv (se ha colonna 'name' la chiamiamo 'race_name')
    if 'race' in filename and 'name' in df.columns and 'race_name' not in df.columns:
         df = df.rename(columns={'name': 'race_name'})

    return df

print("⚙️ Caricamento file CSV...")
races = load_clean_csv('races.csv')
results = load_clean_csv('results.csv')
drivers = load_clean_csv('drivers.csv')
constructors = load_clean_csv('constructors.csv')
status = load_clean_csv('status.csv')
circuits = load_clean_csv('circuits.csv')
sprint_results = load_clean_csv('sprint_results.csv')

# Fix driver columns
if drivers is not None:
    drivers = drivers.rename(columns={'url': 'driver_url', 'nationality': 'nationality'})
if constructors is not None:
    constructors = constructors.rename(columns={'name': 'team_name'})

# --- 3. FEATURE ENGINEERING ---
print("⚙️ Elaborazione Dati...")

# Merge
df = results.merge(races[['raceId', 'year', 'race_name', 'race_date', 'circuitId']], on='raceId', how='left')
df = df.merge(drivers[['driverId', 'driverRef', 'dob', 'nationality']], on='driverId', how='left')
df = df.merge(constructors[['constructorId', 'team_name']], on='constructorId', how='left')
df = df.merge(status[['statusId', 'status']], on='statusId', how='left')

# Filtro
df = df[df['year'] >= 2014].copy()
df['race_date'] = pd.to_datetime(df['race_date'])
df = df.sort_values(['race_date', 'driverId'])

# Sprint
if sprint_results is not None:
    # Normalizza nome colonna posizione sprint
    sprint_col = 'positionOrder' if 'positionOrder' in sprint_results.columns else 'position'
    sprint_subset = sprint_results[['raceId', 'driverId', sprint_col]].rename(columns={sprint_col: 'sprint_pos'})
    df = df.merge(sprint_subset, on=['raceId', 'driverId'], how='left')
    df['pre_race_performance'] = df['sprint_pos'].fillna(df['grid'])
    df['is_sprint_weekend'] = df['sprint_pos'].notna().astype(int)
else:
    df['pre_race_performance'] = df['grid']
    df['is_sprint_weekend'] = 0

# Grid Importance
finished = df[df['statusId'] == 1]
grid_corr = finished.groupby('circuitId')[['grid', 'positionOrder']].corr().iloc[0::2, -1].reset_index()
grid_corr = grid_corr.drop(columns=['level_1']).rename(columns={'positionOrder': 'grid_importance'})
df = df.merge(grid_corr, on='circuitId', how='left')
df['grid_importance'] = df['grid_importance'].fillna(0.5)

# Teammate Dominance
tm_comp = df[['raceId', 'constructorId', 'driverId', 'positionOrder']].merge(
    df[['raceId', 'constructorId', 'driverId', 'positionOrder']], 
    on=['raceId', 'constructorId'], suffixes=('', '_tm')
)
tm_comp = tm_comp[tm_comp['driverId'] != tm_comp['driverId_tm']]
tm_comp['beat_tm'] = (tm_comp['positionOrder'] < tm_comp['positionOrder_tm']).astype(int)
tm_stats = tm_comp.sort_values('raceId')
tm_stats['teammate_dominance'] = tm_stats.groupby('driverId')['beat_tm'].transform(lambda x: x.shift(1).rolling(5).mean())
tm_feats = tm_stats[['raceId', 'driverId', 'teammate_dominance']].drop_duplicates(subset=['raceId', 'driverId'])
df = df.merge(tm_feats, on=['raceId', 'driverId'], how='left')
df['teammate_dominance'] = df['teammate_dominance'].fillna(0.5)

# Altre Feature
df['driver_age'] = (df['race_date'] - pd.to_datetime(df['dob'])).dt.days / 365.25
df['prev_3_pos_avg'] = df.groupby('driverId')['positionOrder'].transform(lambda x: x.shift(1).rolling(3).mean())
df['driver_circuit_history'] = df.groupby(['driverId', 'circuitId'])['positionOrder'].transform(lambda x: x.expanding().mean().shift(1)).fillna(12)

team_pts = df.groupby(['raceId', 'constructorId'])['points'].sum().reset_index()
df = df.merge(team_pts, on=['raceId', 'constructorId'], suffixes=('', '_tm_tot'))
df['team_recent_form'] = df.groupby('constructorId')['points_tm_tot'].transform(lambda x: x.shift(1).rolling(5).mean())

mech_ids = [5, 6, 7, 8, 9, 10, 21, 22, 23]
df['is_mech_fail'] = df['statusId'].isin(mech_ids).astype(int)
df['team_reliability_bad'] = df.groupby('constructorId')['is_mech_fail'].transform(lambda x: x.shift(1).rolling(5).mean()).fillna(0)

# Training Data
df_final = df.dropna(subset=['prev_3_pos_avg', 'team_recent_form']).copy()
df_final['is_winner'] = (df_final['positionOrder'] == 1).astype(int)

features = [
    'pre_race_performance', 'grid', 'team_recent_form', 'teammate_dominance',
    'driver_circuit_history', 'prev_3_pos_avg', 'driver_age', 
    'team_reliability_bad', 'is_sprint_weekend', 'grid_importance'
]

# --- 4. ADDESTRAMENTO E SALVATAGGIO ---
print(f"🧠 Addestramento su {len(df_final)} righe...")
model_god = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, subsample=0.8, random_state=42)
model_god.fit(df_final[features], df_final['is_winner'])

print("💾 Salvataggio modello...")
joblib.dump(model_god, 'f1_champion_predictor_v2.pkl')

# Pulizia temp
shutil.rmtree(TEMP_DIR)
print("\n✅ MODELLO RIPARATO CON SUCCESSO!")
print("Ora puoi lanciare: streamlit run f1_app.py")
