import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURAZIONE PAGINA (WIDE LAYOUT) ---
st.set_page_config(
    page_title="F1 AI Race Strategist",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNZIONI DI SUPPORTO (CARICAMENTO E GRAFICI) ---
@st.cache_resource
def load_model():
    try:
        # Assicurati che il file .pkl sia nella stessa cartella dell'app
        return joblib.load('f1_champion_predictor_v2.pkl')
    except FileNotFoundError:
        st.error("⚠️ ERRORE CRITICO: File 'f1_champion_predictor_v2.pkl' non trovato.")
        st.stop()

def create_gauge_chart(probability):
    """Crea un tachimetro stiloso per la probabilità di vittoria"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "WIN PROBABILITY", 'font': {'size': 24, 'color': 'white'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#00FF00'}}, # Verde brillante
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00FF00"}, # Barra verde
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 30], 'color': '#555555'},
                {'range': [30, 70], 'color': '#888888'},
                {'range': [70, 100], 'color': '#aaaaaa'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "white", 'family': "Arial"})
    return fig

def create_bump_chart(df_results):
    """Crea il grafico di flusso Griglia -> Predizione"""
    fig = go.Figure()
    team_colors = {
        'McLaren': '#FF8700', 'Red Bull': '#0600EF', 'Ferrari': '#DC0000',
        'Mercedes': '#00D2BE', 'Aston Martin': '#006F62', 'Alpine': '#0090FF',
        'Williams': '#005AFF', 'Haas': '#FFFFFF', 'Racing Bulls': '#1634CC', 'Stake': '#52E252'
    }
    
    x_labels = ['GRID START', 'AI PREDICTION']
    
    for i, row in df_results.head(10).iterrows(): # Mostriamo solo i top 10 per pulizia
        driver = row['Pilota'].split()[-1].upper()
        team = row['Team']
        y_coords = [row['Grid'], i + 1] # i+1 è il Rank Predetto
        color = team_colors.get(team, 'grey')
        
        # Evidenziamo i movimenti importanti
        is_big_mover = abs(row['Grid'] - (i+1)) >= 3
        lw = 4 if is_big_mover or i < 3 else 2
        
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_coords,
            mode='lines+markers+text',
            name=driver,
            text=[f"{driver} (P{row['Grid']})", f"P{i+1}"],
            textposition=["middle left", "middle right"],
            line=dict(color=color, width=lw),
            marker=dict(size=10, color=color),
            hoverinfo='text',
            hovertext=f"{driver} ({team})<br>Grid: P{row['Grid']} -> Pred: P{i+1}"
        ))

    fig.update_layout(
        title=dict(text='<b>RACE FLOW PREDICTION (Top 10)</b>', font=dict(size=18, color='white')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=500,
        yaxis=dict(title='Position', autorange="reversed", tickmode='linear', tick0=1, dtick=1, range=[11, 0.5], showgrid=True, gridcolor='#333333', color='white'),
        xaxis=dict(showgrid=False, color='white')
    )
    return fig

# --- CARICAMENTO ---
model = load_model()

# --- SIDEBAR: CONFIGURAZIONE ---
with st.sidebar:
    st.title("⚙️ RACE CONTROL")
    st.markdown("---")
    st.header("📍 Circuit & Conditions")
    circuit_score = st.slider("Grid Importance Factor (0.3=Spa, 0.9=Monaco)", 0.0, 1.0, 0.4, 0.05, help="Quanto è difficile sorpassare? Valori alti favoriscono la Pole.")
    
    st.markdown("---")
    st.header("🏎️ Team Power Index (Weekend Pace)")
    with st.expander("Adjust Team Performance", expanded=False):
        teams = ['McLaren', 'Red Bull', 'Ferrari', 'Mercedes', 'Aston Martin', 'Alpine', 'Williams', 'Haas', 'Racing Bulls', 'Stake']
        team_power = {}
        for team in teams:
            default_val = 36.0 if team == 'McLaren' else (34.0 if team == 'Red Bull' else (32.0 if team == 'Ferrari' else 15.0))
            team_power[team] = st.slider(f"{team}", 0.0, 45.0, default_val)

# --- MAIN PAGE: DASHBOARD ---
st.title("🏁 F1 AI STRATEGIST: RACE PREDICTOR")
st.markdown("#### *Machine Learning powered analysis based on Hybrid Era data.*")
st.markdown("---")

# --- SEZIONE SUPERIORE: INPUT GRIGLIA ---
col_grid_input, col_top_stats = st.columns([2, 1])

with col_grid_input:
    st.subheader("🚦 Starting Grid Configuration")
    st.info("Edit the grid below. The model recalculates instantly upon clicking Predict.")
    
    # Dataframe di default (modificabile dall'utente)
    default_grid_data = [
        {'Pilota': 'Lando Norris', 'Team': 'McLaren', 'Grid': 1, 'Dominance': 0.7},
        {'Pilota': 'Max Verstappen', 'Team': 'Red Bull', 'Grid': 2, 'Dominance': 0.95},
        {'Pilota': 'Charles Leclerc', 'Team': 'Ferrari', 'Grid': 3, 'Dominance': 0.6},
        {'Pilota': 'Oscar Piastri', 'Team': 'McLaren', 'Grid': 4, 'Dominance': 0.3},
        {'Pilota': 'George Russell', 'Team': 'Mercedes', 'Grid': 5, 'Dominance': 0.8},
        {'Pilota': 'Lewis Hamilton', 'Team': 'Ferrari', 'Grid': 6, 'Dominance': 0.4},
        {'Pilota': 'Fernando Alonso', 'Team': 'Aston Martin', 'Grid': 7, 'Dominance': 0.9},
        {'Pilota': 'Alex Albon', 'Team': 'Williams', 'Grid': 8, 'Dominance': 0.8},
        {'Pilota': 'Yuki Tsunoda', 'Team': 'Racing Bulls', 'Grid': 9, 'Dominance': 0.5},
        {'Pilota': 'Nico Hulkenberg', 'Team': 'Haas', 'Grid': 10, 'Dominance': 0.7},
        # Aggiungi altri se vuoi...
    ]
    df_grid_input = pd.DataFrame(default_grid_data)
    
    # L'editor che permette all'utente di cambiare i dati
    edited_grid_df = st.data_editor(
        df_grid_input, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Grid": st.column_config.NumberColumn(min_value=1, max_value=20, format="%d°"),
            "Dominance": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f", help="Driver Skill vs Teammate (0.9=Max, 0.1=Stroll)"),
        },
        hide_index=True
    )
    
    predict_button = st.button("🚀 RUN AI PREDICTION MODEL", type="primary", use_container_width=True)

# --- LOGICA DI PREDIZIONE (Succede solo se premi il bottone) ---
if predict_button:
    with st.spinner("AI Model is crunching the numbers..."):
        input_rows = []
        # Iteriamo sul dataframe MODIFICATO dall'utente
        for index, row in edited_grid_df.iterrows():
            # Pulizia base del nome pilota per ref (es. "Lando Norris" -> "norris")
            driver_parts = row['Pilota'].split()
            ref_name = driver_parts[-1].lower() if len(driver_parts) > 0 else "unknown"
            if ref_name == 'verstappen': ref_name = 'max_verstappen'
            
            # Costruzione della riga di input per il modello
            # IMPORTANTE: L'ordine e i nomi delle feature devono essere IDENTICI al training
            input_rows.append({
                'pre_race_performance': row['Grid'],
                'grid': row['Grid'],
                'team_recent_form': team_power.get(row['Team'], 5.0), # Prende la potenza dalla sidebar
                'teammate_dominance': row['Dominance'],
                'driver_circuit_history': 12.0, # Default neutro
                'prev_3_pos_avg': row['Grid'], # Default basato sulla griglia
                'driver_age': 26.0, # Default età media
                'team_reliability_bad': 0.1,
                'is_sprint_weekend': 0,
                'grid_importance': circuit_score # Dalla sidebar
            })
            
        df_pred_ready = pd.DataFrame(input_rows)
        
        # Features richieste dal modello (ORDINE TASSATIVO)
        features_list = ['pre_race_performance', 'grid', 'team_recent_form', 'teammate_dominance', 
                         'driver_circuit_history', 'prev_3_pos_avg', 'driver_age', 
                         'team_reliability_bad', 'is_sprint_weekend', 'grid_importance']
        
        # Esecuzione Predizione
        try:
            probs = model.predict_proba(df_pred_ready[features_list])[:, 1]
            edited_grid_df['Win Probability'] = (probs / probs.sum()) * 100
        except Exception as e:
            st.error(f"Errore durante la predizione: {e}")
            st.stop()
        
        # Ordinamento risultati
        results_df = edited_grid_df.sort_values('Win Probability', ascending=False).reset_index(drop=True)
        winner_row = results_df.iloc[0]
        
        # --- VISUALIZZAZIONE RISULTATI (Layout Dashboard) ---
        
        # 1. TOP METRICS (Nella colonna di destra in alto)
        with col_top_stats:
            st.subheader("🏆 Predicted Winner")
            # Usiamo delle "Metriche" native di Streamlit per un look pulito
            st.metric(label="WINNER", value=winner_row['Pilota'], delta=f"P{winner_row['Grid']} to P1")
            st.metric(label="WIN PROBABILITY", value=f"{winner_row['Win Probability']:.1f}%")
            st.metric(label="WINNING TEAM", value=winner_row['Team'])

        st.markdown("---") # Separatore

        # 2. AREA GRAFICI PRINCIPALE (Due colonne)
        col_gauge, col_bump = st.columns([2, 3])
        
        with col_gauge:
            # Grafico Tachimetro (Plotly)
            st.plotly_chart(create_gauge_chart(winner_row['Win Probability']), use_container_width=True)
            
            # Tabella riassuntiva sotto il tachimetro
            st.subheader("📊 Podium Probabilities")
            st.dataframe(
                results_df[['Pilota', 'Team', 'Win Probability']].head(5).style.format({'Win Probability': '{:.1f}%'}).background_gradient(cmap='Greens', subset=['Win Probability']),
                use_container_width=True,
                hide_index=True
            )

        with col_bump:
            # Grafico di Flusso (Plotly Bump Chart)
            st.plotly_chart(create_bump_chart(results_df), use_container_width=True)

elif predict_button is False:
    # Messaggio iniziale se non hai ancora premuto il bottone
    with col_top_stats:
         st.info("👈 Configure the grid and click 'RUN AI PREDICTION' to see the results here.")