import streamlit as st
import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ===================================
# ÉTAT DE L'APPLICATION
# ===================================
if "calcul_lance" not in st.session_state:
    st.session_state.calcul_lance = False
if "conso_avant" not in st.session_state:
    st.session_state.conso_avant = None
if "dpe_avant" not in st.session_state:
    st.session_state.dpe_avant = None
if "conso_apres" not in st.session_state:
    st.session_state.conso_apres = None
if "dpe_apres" not in st.session_state:
    st.session_state.dpe_apres = None

# ===================================
# CLASSE DPE VISUALIZER
# ===================================
class DPEVisualizer:
    def __init__(self):
        self.classes = {
            'A': {'min': 0, 'max': 50, 'color': '#319834', 'label': '≤ 50'},
            'B': {'min': 51, 'max': 90, 'color': '#35B44A', 'label': '51 à 90'},
            'C': {'min': 91, 'max': 150, 'color': '#C7D301', 'label': '91 à 150'},
            'D': {'min': 151, 'max': 230, 'color': '#FFED00', 'label': '151 à 230'},
            'E': {'min': 231, 'max': 330, 'color': '#FCAF17', 'label': '231 à 330'},
            'F': {'min': 331, 'max': 450, 'color': '#EF7D08', 'label': '331 à 450'},
            'G': {'min': 451, 'max': 999, 'color': '#E2001A', 'label': '> 450'}
        }
        self.order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    def get_class_from_conso(self, conso_kwh_m2_an):
        for c in self.order:
            if self.classes[c]['min'] <= conso_kwh_m2_an <= self.classes[c]['max']:
                return c
        return 'G'

# ===================================
# CHARGEMENT DES MODÈLES
# ===================================
try:
    model_classification = joblib.load('../science/models/dpe_classification_xgb.joblib')
    # Assumer que le modèle de régression est sauvegardé de manière similaire
    # model_regression = joblib.load('science/models/conso_regression_xgb.joblib')
    with open('../science/models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
except FileNotFoundError as e:
    st.error(f"Erreur de chargement : {e}. Vérifiez les chemins des modèles.")
    st.stop()

# ===================================
# VISUALISATION DPE
# ===================================
def visualiser_dpe(consommation_kwh_m2_an, classe_dpe, title="DPE"):
    dpe = DPEVisualizer()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(dpe.order) + 2)
    ax.axis("off")
    ax.text(0.5, 1.05, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=16, fontweight='bold')
    ax.text(0.01, 0.85, "Logement économe", transform=ax.transAxes, fontsize=11, style='italic')
    ax.text(0.01, 0.01, "Logement énergivore", transform=ax.transAxes, fontsize=11, style='italic')
    
    y_start = len(dpe.order)
    for i, c in enumerate(dpe.order):
        y = y_start - i
        info = dpe.classes[c]
        length = 3.5 + i * 0.3
        arrow = FancyBboxPatch(
            (0.5, y - 0.4), length, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=info['color'],
            edgecolor='black',
            linewidth=2 if c == classe_dpe else 1
        )
        ax.add_patch(arrow)
        ax.text(0.8, y, c, fontsize=18, fontweight='bold', color='white', va='center')
        ax.text(length * 0.6, y, info['label'], fontsize=10, va='center', color='white' if c in ['F', 'G'] else 'black', fontweight='bold')
        
        if c == classe_dpe:
            indicator = FancyBboxPatch(
                (7, y - 0.35), 2.5, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=info['color'],
                edgecolor=info['color']
            )
            ax.add_patch(indicator)
            ax.text(8.25, y, c, fontsize=16, fontweight='bold', color='white', va='center', ha='center')
    
    ax.text(5, -1.2, f"Consommation : {consommation_kwh_m2_an:.1f} kWh/m²/an", ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax.text(5, -1.8, f"Classe énergétique : {classe_dpe}", ha='center', fontsize=14, fontweight='bold', color=dpe.classes[classe_dpe]['color'])
    return fig

# ===================================
# PRÉPARATION DES FEATURES
# ===================================
def prepare_features(inputs_dict):
    df = pd.DataFrame([inputs_dict])
    # Appliquez les transformations identiques à celles du notebook
    # Exemple : encodage one-hot pour variables catégorielles
    categoricals = ['type_logement', 'type_chauffage', 'zone_climatique']  # Ajustez selon feature_columns
    df = pd.get_dummies(df, columns=categoricals)
    # Ajoutez colonnes manquantes à 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    # Réordonnez les colonnes
    df = df[feature_columns]
    return df

# ===================================
# PIPELINE DE PRÉDICTION
# ===================================
def predict_conso_and_dpe(features_df):
    # Prédire la consommation (régression)
    #conso_kwh_m2_an = model_regression.predict(features_df)[0]
    # Prédire la classe DPE (classification)
    dpe_pred = model_classification.predict(features_df)[0]
    # return conso_kwh_m2_an, dpe_pred
    return dpe_pred

# ===================================
# SIMULATION RÉNOVATION
# ===================================
def apply_renovation(inputs_dict, renovation_type):
    inputs_apres = inputs_dict.copy()
    # Modifiez les features en fonction des travaux
    if "Isolation des murs" in renovation_type:
        inputs_apres['isolation_murs'] = 1  # Exemple, assumez feature existe
    if "Changement de chauffage" in renovation_type:
        inputs_apres['type_chauffage'] = 'Efficace'  # Ajustez
    if "Fenêtres double vitrage" in renovation_type:
        inputs_apres['fenetres'] = 'Double' 
    if "Panneaux solaires" in renovation_type:
        inputs_apres['energie_renouvelable'] = 1
    return inputs_apres

# ===================================
# VISUALISATION CONSOMMATION MENSUELLE
# ===================================
def visualiser_conso_mensuelle(conso_kwh_m2_an, surface, unite="kWh", title="Consommation mensuelle"):
    PRIX_KWH = 0.2516
    conso_annuelle = conso_kwh_m2_an * surface
    conso_mensuelle_moy = conso_annuelle / 12
    mois = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    coeffs = [1.4, 1.3, 1.2, 0.9, 0.7, 0.6, 0.5, 0.5, 0.7, 1.0, 1.3, 1.6]
    conso_kwh = [conso_mensuelle_moy * c for c in coeffs]
    if unite == "€":
        valeurs = [c * PRIX_KWH for c in conso_kwh]
        ylabel = "Coût (€)"
    else:
        valeurs = conso_kwh
        ylabel = "Consommation (kWh)"
    fig_conso, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mois, valeurs, marker='o', linewidth=2)
    ax.set_title(f"{title} en {unite}")
    ax.set_xlabel("Mois")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig_conso

# ===================================
# STREAMLIT CONFIG
# ===================================
st.set_page_config(page_title="Simulateur DPE avec ML", layout="centered")
st.title("🏠 Simulateur de DPE et Rénovation Énergétique")

# ===================================
# SIDEBAR – INPUTS LOGEMENT ACTUEL
# ===================================
st.sidebar.header("Informations sur le logement actuel")
surface = st.sidebar.number_input("Surface du logement (m²)", min_value=10, max_value=500, value=75)
annee_construction = st.sidebar.number_input("Année de construction", min_value=1800, max_value=2025, value=2000)
type_logement = st.sidebar.selectbox("Type de logement", ["Maison", "Appartement"])
type_chauffage = st.sidebar.selectbox("Type de chauffage", ["Individuel", "Collectif", "Electrique", "Gaz", "Fioul"])
code_postal = st.sidebar.text_input("Code postal", "75000")
# Ajoutez plus d'inputs selon les features : ex isolation, etc.
isolation_murs = st.sidebar.selectbox("Isolation des murs", [0, 1])  # 0: non, 1: oui
# ... ajoutez d'autres

# Dériver zone climatique de code_postal (simplifié)
zone_climatique = 'H1' if int(code_postal[:2]) < 50 else 'H2'  # Exemple grossier

# ===================================
# SIDEBAR – OPTIONS RÉNOVATION
# ===================================
st.sidebar.header("Scénarios de rénovation")
renovation_type = st.sidebar.multiselect("Travaux envisagés", ["Isolation des murs", "Changement de chauffage", "Fenêtres double vitrage", "Panneaux solaires"])

# ===================================
# 🔘 BOUTON SIMULATION
# ===================================
if st.button("🔍 Lancer la simulation"):
    st.session_state.calcul_lance = True
    
    # Inputs actuels
    inputs_actuel = {
        'surface': surface,
        'annee_construction': annee_construction,
        'type_logement': type_logement,
        'type_chauffage': type_chauffage,
        'zone_climatique': zone_climatique,
        'isolation_murs': isolation_murs
    }
    features_actuel = prepare_features(inputs_actuel)
    
    # Prédiction actuelle
    # st.session_state.conso_avant, 
    dpe_pred_avant = predict_conso_and_dpe(features_actuel)
    st.session_state.dpe_avant = dpe_pred_avant  # Ou use get_class_from_conso si mismatch
    
    # Simulation rénovation
    if renovation_type:
        inputs_apres = apply_renovation(inputs_actuel, renovation_type)
        features_apres = prepare_features(inputs_apres)
        st.session_state.conso_apres, dpe_pred_apres = predict_conso_and_dpe(features_apres)
        st.session_state.dpe_apres = dpe_pred_apres
    else:
        st.session_state.conso_apres = None
        st.session_state.dpe_apres = None

# ===================================
# RÉSULTATS
# ===================================
if st.session_state.calcul_lance:
    unite = st.selectbox("Afficher la consommation en :", ["kWh", "€"])
    st.caption("Tarif standard: 0,2516 €/kWh")
    
    st.subheader("État actuel")
    fig_avant = visualiser_dpe(st.session_state.conso_avant, st.session_state.dpe_avant, title="DPE Actuel")
    st.pyplot(fig_avant)
    fig_conso_avant = visualiser_conso_mensuelle(st.session_state.conso_avant, surface, unite, "Consommation mensuelle actuelle")
    st.pyplot(fig_conso_avant)
    
    if st.session_state.conso_apres is not None:
        st.subheader("Après rénovation")
        fig_apres = visualiser_dpe(st.session_state.conso_apres, st.session_state.dpe_apres, title="DPE Après Rénovation")
        st.pyplot(fig_apres)
        fig_conso_apres = visualiser_conso_mensuelle(st.session_state.conso_apres, surface, unite, "Consommation mensuelle après rénovation")
        st.pyplot(fig_conso_apres)
    else:
        st.info("Sélectionnez des travaux de rénovation pour simuler l'impact.")