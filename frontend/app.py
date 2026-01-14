import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
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
# CHARGEMENT DES MODÈLES & FEATURES
# ===================================
try:
    model_regression = joblib.load('../science/models/dpe_Regression_xgb.joblib')
    model_classification = joblib.load('../science/models/dpe_classification_xgb.joblib')
    
    with open('../science/models/feature_columns.json', 'r') as f:
        feature_columns_data = json.load(f)
    
    numerical = feature_columns_data.get('numerical_features', [])
    categorical = feature_columns_data.get('categorical_features', [])
    all_features = numerical + categorical

    # ── Pour la régression : on enlève uniquement la cible ─────────────────────
    forbidden_regression = {'conso_finale'}

    # ── Pour la classification : on enlève la cible DPE (et GES si non utilisé) ──
    forbidden_classification = {'etiquette_dpe', 'etiquette_ges'}

    features_regression = [f for f in all_features if f not in forbidden_regression]
    features_classification = [f for f in all_features if f not in forbidden_classification]

except FileNotFoundError as e:
    st.error(f"Erreur de chargement des modèles ou features : {e}")
    st.stop()

# ===================================
# PRÉPARATION DES FEATURES
# ===================================
def prepare_features(inputs_dict, model_type="regression", predicted_conso=None):
    enriched = inputs_dict.copy()

    # Valeurs par défaut / dérivations
    if 'surface_habitable_logement' not in enriched:
        enriched['surface_habitable_logement'] = enriched.get('surface', 75)
    if 'nombre_niveau_logement' not in enriched:
        enriched['nombre_niveau_logement'] = 1 if enriched.get('type_logement') == 'Appartement' else 2
    if 'hauteur_sous_plafond' not in enriched:
        enriched['hauteur_sous_plafond'] = 2.5
    if 'type_energie_principale' not in enriched:
        type_ch = enriched.get('type_chauffage', 'Electrique')
        mapping = {
            'Electrique': 'Électricité',
            'Gaz': 'Gaz naturel',
            'Fioul': 'Fioul domestique',
            'Individuel': 'Gaz naturel',
            'Collectif': 'Gaz naturel'
        }
        enriched['type_energie_principale'] = mapping.get(type_ch, 'Électricité')
    if 'isolation' not in enriched:
        annee = enriched.get('annee_construction', 2000)
        iso_murs = enriched.get('isolation_murs', 0)
        if annee >= 2012 or iso_murs == 1:
            enriched['isolation'] = 'Bonne'
        elif annee >= 1990:
            enriched['isolation'] = 'Moyenne'
        else:
            enriched['isolation'] = 'Faible'
    if 'type_batiment' not in enriched:
        enriched['type_batiment'] = "immeuble collectif d'habitation" if enriched.get('type_logement') == 'Appartement' else 'maison'
    if 'type_installation_chauffage' not in enriched:
        enriched['type_installation_chauffage'] = 'chauffage collectif' if enriched.get('type_chauffage') == 'Collectif' else 'chauffage individuel'
    if 'type_installation_ecs' not in enriched:
        enriched['type_installation_ecs'] = 'production ECS collective' if enriched.get('type_chauffage') == 'Collectif' else 'production ECS individuelle'
    if 'code_postal' not in enriched:
        enriched['code_postal'] = '75000'

    # Création du DataFrame de base
    df_final = pd.DataFrame(index=[0])

    if model_type == "regression":
        use_cols = features_regression
    else:
        use_cols = features_classification

    for col in use_cols:
        if col in enriched:
            df_final[col] = enriched[col]
        else:
            # Valeur par défaut : 0 pour la plupart, mais attention au pré-processing d'entraînement
            df_final[col] = 0

    # Injection de la conso prédite pour le modèle de classification
    if model_type == "classification" and predicted_conso is not None:
        df_final['conso_finale'] = predicted_conso

    # Réordonner les colonnes exactement comme lors de l'entraînement (important pour XGBoost)
    if model_type == "regression":
        df_final = df_final[features_regression]
    else:
        df_final = df_final[features_classification]

    return df_final

# ===================================
# PIPELINE DE PRÉDICTION
# ===================================
def predict_conso_and_dpe(inputs_dict):
    # 1. Prédiction conso (régression)
    X_reg = prepare_features(inputs_dict, model_type="regression")
    conso_kwh_m2_an = float(model_regression.predict(X_reg)[0])

    # 2. Prédiction classe DPE (classification) avec conso injectée
    X_clf = prepare_features(inputs_dict, model_type="classification", predicted_conso=conso_kwh_m2_an)
    dpe_idx = model_classification.predict(X_clf)[0]

    dpe_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    dpe_class = dpe_classes[int(dpe_idx)] if 0 <= int(dpe_idx) < len(dpe_classes) else 'G'

    return conso_kwh_m2_an, dpe_class

# ===================================
# VISUALISATION DPE (inchangé)
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
        arrow = FancyBboxPatch((0.5, y - 0.4), length, 0.8, boxstyle="round,pad=0.05",
                               facecolor=info['color'], edgecolor='black', linewidth=2 if c == classe_dpe else 1)
        ax.add_patch(arrow)
        ax.text(0.8, y, c, fontsize=18, fontweight='bold', color='white', va='center')
        ax.text(length * 0.6, y, info['label'], fontsize=10, va='center',
                color='white' if c in ['F', 'G'] else 'black', fontweight='bold')
        
        if c == classe_dpe:
            indicator = FancyBboxPatch((7, y - 0.35), 2.5, 0.7, boxstyle="round,pad=0.05",
                                       facecolor=info['color'], edgecolor=info['color'])
            ax.add_patch(indicator)
            ax.text(8.25, y, c, fontsize=16, fontweight='bold', color='white', va='center', ha='center')
    
    ax.text(5, -1.2, f"Consommation : {consommation_kwh_m2_an:.1f} kWh/m²/an", ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax.text(5, -1.8, f"Classe énergétique : {classe_dpe}", ha='center', fontsize=14, fontweight='bold',
            color=dpe.classes[classe_dpe]['color'])
    return fig

# ===================================
# CONSOMMATION MENSUELLE (inchangé)
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
    
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mois, valeurs, marker='o', linewidth=2)
    ax.set_title(f"{title} en {unite}")
    ax.set_xlabel("Mois")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig

# ===================================
# APPLICATION DES TRAVAUX (simplifié)
# ===================================
def apply_renovation(inputs_dict, renovation_type):
    inputs_apres = inputs_dict.copy()
    if "Isolation des murs" in renovation_type:
        inputs_apres['isolation_murs'] = 1
    if "Changement de chauffage" in renovation_type:
        inputs_apres['type_chauffage'] = 'Electrique'
    # → Ajoute ici d'autres effets si tu as implémenté plus de travaux
    return inputs_apres

# ===================================
# INTERFACE STREAMLIT
# ===================================
st.set_page_config(page_title="Simulateur DPE avec ML", layout="centered")
st.title("🏠 Simulateur de DPE et Rénovation Énergétique")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Logement actuel")
surface = st.sidebar.number_input("Surface (m²)", 10, 500, 75)
annee_construction = st.sidebar.number_input("Année construction", 1800, 2025, 2000)
type_logement = st.sidebar.selectbox("Type logement", ["Maison", "Appartement"])
type_chauffage = st.sidebar.selectbox("Type chauffage", ["Individuel", "Collectif", "Electrique", "Gaz", "Fioul"])
code_postal = st.sidebar.text_input("Code postal", "75000")
isolation_murs = st.sidebar.selectbox("Isolation murs", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

try:
    zone = 'H1' if int(code_postal[:2]) < 50 else 'H2'
except:
    zone = 'H1'

st.sidebar.header("Travaux envisagés")
renovation_type = st.sidebar.multiselect("Travaux", [
    "Isolation des murs",
    "Changement de chauffage",
    "Fenêtres double vitrage",
    "Panneaux solaires"
])

# ── Lancer la simulation ───────────────────────────────────────────────────
if st.button("🔍 Lancer la simulation"):
    inputs_actuel = {
        'surface': surface,
        'surface_habitable_logement': surface,  # souvent requis
        'annee_construction': annee_construction,
        'type_logement': type_logement,
        'type_chauffage': type_chauffage,
        'zone_climatique': zone,
        'isolation_murs': isolation_murs,
        'code_postal': code_postal
    }

    try:
        st.session_state.conso_avant, st.session_state.dpe_avant = predict_conso_and_dpe(inputs_actuel)
        
        if renovation_type:
            inputs_apres = apply_renovation(inputs_actuel, renovation_type)
            st.session_state.conso_apres, st.session_state.dpe_apres = predict_conso_and_dpe(inputs_apres)
        else:
            st.session_state.conso_apres = None
            st.session_state.dpe_apres = None
            
        st.session_state.calcul_lance = True
    except Exception as e:
        st.error(f"Erreur prédiction : {str(e)}")
        st.session_state.calcul_lance = False

# ── Affichage des résultats ────────────────────────────────────────────────
if st.session_state.calcul_lance and st.session_state.conso_avant is not None:
    unite = st.selectbox("Unité consommation", ["kWh", "€"])
    st.caption("Tarif : 0,2516 €/kWh")

    st.subheader("État actuel")
    fig_avant = visualiser_dpe(st.session_state.conso_avant, st.session_state.dpe_avant, "DPE Actuel")
    st.pyplot(fig_avant)
    fig_conso_avant = visualiser_conso_mensuelle(st.session_state.conso_avant, surface, unite, "Actuel")
    st.pyplot(fig_conso_avant)

    if st.session_state.conso_apres is not None:
        st.subheader("Après travaux")
        fig_apres = visualiser_dpe(st.session_state.conso_apres, st.session_state.dpe_apres, "DPE Après travaux")
        st.pyplot(fig_apres)
        fig_conso_apres = visualiser_conso_mensuelle(st.session_state.conso_apres, surface, unite, "Après travaux")
        st.pyplot(fig_conso_apres)

        st.subheader("Gains estimés")
        delta_conso = st.session_state.conso_avant - st.session_state.conso_apres
        delta_pct = (delta_conso / st.session_state.conso_avant) * 100 if st.session_state.conso_avant != 0 else 0
        col1, col2 = st.columns(2)
        col1.metric("Réduction conso", f"{delta_conso:.1f} kWh/m²/an", f"-{delta_pct:.1f}%")
        col2.metric("Évolution DPE", st.session_state.dpe_apres, f"{st.session_state.dpe_avant} → {st.session_state.dpe_apres}")
    else:
        st.info("Sélectionnez des travaux pour voir l'impact.")