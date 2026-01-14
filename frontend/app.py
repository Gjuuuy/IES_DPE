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
# CHARGEMENT DES MODÈLES
# ===================================
try:
    # Charger les modèles
    model_data_classification = joblib.load('../science/models/dpe_classification_xgb.joblib')
    model_data_regression = joblib.load('../science/models/dpe_Regression_xgb.joblib')
    
    # Gérer le cas où joblib charge un dict ou directement le modèle
    if isinstance(model_data_classification, dict):
        model_classification = model_data_classification.get('model', model_data_classification)
    else:
        model_classification = model_data_classification
    
    if isinstance(model_data_regression, dict):
        model_regression = model_data_regression.get('model', model_data_regression)
    else:
        model_regression = model_data_regression
    
    with open('../science/models/feature_columns.json', 'r') as f:
        feature_columns_data = json.load(f)
    
    # Extraire les colonnes du format JSON structuré
    if isinstance(feature_columns_data, dict):
        if 'numerical_features' in feature_columns_data and 'categorical_features' in feature_columns_data:
            # Combiner les features numériques et catégorielles
            feature_columns = (feature_columns_data['numerical_features'] + 
                             feature_columns_data['categorical_features'])
        else:
            feature_columns = list(feature_columns_data.keys())
    else:
        feature_columns = feature_columns_data
        
except FileNotFoundError as e:
    st.error(f"Erreur de chargement : {e}. Vérifiez les chemins des modèles.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")
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
    # Enrichir inputs_dict avec des valeurs par défaut pour les colonnes complexes
    enriched_inputs = inputs_dict.copy()
    
    # Colonnes dérivées automatiquement
    if 'surface_habitable_logement' not in enriched_inputs:
        enriched_inputs['surface_habitable_logement'] = enriched_inputs.get('surface', 75)
    
    if 'nombre_niveau_logement' not in enriched_inputs:
        # Estimation: appartement = 1 niveau, maison = 2 niveaux en moyenne
        enriched_inputs['nombre_niveau_logement'] = 1 if enriched_inputs.get('type_logement') == 'Appartement' else 2
    
    if 'hauteur_sous_plafond' not in enriched_inputs:
        # Valeur standard en France
        enriched_inputs['hauteur_sous_plafond'] = 2.5
    
    if 'type_energie_principale' not in enriched_inputs:
        # Dériver du type de chauffage
        type_chauffage = enriched_inputs.get('type_chauffage', 'Electrique')
        mapping_energie = {
            'Electrique': 'Électricité',
            'Gaz': 'Gaz naturel',
            'Fioul': 'Fioul domestique',
            'Individuel': 'Gaz naturel',
            'Collectif': 'Gaz naturel'
        }
        enriched_inputs['type_energie_principale'] = mapping_energie.get(type_chauffage, 'Électricité')
    
    if 'isolation' not in enriched_inputs:
        # Estimation basée sur l'année de construction et l'isolation des murs
        annee = enriched_inputs.get('annee_construction', 2000)
        isolation_murs = enriched_inputs.get('isolation_murs', 0)
        if annee >= 2012 or isolation_murs == 1:
            enriched_inputs['isolation'] = 'Bonne'
        elif annee >= 1990:
            enriched_inputs['isolation'] = 'Moyenne'
        else:
            enriched_inputs['isolation'] = 'Faible'
    
    if 'type_batiment' not in enriched_inputs:
        # Dériver du type de logement
        enriched_inputs['type_batiment'] = 'immeuble collectif d\'habitation' if enriched_inputs.get('type_logement') == 'Appartement' else 'maison'
    
    if 'type_installation_chauffage' not in enriched_inputs:
        # Estimation basée sur le type de chauffage
        type_chauffage = enriched_inputs.get('type_chauffage', 'Individuel')
        if type_chauffage == 'Collectif':
            enriched_inputs['type_installation_chauffage'] = 'chauffage collectif'
        else:
            enriched_inputs['type_installation_chauffage'] = 'chauffage individuel'
    
    if 'type_installation_ecs' not in enriched_inputs:
        # ECS = Eau Chaude Sanitaire
        type_chauffage = enriched_inputs.get('type_chauffage', 'Individuel')
        if type_chauffage == 'Collectif':
            enriched_inputs['type_installation_ecs'] = 'production ECS collective'
        else:
            enriched_inputs['type_installation_ecs'] = 'production ECS individuelle'
    
    # Estimations de consommation (valeurs moyennes basées sur la classe estimée)
    if 'conso_primaire' not in enriched_inputs:
        # Estimation grossière basée sur surface et année
        annee = enriched_inputs.get('annee_construction', 2000)
        surface = enriched_inputs.get('surface', 75)
        if annee >= 2012:
            base_conso = 120  # RT2012
        elif annee >= 2005:
            base_conso = 200
        elif annee >= 1990:
            base_conso = 280
        else:
            base_conso = 350
        enriched_inputs['conso_primaire'] = base_conso * surface
    
    if 'conso_finale' not in enriched_inputs:
        # Environ 0.8 de la conso primaire en moyenne
        enriched_inputs['conso_finale'] = enriched_inputs['conso_primaire'] * 0.8
    
    if 'cout_conso' not in enriched_inputs:
        # Estimation du coût (0.25€/kWh en moyenne)
        enriched_inputs['cout_conso'] = enriched_inputs['conso_finale'] * 0.25
    
    if 'code_postal' not in enriched_inputs:
        enriched_inputs['code_postal'] = '75000'
    
    # Créer le DataFrame avec TOUTES les colonnes enrichies
    df = pd.DataFrame([enriched_inputs])

    # Créer un DataFrame avec seulement les colonnes attendues
    df_final = pd.DataFrame(index=[0])
    
    # Ajouter les colonnes une par une depuis enriched_inputs
    for col in feature_columns:
        if col in enriched_inputs:
            df_final[col] = enriched_inputs[col]
        else:
            # Valeur par défaut si la colonne n'existe pas
            df_final[col] = 0
    
    return df_final

# ===================================
# PIPELINE DE PRÉDICTION
# ===================================
def predict_conso_and_dpe(features_df):
    # Prédire la consommation (régression)
    conso_kwh_m2_an = model_regression.predict(features_df)[0]
    
    # Déduire la classe DPE à partir de la consommation prédite
    dpe_viz = DPEVisualizer()
    dpe_pred = dpe_viz.get_class_from_conso(conso_kwh_m2_an)
    
    return conso_kwh_m2_an, dpe_pred

# ===================================
# SIMULATION RÉNOVATION
# ===================================
def apply_renovation(inputs_dict, renovation_type):
    inputs_apres = inputs_dict.copy()
    
    # Modifiez les features en fonction des travaux
    if "Isolation des murs" in renovation_type:
        inputs_apres['isolation_murs'] = 1
    if "Changement de chauffage" in renovation_type:
        # Améliorer le type de chauffage
        inputs_apres['type_chauffage'] = 'Electrique'
    if "Fenêtres double vitrage" in renovation_type:
        if 'fenetres' in inputs_apres:
            inputs_apres['fenetres'] = 'Double'
    if "Panneaux solaires" in renovation_type:
        if 'energie_renouvelable' in inputs_apres:
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
isolation_murs = st.sidebar.selectbox("Isolation des murs", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

# Dériver zone climatique de code_postal (simplifié)
try:
    zone_climatique = 'H1' if int(code_postal[:2]) < 50 else 'H2'
except:
    zone_climatique = 'H1'

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
    
    try:
        features_actuel = prepare_features(inputs_actuel)
        
        # Prédiction actuelle
        st.session_state.conso_avant, dpe_pred_avant = predict_conso_and_dpe(features_actuel)
        st.session_state.dpe_avant = dpe_pred_avant
        
        # Simulation rénovation
        if renovation_type:
            inputs_apres = apply_renovation(inputs_actuel, renovation_type)
            features_apres = prepare_features(inputs_apres)
            st.session_state.conso_apres, dpe_pred_apres = predict_conso_and_dpe(features_apres)
            st.session_state.dpe_apres = dpe_pred_apres
        else:
            st.session_state.conso_apres = None
            st.session_state.dpe_apres = None
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.session_state.calcul_lance = False

# ===================================
# RÉSULTATS
# ===================================
if st.session_state.calcul_lance and st.session_state.conso_avant is not None:
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
        
        # Afficher les gains
        st.subheader("📊 Gains estimés")
        reduction_conso = st.session_state.conso_avant - st.session_state.conso_apres
        reduction_pct = (reduction_conso / st.session_state.conso_avant) * 100
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Réduction de consommation", f"{reduction_conso:.1f} kWh/m²/an", f"-{reduction_pct:.1f}%")
        with col2:
            st.metric("Classe DPE", st.session_state.dpe_apres, f"{st.session_state.dpe_avant} → {st.session_state.dpe_apres}")
    else:
        st.info("Sélectionnez des travaux de rénovation pour simuler l'impact.")