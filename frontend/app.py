import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Configuration Streamlit
st.set_page_config(
    page_title="Prédiction Consommation Énergétique",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation session state
for key in ["calcul_lance", "conso_avant", "dpe_avant", "conso_apres", "dpe_apres", "profil_avant"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.session_state.calcul_lance = st.session_state.calcul_lance or False

# Valeurs par défaut pour colonnes manquantes
DEFAULT_VALUES = {
    'etiquette_dpe': 4.2,
    'etiquette_ges': 4.0,
    'conso_primaire': 180.0,
    'cout_conso': 1200.0,
}

# Classe visualisation DPE
class DPEVisualizer:
    def __init__(self):
        self.classes_dpe = {
            'A': {'min': 0,   'max': 50,  'color': '#319834'},
            'B': {'min': 51,  'max': 90,  'color': '#35B44A'},
            'C': {'min': 91,  'max': 150, 'color': '#C7D301'},
            'D': {'min': 151, 'max': 230, 'color': '#FFED00'},
            'E': {'min': 231, 'max': 330, 'color': '#FCAF17'},
            'F': {'min': 331, 'max': 450, 'color': '#EF7D08'},
            'G': {'min': 451, 'max': 999, 'color': '#E2001A'}
        }

    def calculer_classe_dpe(self, conso_kwh_m2_an):
        for classe, info in self.classes_dpe.items():
            if info['min'] <= conso_kwh_m2_an <= info['max']:
                return classe
        return 'G'

# Chargement des modèles
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'science', 'models')

    try:
        reg = joblib.load(os.path.join(MODEL_DIR, 'dpe_Regression_xgb.joblib'))
        clf = joblib.load(os.path.join(MODEL_DIR, 'dpe_classification_xgb.joblib'))

        with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'r') as f:
            feats = json.load(f)
        return reg, clf, feats
    except Exception as e:
        st.error(f"Erreur chargement modèles : {e}")
        st.stop()

model_regression, model_classification, feature_data = load_models()

numerical   = feature_data.get('numerical_features', [])
categorical = feature_data.get('categorical_features', [])
all_features = numerical + categorical

# CORRECTION: Pour la régression, on garde les étiquettes mais pas conso_finale
features_reg = [f for f in all_features if f not in {'conso_finale'}]

# CORRECTION: Pour la classification, on enlève les étiquettes mais on GARDE conso_finale
features_clf_base = [f for f in all_features if f not in {'etiquette_dpe', 'etiquette_ges'}]

# Fonction de préparation des features CORRIGÉE
def prepare_features(inputs_dict, model_type="regression", predicted_conso=None):
    d = inputs_dict.copy()

    # Mappings de base
    d.setdefault('surface_habitable_logement', d.get('surface', 80))
    d.setdefault('nombre_niveau_logement', 1 if d.get('type_logement') in ['Appartement', 'appartement'] else 2)
    d.setdefault('hauteur_sous_plafond', 2.5)
    d.setdefault('code_postal', '75000')

    # Mapping type_energie_principale
    if 'type_energie_principale' not in d:
        ch = d.get('energie_chauffage', d.get('type_chauffage', 'Electrique'))
        mapping = {
            'Électricité': 'Électricité',
            'Electrique': 'Électricité',
            'Gaz naturel': 'Gaz naturel',
            'Gaz': 'Gaz naturel',
            'Fioul': 'Fioul domestique',
            'Bois': 'Bois',
            'Pompe à chaleur': 'Électricité',
            'Réseau de chaleur': 'Réseau de chaleur',
            'Collectif': 'Gaz naturel',
            'Individuel': 'Électricité'
        }
        d['type_energie_principale'] = mapping.get(ch, 'Électricité')

    # Isolation
    if 'isolation' not in d:
        an = d.get('annee_construction', 1990)
        qual_iso = d.get('qualite_isolation', '')
        
        # Mapping de qualite_isolation vers isolation
        iso_mapping = {
            'Très bonne': 'Bonne',
            'Bonne': 'Bonne',
            'Moyenne': 'Moyenne',
            'Insuffisante': 'Faible'
        }
        
        if qual_iso in iso_mapping:
            d['isolation'] = iso_mapping[qual_iso]
        else:
            # Sinon on se base sur l'année
            d['isolation'] = 'Bonne' if an >= 2012 else 'Moyenne' if an >= 1990 else 'Faible'

    # Type de bâtiment
    type_log = d.get('type_logement', d.get('type_batiment', ''))
    if 'Appartement' in type_log or 'appartement' in type_log:
        d['type_batiment'] = "immeuble collectif d'habitation"
    else:
        d['type_batiment'] = 'maison'

    # Installation chauffage
    type_chauf = d.get('type_chauffage', 'Individuel')
    if 'Collectif' in type_chauf or 'collectif' in type_chauf:
        d['type_installation_chauffage'] = 'chauffage collectif'
    else:
        d['type_installation_chauffage'] = 'chauffage individuel'

    # Installation ECS
    type_ecs_input = d.get('type_ecs', '')
    if 'collectif' in type_ecs_input.lower() or 'Collectif' in type_chauf:
        d['type_installation_ecs'] = 'production ECS collective'
    else:
        d['type_installation_ecs'] = 'production ECS individuelle'

    # Création du DataFrame
    df = pd.DataFrame([d])

    # CORRECTION MAJEURE: Gestion différente selon le type de modèle
    if model_type == "regression":
        required = features_reg
        # Pour la régression, on garde etiquette_dpe et etiquette_ges
        for col in required:
            if col not in df.columns:
                df[col] = DEFAULT_VALUES.get(col, 0 if col in numerical else 'Inconnu')
        
        # On garde uniquement les colonnes de features_reg
        present_cols = [c for c in features_reg if c in df.columns]
        df = df[present_cols]
        
    else:  # classification
        # Pour la classification, on DOIT inclure conso_finale
        required = features_clf_base
        
        for col in required:
            if col not in df.columns:
                if col == 'conso_finale' and predicted_conso is not None:
                    df[col] = predicted_conso
                else:
                    df[col] = DEFAULT_VALUES.get(col, 0 if col in numerical else 'Inconnu')
        
        # S'assurer que conso_finale est bien présente
        if 'conso_finale' not in df.columns and predicted_conso is not None:
            df['conso_finale'] = predicted_conso
        
        # Ordre des colonnes: on met conso_finale à la fin
        cols_order = [c for c in features_clf_base if c != 'conso_finale' and c in df.columns]
        if 'conso_finale' in df.columns:
            cols_order.append('conso_finale')
        
        df = df[cols_order]

    return df

# Fonction de prédiction AVEC DEBUG
def predict_conso_and_dpe(inputs_dict, debug=False):
    # Étape 1: Prédiction de la consommation
    X_reg = prepare_features(inputs_dict, "regression")
    
    if debug:
        st.write("### DEBUG - Features pour régression:")
        st.dataframe(X_reg)
    
    conso_raw = float(model_regression.predict(X_reg)[0])
    
    # VÉRIFICATION CRITIQUE : Le modèle prédit-il la conso totale ou au m² ?
    surface = inputs_dict.get('surface', inputs_dict.get('surface_habitable_logement', 100))
    
    # Si la valeur prédite est > 1000, c'est probablement la conso TOTALE
    if conso_raw > 1000:
        # Le modèle prédit la consommation totale en kWh/an
        conso_totale = conso_raw
        conso = conso_raw / surface  # On calcule le kWh/m²/an
        if debug:
            st.write(f"### DEBUG - Le modèle prédit la CONSOMMATION TOTALE")
            st.write(f"- Consommation totale prédite: {conso_totale:.2f} kWh/an")
            st.write(f"- Surface: {surface} m²")
            st.write(f"- Consommation au m²: {conso:.2f} kWh/m²/an")
    else:
        # Le modèle prédit directement le kWh/m²/an
        conso = conso_raw
        if debug:
            st.write(f"### DEBUG - Le modèle prédit la CONSOMMATION AU M²")
            st.write(f"- Consommation prédite: {conso:.2f} kWh/m²/an")

    # Étape 2: Prédiction du DPE avec la consommation
    X_clf = prepare_features(inputs_dict, "classification", predicted_conso=conso)
    
    if debug:
        st.write("### DEBUG - Features pour classification:")
        st.dataframe(X_clf)
        st.write(f"Colonnes: {list(X_clf.columns)}")
        st.write(f"Valeur conso_finale: {X_clf['conso_finale'].values[0] if 'conso_finale' in X_clf.columns else 'MANQUANTE!'}")
    
    idx = int(model_classification.predict(X_clf)[0])
    
    if debug:
        st.write(f"### DEBUG - Index prédit: {idx}")
    
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    dpe = classes[idx] if 0 <= idx < len(classes) else 'G'

    return conso, dpe

# Visualisation DPE
def visualiser_dpe(conso_kwh_m2_an, predicteur, titre="Classe Énergétique"):
    classe = predicteur.calculer_classe_dpe(conso_kwh_m2_an)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")
    
    ax.text(5, 8.5, titre, ha='center', fontsize=13, fontweight='bold')
    
    y_start = 7
    for i, (c, info) in enumerate(predicteur.classes_dpe.items()):
        y = y_start - i
        length = 3.5 + i * 0.3
        
        arrow = FancyBboxPatch(
            (0.5, y - 0.4), length, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=info['color'],
            edgecolor='black',
            linewidth=2 if c == classe else 1
        )
        ax.add_patch(arrow)
        
        ax.text(0.8, y, c, fontsize=16, fontweight='bold',
                color='white', va='center')
        
        label = f"≤ {info['max']}" if c == 'A' else f"{info['min']}-{info['max']}" if c != 'G' else f"> {info['min']}"
        ax.text(length * 0.6, y, label, fontsize=9, va='center',
                color='white' if c in ['F', 'G'] else 'black', fontweight='bold')
        
        if c == classe:
            indicator = FancyBboxPatch(
                (6.5, y - 0.35), 2.2, 0.7,
                boxstyle="round,pad=0.05",
                facecolor='black', edgecolor='black'
            )
            ax.add_patch(indicator)
            ax.text(7.6, y, c, fontsize=15, fontweight='bold',
                    color='white', va='center', ha='center')
    
    ax.text(5, -0.3, f"{conso_kwh_m2_an:.1f} kWh/m²/an",
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.text(5, -0.9, f"Classe {classe}",
            ha='center', fontsize=12, fontweight='bold',
            color=predicteur.classes_dpe[classe]['color'])
    
    plt.tight_layout()
    return fig

# ────────────────────────────────────────────────
# INTERFACE
# ────────────────────────────────────────────────

st.title("Prédiction de Consommation Énergétique")
st.markdown("Prédisez votre consommation et simulez l'impact de travaux de rénovation")
st.markdown("---")

# MODE DEBUG (ajout d'une checkbox en sidebar)
debug_mode = st.sidebar.checkbox("Mode Debug", value=False)

# Sidebar
st.sidebar.header("Localisation")
code_postal = st.sidebar.text_input("Code postal", "75000", max_chars=5)
ville = st.sidebar.text_input("Ville", placeholder="Veuillez saisir la ville")

st.sidebar.markdown("---")
st.sidebar.header("Type de logement")
type_batiment = st.sidebar.selectbox(
    "Type de bâtiment",
    ["Veuillez sélectionner", "Maison", "Appartement"]
)

tab1, tab2 = st.tabs(["Informations du logement", "Scénarios de rénovation"])

# Tab 1 : Profil et prédiction
with tab1:
    st.subheader("Caractéristiques du logement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        surface = st.number_input("Surface habitable (m²)", min_value=10, max_value=500, value=100)
        nombre_niveaux = st.number_input("Nombre de niveaux", min_value=1, max_value=5, 
                                         value=1 if type_batiment == "Appartement" else 2)
    
    with col2:
        hauteur_sous_plafond = st.number_input("Hauteur sous plafond (m)", 
                                               min_value=2.0, max_value=4.0, value=2.5, step=0.1)
        type_chauffage = st.selectbox("Type installation chauffage", ["Veuillez sélectionner","Individuel", "Collectif"])
    
    st.markdown("---")
    st.subheader("Énergie et isolation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        energie_chauffage = st.selectbox(
            "Énergie principale chauffage",
            ["Veuillez sélectionner","Électricité", "Gaz naturel", "Fioul", "Bois", "Pompe à chaleur", "Réseau de chaleur"]
        )
        
        type_ecs = st.selectbox(
            "Type installation eau chaude sanitaire (ECS)",
            ["Veuillez sélectionner","Ballon électrique", "Chaudière", "Chauffe-eau thermodynamique", 
             "Chauffe-eau solaire", "Instantané gaz"]
        )
        
        qualite_isolation = st.selectbox(
            "Qualité isolation générale",
            ["Veuillez sélectionner","Insuffisante", "Moyenne", "Bonne", "Très bonne"]
        )
    
    with col2:
        isolation_murs = st.selectbox(
            "Isolation des murs",
            ["Veuillez sélectionner","Non isolé", "Partiellement isolé", "Bien isolé"]
        )
        
        isolation_sous_sol = st.selectbox(
            "Isolation du sous-sol",
            ["Veuillez sélectionner","Non isolé", "Isolé", "Pas de sous-sol"]
        )

        type_fenetres = st.selectbox(
            "Type de fenêtres",
            ["Veuillez sélectionner","Simple vitrage", "Double vitrage ancien", "Double vitrage récent", "Triple vitrage"]
        )
    
    st.markdown("---")
    
    if st.button("PRÉDIRE LA CONSOMMATION", type="primary", use_container_width=True):
        
        # Création profil
        profil = {
            'code_postal': code_postal,
            'ville': ville,
            'type_batiment': type_batiment,
            'surface': surface,
            'nombre_niveaux': nombre_niveaux,
            'hauteur_sous_plafond': hauteur_sous_plafond,
            'type_chauffage': type_chauffage,
            'energie_chauffage': energie_chauffage,
            'type_ecs': type_ecs,
            'qualite_isolation': qualite_isolation,
            'isolation_murs': isolation_murs,
            'isolation_sous_sol': isolation_sous_sol,
            'type_fenetres': type_fenetres,
            'annee_construction': 1990,
            'type_logement': type_batiment.lower() if type_batiment else 'maison',
            'zone_climatique': 'H2' if code_postal.startswith(('49','53','72')) else 'H1'
        }

        try:
            conso_kwh_m2, classe_dpe = predict_conso_and_dpe(profil, debug=debug_mode)
            conso_annuelle = conso_kwh_m2 * surface

            st.session_state.profil_initial = profil.copy()
            st.session_state.conso_annuelle_initiale = conso_annuelle
            st.session_state.conso_m2_an_initiale = conso_kwh_m2
            st.session_state.classe_initiale = classe_dpe
            st.session_state.prediction_faite = True

            st.success("Prédiction effectuée avec succès")

            st.markdown("---")
            st.subheader("Résultats de la prédiction")

            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Consommation annuelle", f"{conso_annuelle:,.0f} kWh/an")
            
            with col2:
                st.metric("Consommation/m²", f"{conso_kwh_m2:.1f} kWh/m²/an")
            
            with col3:
                cout_annuel = conso_annuelle * 0.18
                st.metric("Coût annuel estimé", f"{cout_annuel:,.0f} €/an")
            
            with col4:
                st.metric("Classe DPE prédite", classe_dpe)

            col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
            with col_center2:
                predicteur = DPEVisualizer()
                st.pyplot(visualiser_dpe(conso_kwh_m2, predicteur, "Classe énergétique estimé en fonction de la consommation"))

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {str(e)}")
            if debug_mode:
                st.exception(e)

# Tab 2 : Scénarios de rénovation
with tab2:
    if not st.session_state.get("prediction_faite", False):
        st.warning("Veuillez d'abord effectuer une prédiction dans l'onglet 'Informations du logement'")
    else:
        st.subheader("Choisissez un ou plusieurs scénarios de rénovation")
        st.info("Modifiez les caractéristiques du logement selon vos travaux envisagés")
        
        profil_initial = st.session_state.profil_initial.copy()
        profil_scenario = profil_initial.copy()
        
        # Scénario 1 : Agrandissement
        if profil_initial.get('type_batiment') == "Maison":
            with st.expander("Scénario 1 : Agrandissement (maisons uniquement)", expanded=False):
                scenario_1_actif = st.checkbox("Activer ce scénario", key="sc1")
                
                if scenario_1_actif:
                    st.write("Modifiez les caractéristiques liées à l'agrandissement :")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        nouvelle_surface = st.number_input(
                            "Nouvelle surface habitable (m²)",
                            min_value=profil_initial['surface'],
                            max_value=500,
                            value=profil_initial['surface'],
                            key="new_surface"
                        )
                        profil_scenario['surface'] = nouvelle_surface
                    
                    with col2:
                        nouveau_nb_niveaux = st.number_input(
                            "Nouveau nombre de niveaux",
                            min_value=profil_initial['nombre_niveaux'],
                            max_value=5,
                            value=profil_initial['nombre_niveaux'],
                            key="new_niveaux"
                        )
                        profil_scenario['nombre_niveaux'] = nouveau_nb_niveaux
                    
                    with col3:
                        nouvelle_hauteur = st.number_input(
                            "Nouvelle hauteur sous plafond (m)",
                            min_value=2.0,
                            max_value=4.0,
                            value=profil_initial['hauteur_sous_plafond'],
                            step=0.1,
                            key="new_hauteur"
                        )
                        profil_scenario['hauteur_sous_plafond'] = nouvelle_hauteur
        
        # Scénario 2 : Modernisation ECS
        with st.expander("Scénario 2 : Modernisation de l'eau chaude sanitaire", expanded=False):
            scenario_2_actif = st.checkbox("Activer ce scénario", key="sc2")
            
            if scenario_2_actif:
                st.write("Choisissez le nouveau système d'eau chaude :")
                
                nouveau_ecs = st.selectbox(
                    "Type d'installation ECS",
                    ["Ballon électrique", "Chaudière", "Chauffe-eau thermodynamique", 
                     "Chauffe-eau solaire", "Instantané gaz"],
                    index=2,
                    key="new_ecs"
                )
                profil_scenario['type_ecs'] = nouveau_ecs
        
        # Scénario 3 : Amélioration isolation
        with st.expander("Scénario 3 : Amélioration de l'isolation de l'enveloppe", expanded=False):
            scenario_3_actif = st.checkbox("Activer ce scénario", key="sc3")
            
            if scenario_3_actif:
                st.write("Améliorez la qualité de l'isolation :")
                
                index_default = ["Insuffisante", "Moyenne", "Bonne", "Très bonne"].index(profil_initial.get('qualite_isolation', "Moyenne")) + 1
                if index_default > 3:
                    index_default = 3
                    
                nouvelle_isolation = st.selectbox(
                    "Qualité isolation enveloppe",
                    ["Insuffisante", "Moyenne", "Bonne", "Très bonne"],
                    index=index_default,
                    key="new_isolation"
                )
                profil_scenario['qualite_isolation'] = nouvelle_isolation
        
        # Scénario 4 : Amélioration chauffage
        with st.expander("Scénario 4 : Amélioration du système de chauffage", expanded=False):
            scenario_4_actif = st.checkbox("Activer ce scénario", key="sc4")
            
            if scenario_4_actif:
                st.write("Modifiez le système de chauffage :")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    nouveau_type_chauffage = st.selectbox(
                        "Type d'installation chauffage",
                        ["Pas de modification","Individuel", "Collectif"],
                        index=0,
                        key="new_type_chauf"
                    )
                    if nouveau_type_chauffage != "Pas de modification":
                        profil_scenario['type_chauffage'] = nouveau_type_chauffage
                
                with col2:
                    nouvelle_energie = st.selectbox(
                        "Énergie principale chauffage",
                        ["Pas de modification","Électricité", "Gaz naturel", "Fioul", "Bois", "Pompe à chaleur", "Réseau de chaleur"],
                        index=0,
                        key="new_energie"
                    )
                    if nouvelle_energie != "Pas de modification":
                        profil_scenario['energie_chauffage'] = nouvelle_energie
        
        st.markdown("---")
        
        # Vérification scénarios actifs
        scenarios_actifs = []
        if profil_initial.get('type_batiment') == "Maison" and st.session_state.get("sc1", False):
            scenarios_actifs.append("Scénario 1 : Agrandissement")
        if st.session_state.get("sc2", False):
            scenarios_actifs.append("Scénario 2 : Modernisation ECS")
        if st.session_state.get("sc3", False):
            scenarios_actifs.append("Scénario 3 : Amélioration isolation")
        if st.session_state.get("sc4", False):
            scenarios_actifs.append("Scénario 4 : Amélioration chauffage")
        
        if scenarios_actifs:
            st.success(f"{len(scenarios_actifs)} scénario(s) activé(s) : {', '.join(scenarios_actifs)}")
        
        if st.button("SIMULER LE(S) SCÉNARIO(S)", type="primary", use_container_width=True):
            
            if not scenarios_actifs:
                st.warning("Veuillez activer au moins un scénario de rénovation")
            else:
                conso_apres_kwh_m2, classe_apres = predict_conso_and_dpe(profil_scenario, debug=debug_mode)
                conso_apres = conso_apres_kwh_m2 * profil_scenario['surface']

                conso_initiale = st.session_state.conso_annuelle_initiale
                conso_m2_initiale = st.session_state.conso_m2_an_initiale
                classe_initiale = st.session_state.classe_initiale

                economie_kwh_an = conso_initiale - conso_apres
                economie_m2 = conso_m2_initiale - conso_apres_kwh_m2
                economie_euros_an = economie_kwh_an * 0.18
                reduction_pct = (economie_kwh_an / conso_initiale) * 100 if conso_initiale > 0 else 0

                st.markdown("---")
                st.success(f"Simulation terminée ! {len(scenarios_actifs)} scénario(s) appliqué(s)")

                st.subheader("Comparaison AVANT / APRÈS rénovation")

                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Consommation annuelle",
                        f"{conso_apres:,.0f} kWh/an",
                        f"{economie_kwh_an:+,.0f} kWh",
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        "Consommation/m²",
                        f"{conso_apres_kwh_m2:.1f} kWh/m²/an",
                        f"{economie_m2:+.1f}",
                        delta_color="inverse"
                    )
                
                with col3:
                    cout_apres = conso_apres * 0.18
                    st.metric(
                        "Coût annuel",
                        f"{cout_apres:,.0f} €/an",
                        f"{economie_euros_an:+,.0f} €",
                        delta_color="inverse"
                    )
                
                with col4:
                    st.metric(
                        "Classe DPE prédite",
                        classe_apres,
                        f"{classe_initiale} → {classe_apres}"
                    )
                
                with col5:
                    st.metric(
                        "Réduction",
                        f"{abs(reduction_pct):.1f}%",
                        f"{economie_kwh_an:,.0f} kWh" if economie_kwh_an >= 0 else f"+{abs(economie_kwh_an):,.0f} kWh"
                    )

                st.markdown("---")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.pyplot(visualiser_dpe(conso_m2_initiale, DPEVisualizer(), "AVANT rénovation"))
                
                with col_g2:
                    st.pyplot(visualiser_dpe(conso_apres_kwh_m2, DPEVisualizer(), "APRÈS rénovation"))

                st.markdown("---")
                st.subheader("Récapitulatif des modifications")
                
                modifications = []
                for key, value_initial in profil_initial.items():
                    value_scenario = profil_scenario.get(key)
                    if value_scenario != value_initial:
                        nom_variable = {
                            'surface': 'Surface habitable',
                            'nombre_niveaux': 'Nombre de niveaux',
                            'hauteur_sous_plafond': 'Hauteur sous plafond',
                            'type_ecs': "Type d'installation ECS",
                            'qualite_isolation': "Qualité isolation enveloppe",
                            'type_chauffage': "Type d'installation chauffage",
                            'energie_chauffage': 'Énergie principale chauffage'
                        }.get(key, key)
                        
                        modifications.append({
                            'Variable': nom_variable,
                            'Avant': f"{value_initial}" + (" m²" if key == 'surface' else " m" if key == 'hauteur_sous_plafond' else ""),
                            'Après': f"{value_scenario}" + (" m²" if key == 'surface' else " m" if key == 'hauteur_sous_plafond' else "")
                        })
                
                if modifications:
                    df_modifs = pd.DataFrame(modifications)
                    st.dataframe(df_modifs, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune modification détectée dans les caractéristiques")

                st.markdown("---")
                st.subheader("Analyse de l'impact")
                
                if economie_kwh_an > 0:
                    st.success(
                        f"Réduction de la consommation de {economie_kwh_an:,.0f} kWh/an "
                        f"({reduction_pct:.1f} %)"
                    )
                    st.write(
                        f"Économie financière estimée : {economie_euros_an:,.0f} € par an"
                    )

                    if classe_apres < classe_initiale:
                        st.write(
                            "Amélioration de la performance énergétique du logement, "
                            "avec un gain de classe énergétique."
                        )
                    else:
                        st.write(
                            "La classe énergétique reste identique, "
                            "mais la consommation et les coûts sont réduits."
                        )

                elif economie_kwh_an == 0:
                    st.warning(
                        "Les scénarios sélectionnés n'ont pas d'impact significatif "
                        "sur la consommation énergétique estimée."
                    )
                else:
                    st.error(
                        "La consommation estimée augmente. "
                        "Cela peut être dû à un agrandissement ou à un changement défavorable."
                    )