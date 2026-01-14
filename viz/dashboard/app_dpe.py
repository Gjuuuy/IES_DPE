import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np

# ===================================
# CONFIGURATION STREAMLIT
# ===================================

st.set_page_config(
    page_title="Prédiction Consommation Énergétique",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================================
# INITIALISATION SESSION STATE
# ===================================

if "prediction_faite" not in st.session_state:
    st.session_state.prediction_faite = False

# ===================================
# CLASSE MODÈLE PRÉDICTION
# ===================================

class PredicteurConsommation:
    """
    Classe pour prédire la consommation énergétique
    À remplacer par votre modèle ML entraîné
    """
    
    def __init__(self):
        # Classes DPE pour conversion consommation -> classe
        self.classes_dpe = {
            'A': {'min': 0,   'max': 50,  'color': '#319834'},
            'B': {'min': 51,  'max': 90,  'color': '#35B44A'},
            'C': {'min': 91,  'max': 150, 'color': '#C7D301'},
            'D': {'min': 151, 'max': 230, 'color': '#FFED00'},
            'E': {'min': 231, 'max': 330, 'color': '#FCAF17'},
            'F': {'min': 331, 'max': 450, 'color': '#EF7D08'},
            'G': {'min': 451, 'max': 999, 'color': '#E2001A'}
        }
    
    def predire_consommation_annuelle(self, profil):
        """
        FONCTION À REMPLACER PAR le MODÈLE ML
        
        Input: dictionnaire avec toutes les caractéristiques du logement
        Output: consommation annuelle en kWh/an
        """
        
        # ========================================
        # MODÈLE SIMPLIFIÉ (à remplacer)
        # ========================================
        
        surface = profil['surface']
        
        # Consommation de base par m² selon qualité isolation
        conso_base_m2 = {
            'Insuffisante': 180,
            'Moyenne': 140,
            'Bonne': 100,
            'Très bonne': 70
        }[profil['qualite_isolation']]
        
        # Facteurs multiplicateurs
        facteurs = {
            'energie': {
                'Électricité': 1.2,
                'Gaz naturel': 1.0,
                'Fioul': 1.3,
                'Bois': 0.8,
                'Pompe à chaleur': 0.6,
                'Réseau de chaleur': 0.85
            },
            'fenetres': {
                'Simple vitrage': 1.3,
                'Double vitrage ancien': 1.1,
                'Double vitrage récent': 1.0,
                'Triple vitrage': 0.85
            },
            'isolation_murs': {
                'Non isolé': 1.2,
                'Partiellement isolé': 1.0,
                'Bien isolé': 0.85
            },
            'type_batiment': {
                'Maison individuelle': 1.1,
                'Appartement': 0.9
            },
            'ecs': {
                'Ballon électrique': 1.2,
                'Chaudière': 1.0,
                'Chauffe-eau thermodynamique': 0.6,
                'Chauffe-eau solaire': 0.5,
                'Instantané gaz': 0.9
            }
        }
        
        # Calcul de la consommation
        conso_m2 = conso_base_m2
        conso_m2 *= facteurs['energie'].get(profil['energie_chauffage'], 1.0)
        conso_m2 *= facteurs['fenetres'].get(profil['type_fenetres'], 1.0)
        conso_m2 *= facteurs['isolation_murs'].get(profil['isolation_murs'], 1.0)
        conso_m2 *= facteurs['type_batiment'].get(profil['type_batiment'], 1.0)
        conso_m2 *= facteurs['ecs'].get(profil['type_ecs'], 1.0)
        
        # Ajustement hauteur sous plafond
        if profil['hauteur_sous_plafond'] > 2.7:
            conso_m2 *= 1.1
        
        # Ajustement nombre de niveaux
        if profil['nombre_niveaux'] > 2:
            conso_m2 *= 1.05
        
        # Ajustement type chauffage
        if profil['type_chauffage'] == 'Collectif':
            conso_m2 *= 0.95  # Légèrement plus efficace
        
        # Consommation annuelle totale
        conso_annuelle = conso_m2 * surface
        
        return conso_annuelle
    
    def calculer_classe_dpe(self, conso_kwh_m2_an):
        """Détermine la classe DPE depuis la consommation/m²/an"""
        for classe, info in self.classes_dpe.items():
            if info['min'] <= conso_kwh_m2_an <= info['max']:
                return classe
        return 'G'


# ===================================
# VISUALISATION DPE
# ===================================

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


# ===================================
# INTERFACE PRINCIPALE
# ===================================

st.title("🏠 Prédiction de Consommation Énergétique")
st.markdown("Prédisez votre consommation et simulez l'impact de travaux de rénovation")
st.markdown("---")

# ===================================
# SIDEBAR - INFORMATIONS LOGEMENT
# ===================================

st.sidebar.header("📍 Localisation")
code_postal = st.sidebar.text_input("Code postal", "0", max_chars=5)
ville = st.sidebar.text_input("Ville", placeholder="Veuillez saisir la ville")

st.sidebar.markdown("---")
st.sidebar.header("🏠 Type de logement")
type_batiment = st.sidebar.selectbox(
    "Type de bâtiment",
    ["Veuillez sélectionner","Maison", "Appartement"]
)

st.sidebar.markdown("---")
st.sidebar.header("📊 DPE/GES actuels (optionnel)")
dpe_actuel = st.sidebar.selectbox(
    "Étiquette DPE actuelle",
    ["Non renseignée", "A", "B", "C", "D", "E", "F", "G"]
)
ges_actuel = st.sidebar.selectbox(
    "Étiquette GES actuelle",
    ["Non renseignée", "A", "B", "C", "D", "E", "F", "G"]
)

# ===================================
# ONGLETS PRINCIPAUX
# ===================================

tab1, tab2 = st.tabs(["📋 Informations du logement", "🔧 Scénarios de rénovation"])

# ===================================
# TAB 1: PROFIL ET PRÉDICTION
# ===================================

with tab1:
    st.subheader("📐 Caractéristiques du logement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        surface = st.number_input("Surface habitable (m²)", min_value=0, max_value=500, value=0)
        nombre_niveaux = st.number_input("Nombre de niveaux", min_value=1, max_value=5, 
                                         value=1 if type_batiment == "Appartement" else 2)
    
    with col2:
        hauteur_sous_plafond = st.number_input("Hauteur sous plafond (m)", 
                                               min_value=0.0, max_value=4.0, value=0.0, step=0.1)
        type_chauffage = st.selectbox("Type installation chauffage", ["Veuillez sélectionner","Individuel", "Collectif"])
    
   
    
    st.markdown("---")
    st.subheader("🔥 Énergie et isolation")
    
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
    
    # Bouton de prédiction
    if st.button("🔮 PRÉDIRE LA CONSOMMATION", type="primary", use_container_width=True):
        
        # Créer le profil complet
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
            'dpe_actuel': dpe_actuel,
            'ges_actuel': ges_actuel
        }
        
        # Prédiction
        predicteur = PredicteurConsommation()
        conso_annuelle = predicteur.predire_consommation_annuelle(profil)
        conso_m2_an = conso_annuelle / surface
        classe_dpe = predicteur.calculer_classe_dpe(conso_m2_an)
        
        # Stocker dans session_state
        st.session_state.profil_initial = profil.copy()
        st.session_state.conso_annuelle_initiale = conso_annuelle
        st.session_state.conso_m2_an_initiale = conso_m2_an
        st.session_state.classe_initiale = classe_dpe
        st.session_state.prediction_faite = True
        
        # Affichage résultats
        st.success("✅ Prédiction effectuée avec succès !")
        
        st.markdown("---")
        st.subheader("📊 Résultats de la prédiction")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Consommation annuelle", f"{conso_annuelle:,.0f} kWh/an")
        
        with col2:
            st.metric("Consommation/m²", f"{conso_m2_an:.1f} kWh/m²/an")
        
        with col3:
            cout_annuel = conso_annuelle * 0.18  # Prix moyen kWh
            st.metric("Coût annuel estimé", f"{cout_annuel:,.0f} €/an")
        
        with col4:
            st.metric("Classe DPE", classe_dpe)
        
        # Visualisation DPE
        col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
        with col_center2:
            st.pyplot(visualiser_dpe(conso_m2_an, predicteur, "Classe Énergétique Prédite"))
        
        st.info("💡 Passez à l'onglet 'Scénarios de rénovation' pour simuler des travaux et voir l'impact sur la consommation")

# ===================================
# TAB 2: SCÉNARIOS DE RÉNOVATION
# ===================================

with tab2:
    if not st.session_state.prediction_faite:
        st.warning("⚠️ Veuillez d'abord effectuer une prédiction dans l'onglet 'Informations du logement'")
    else:
        st.subheader("🔧 Choisissez un ou plusieurs scénarios de rénovation")
        st.info("💡 Modifiez les caractéristiques du logement selon vos travaux envisagés")
        
        # Récupérer le profil initial
        profil_initial = st.session_state.profil_initial.copy()
        
        # Créer un nouveau profil qui sera modifié
        profil_scenario = profil_initial.copy()
        
        # ===================================
        # SCÉNARIO 1: AGRANDISSEMENT (MAISONS UNIQUEMENT)
        # ===================================
        
        if profil_initial['type_batiment'] == "Maison":
            with st.expander("📐 **Scénario 1 : Agrandissement** (maisons uniquement)", expanded=False):
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
                    
        
        
        # ===================================
        # SCÉNARIO 2: MODERNISATION ECS
        # ===================================
        
        with st.expander("💧 **Scénario 2 : Modernisation de l'eau chaude sanitaire**", expanded=False):
            scenario_2_actif = st.checkbox("Activer ce scénario", key="sc2")
            
            if scenario_2_actif:
                st.write("Choisissez le nouveau système d'eau chaude :")
                
                nouveau_ecs = st.selectbox(
                    "Type d'installation ECS",
                    ["Ballon électrique", "Chaudière", "Chauffe-eau thermodynamique", 
                     "Chauffe-eau solaire", "Instantané gaz"],
                    index=2,  # Par défaut sur thermodynamique
                    key="new_ecs"
                )
                profil_scenario['type_ecs'] = nouveau_ecs
                
                
        
        # ===================================
        # SCÉNARIO 3: AMÉLIORATION ISOLATION
        # ===================================
        
        with st.expander("🏠 **Scénario 3 : Amélioration de l'isolation de l'enveloppe**", expanded=False):
            scenario_3_actif = st.checkbox("Activer ce scénario", key="sc3")
            
            if scenario_3_actif:
                st.write("Améliorez la qualité de l'isolation :")
                
                nouvelle_isolation = st.selectbox(
                    "Qualité isolation enveloppe",
                    ["Insuffisante", "Moyenne", "Bonne", "Très bonne"],
                    index=["Insuffisante", "Moyenne", "Bonne", "Très bonne"].index(profil_initial['qualite_isolation']) + 1 
                          if profil_initial['qualite_isolation'] != "Très bonne" 
                          else 3,
                    key="new_isolation"
                )
                profil_scenario['qualite_isolation'] = nouvelle_isolation
                
        
        # ===================================
        # SCÉNARIO 4: AMÉLIORATION CHAUFFAGE
        # ===================================
        
        with st.expander("🔥 **Scénario 4 : Amélioration du système de chauffage**", expanded=False):
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
                    profil_scenario['type_chauffage'] = nouveau_type_chauffage
                
                with col2:
                    nouvelle_energie = st.selectbox(
                        "Énergie principale chauffage",
                        ["Pas de modification","Électricité", "Gaz naturel", "Fioul", "Bois", "Pompe à chaleur", "Réseau de chaleur"],
                        index=0,  # Par défaut sur Pompe à chaleur
                        key="new_energie"
                    )
                    profil_scenario['energie_chauffage'] = nouvelle_energie
                
                
        
        # ===================================
        # BOUTON SIMULATION
        # ===================================
        
        st.markdown("---")
        
        # Vérifier si au moins un scénario est actif
        scenarios_actifs = []
        if profil_initial['type_batiment'] == "Maison individuelle" and 'sc1' in st.session_state and st.session_state.sc1:
            scenarios_actifs.append("Scénario 1 : Agrandissement")
        if 'sc2' in st.session_state and st.session_state.sc2:
            scenarios_actifs.append("Scénario 2 : Modernisation ECS")
        if 'sc3' in st.session_state and st.session_state.sc3:
            scenarios_actifs.append("Scénario 3 : Amélioration isolation")
        if 'sc4' in st.session_state and st.session_state.sc4:
            scenarios_actifs.append("Scénario 4 : Amélioration chauffage")
        
        if scenarios_actifs:
            st.success(f"✅ {len(scenarios_actifs)} scénario(s) activé(s) : {', '.join(scenarios_actifs)}")
        
        if st.button("📊 SIMULER LE(S) SCÉNARIO(S)", type="primary", use_container_width=True):
            
            if not scenarios_actifs:
                st.warning("⚠️ Veuillez activer au moins un scénario de rénovation")
            else:
                # Prédiction avec le nouveau profil
                predicteur = PredicteurConsommation()
                conso_apres = predicteur.predire_consommation_annuelle(profil_scenario)
                conso_m2_apres = conso_apres / profil_scenario['surface']
                classe_apres = predicteur.calculer_classe_dpe(conso_m2_apres)
                
                # Récupération des valeurs initiales
                conso_initiale = st.session_state.conso_annuelle_initiale
                conso_m2_initiale = st.session_state.conso_m2_an_initiale
                classe_initiale = st.session_state.classe_initiale
                surface_initiale = profil_initial['surface']
                
                # Calculs des gains
                economie_kwh_an = conso_initiale - conso_apres
                economie_m2 = conso_m2_initiale - conso_m2_apres
                economie_euros_an = economie_kwh_an * 0.18
                reduction_pct = (economie_kwh_an / conso_initiale) * 100 if conso_initiale > 0 else 0
                
                # Affichage résultats
                st.markdown("---")
                st.success(f"✅ Simulation terminée ! {len(scenarios_actifs)} scénario(s) appliqué(s)")
                
                st.subheader("📈 Comparaison AVANT / APRÈS rénovation")
                
                # Métriques comparatives
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
                        f"{conso_m2_apres:.1f} kWh/m²/an",
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
                        "Classe DPE",
                        classe_apres,
                        f"{classe_initiale} → {classe_apres}"
                    )
                
                with col5:
                    st.metric(
                        "Réduction",
                        f"{abs(reduction_pct):.1f}%",
                        f"{economie_kwh_an:,.0f} kWh" if economie_kwh_an >= 0 else f"+{abs(economie_kwh_an):,.0f} kWh"
                    )
                
                # Graphiques DPE avant/après
                st.markdown("---")
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.pyplot(visualiser_dpe(conso_m2_initiale, predicteur, "AVANT rénovation"))
                
                with col_g2:
                    st.pyplot(visualiser_dpe(conso_m2_apres, predicteur, "APRÈS rénovation"))
                
                # Tableau comparatif des variables modifiées
                st.markdown("---")
                st.subheader("📋 Récapitulatif des modifications")
                
                modifications = []
                
                # Vérifier les changements
                for key, value_initial in profil_initial.items():
                    value_scenario = profil_scenario[key]
                    if value_initial != value_scenario:
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
                
                # Analyse de l'impact
                st.markdown("---")
                st.subheader("💡 Analyse de l'impact")
            
                if economie_kwh_an > 0:
                    st.success(
                        f"✅ **Réduction de la consommation de {economie_kwh_an:,.0f} kWh/an** "
                        f"({reduction_pct:.1f} %)"
                    )
                    st.write(
                        f"💰 **Économie financière estimée : {economie_euros_an:,.0f} € par an**"
                    )

                    if classe_apres < classe_initiale:
                        st.write(
                            "🏷️ **Amélioration de la performance énergétique du logement**, "
                            "avec un gain de classe énergétique."
                        )
                    else:
                        st.write(
                            "🏷️ La classe énergétique reste identique, "
                            "mais la consommation et les coûts sont réduits."
                        )

                elif economie_kwh_an == 0:
                    st.warning(
                        "⚠️ Les scénarios sélectionnés n'ont pas d'impact significatif "
                        "sur la consommation énergétique estimée."
                    )
                else:
                    st.error(
                        "❌ La consommation estimée augmente. "
                        "Cela peut être dû à un agrandissement ou à un changement défavorable."
                    )

               