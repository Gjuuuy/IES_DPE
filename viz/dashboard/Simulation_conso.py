import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ===================================
# ÉTAT DE L'APPLICATION
# ===================================

if "calcul_lance" not in st.session_state:
    st.session_state.calcul_lance = False

# ===================================
# CLASSE DPE
# ===================================

class DPE:
    def __init__(self):
        self.classes = {
            'A': {'min': 0,   'max': 50,  'color': '#319834', 'label': '≤ 50'},
            'B': {'min': 51,  'max': 90,  'color': '#35B44A', 'label': '51 à 90'},
            'C': {'min': 91,  'max': 150, 'color': '#C7D301', 'label': '91 à 150'},
            'D': {'min': 151, 'max': 230, 'color': '#FFED00', 'label': '151 à 230'},
            'E': {'min': 231, 'max': 330, 'color': '#FCAF17', 'label': '231 à 330'},
            'F': {'min': 331, 'max': 450, 'color': '#EF7D08', 'label': '331 à 450'},
            'G': {'min': 451, 'max': 999, 'color': '#E2001A', 'label': '> 450'}
        }
        self.order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    def calculer_classe(self, conso_kwh_m2_an):
        for c in self.order:
            if self.classes[c]['min'] <= conso_kwh_m2_an <= self.classes[c]['max']:
                return c
        return 'G'

    def calculer_conso_m2_an(self, conso_mensuelle, surface):
        return (conso_mensuelle * 12) / surface


# ===================================
# VISUALISATION DPE
# ===================================
def visualiser_dpe(consommation_kwh_m2_an, dpe):
    classe = dpe.calculer_classe(consommation_kwh_m2_an)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(dpe.order) + 2)
    ax.axis("off")

    ax.text(0.5, 1.05, "Diagnostic de Performance Énergétique",
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=16, fontweight='bold')

    ax.text(0.01, 0.85, "Logement économe",
            transform=ax.transAxes, fontsize=11, style='italic')

    ax.text(0.01, 0.01, "Logement énergivore",
            transform=ax.transAxes, fontsize=11, style='italic')

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
            linewidth=2 if c == classe else 1
        )
        ax.add_patch(arrow)

        ax.text(0.8, y, c, fontsize=18,
                fontweight='bold', color='white', va='center')

        ax.text(length * 0.6, y, info['label'],
                fontsize=10, va='center',
                color='white' if c in ['F', 'G'] else 'black',
                fontweight='bold')

        # ✅ ÉTIQUETTE À DROITE — BIEN DANS LA BOUCLE
        if c == classe:
            indicator = FancyBboxPatch(
                (7, y - 0.35), 2.5, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=info['color'],
                edgecolor=info['color']
            )
            ax.add_patch(indicator)
            ax.text(8.25, y, c,
                    fontsize=16, fontweight='bold',
                    color='white', va='center', ha='center')

    ax.text(5, -1.2,
            f"Consommation : {consommation_kwh_m2_an:.1f} kWh/m²/an",
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgray'))

    ax.text(5, -1.8,
            f"Classe énergétique : {classe}",
            ha='center', fontsize=14,
            fontweight='bold',
            color=dpe.classes[classe]['color'])

    return fig


# ===================================
# STREAMLIT CONFIG
# ===================================

st.set_page_config(page_title="Simulateur DPE", layout="centered")
st.title("🏠 Simulateur de DPE")

# ===================================
# SIDEBAR – INPUTS
# ===================================

st.sidebar.header("Informations sur le logement")

surface = st.sidebar.number_input("Surface du logement (m²)", 10, 500, 75)
conso_mensuelle = st.sidebar.number_input("Consommation mensuelle (kWh/mois)", 50, 3000, 500)

st.sidebar.selectbox("Type de logement", ["Maison", "Appartement"])
st.sidebar.selectbox("Type de chauffage", ["Individuel", "Collectif"])
st.sidebar.text_input("Code postal", "75000")

# ===================================
# 🔘 BOUTON
# ===================================

if st.button("🔍 Faire la simulation"):
    st.session_state.calcul_lance = True

# ===================================
# RÉSULTATS
# ===================================

if st.session_state.calcul_lance:

    st.subheader("📈 Consommation énergétique mensuelle")

    unite = st.selectbox("Afficher la consommation en :", ["kWh", "€"])
    PRIX_KWH = 0.2516
    st.caption("Tarif standard: 0,2516 €/kWh")


    mois = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
            "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

    coeffs = [1.4, 1.3, 1.2, 0.9, 0.7, 0.6,
              0.5, 0.5, 0.7, 1.0, 1.3, 1.6]

    conso_kwh = [conso_mensuelle * c for c in coeffs]

    if unite == "€":
        valeurs = [c * PRIX_KWH for c in conso_kwh]
        ylabel = "Coût (€)"
    else:
        valeurs = conso_kwh
        ylabel = "Consommation (kWh)"

    fig_conso, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mois, valeurs, marker='o', linewidth=2)
    ax.set_title(f"Consommation mensuelle en {unite}")
    ax.set_xlabel("Mois")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    st.pyplot(fig_conso)

    dpe = DPE()
    conso_kwh_m2_an = dpe.calculer_conso_m2_an(conso_mensuelle, surface)

    st.subheader("🏷️ Classe énergétique du logement")
    st.pyplot(visualiser_dpe(conso_kwh_m2_an, dpe))
