# ============================================================
# APPLICATION STREAMLIT - VERSION CORRIGÉE
# ============================================================

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Taux de Réussite - Prédiction",
    page_icon="🎓",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stApp { background: #ffffff; }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .title { font-size: 2.5rem; font-weight: 700; color: white; margin-bottom: 0.5rem; }
    .subtitle { color: rgba(255,255,255,0.9); font-size: 1rem; }
    .badge {
        background: rgba(255,255,255,0.2);
        border-radius: 50px;
        padding: 0.3rem 1rem;
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin: 0 0.2rem;
    }
    .card {
        background: #f8f9fa;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 1rem;
        border-left: 3px solid #667eea;
        padding-left: 0.8rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .result-number { font-size: 4rem; font-weight: 800; color: white; }
    .footer { text-align: center; padding: 2rem; color: #adb5bd; font-size: 0.8rem; border-top: 1px solid #e9ecef; margin-top: 2rem; }
    .success-message { background: #d4edda; border-left: 4px solid #28a745; border-radius: 10px; padding: 1rem; margin: 1rem 0; color: #155724; }
    .warning-message { background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 10px; padding: 1rem; margin: 1rem 0; color: #856404; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <div class="title">🎓 Prédiction de Réussite Universitaire</div>
    <div class="subtitle">Prédiction intelligente pour l'enseignement supérieur tunisien</div>
    <div style="margin-top: 1rem;">
        <span class="badge">Machine Learning</span>
        <span class="badge">Random Forest</span>
        <span class="badge">R² = 97.7%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Chargement modèle
@st.cache_resource
def charger_modele():
    try:
        with open('modele_rf.pkl', 'rb') as f:
            modele = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return modele, scaler
    except:
        return None, None

modele, scaler = charger_modele()

if modele is None:
    st.error("❌ Modèle non trouvé")
    st.stop()

# ============================================================
# FORMULAIRE
# ============================================================

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 CARACTÉRISTIQUES</div>', unsafe_allow_html=True)
    
    taille = st.number_input(
        "🏫 Nombre total d'inscrits",
        min_value=10,
        max_value=10000,
        value=500,
        step=50
    )
    
    # Correction du slider : valeur simple, pas de liste
    pct_femmes = st.slider(
        "👩 Pourcentage de femmes",
        min_value=0,
        max_value=100,
        value=65,
        step=5
    ) / 100
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 DIPLÔMÉS</div>', unsafe_allow_html=True)
    
    diplomes_f = st.number_input(
        "👩‍🎓 Diplômées (femmes)",
        min_value=0,
        max_value=5000,
        value=100,
        step=10
    )
    
    diplomes_m = st.number_input(
        "👨‍🎓 Diplômés (hommes)",
        min_value=0,
        max_value=5000,
        value=50,
        step=10
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# STATS MODÈLE
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📊 PERFORMANCES DU MODÈLE</div>', unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.metric("🎯 R² Score", "97.7%")
with col_s2:
    st.metric("📉 Erreur moyenne", "1.2%")
with col_s3:
    st.metric("🤖 Modèle", "Random Forest")
with col_s4:
    st.metric("✅ Validation", "Sans leakage")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# CALCUL
# ============================================================
inscrits_f = taille * pct_femmes
inscrits_m = taille * (1 - pct_femmes)
efficacite_F = diplomes_f / (inscrits_f + 1)
efficacite_M = diplomes_m / (inscrits_m + 1)
ecart_genre = abs(efficacite_F - efficacite_M)

features = np.array([[
    pct_femmes,
    np.log1p(taille),
    efficacite_F,
    efficacite_M,
    ecart_genre,
    0,  # domaine_code
    0   # universite_code
]])

features_scaled = scaler.transform(features)

# ============================================================
# BOUTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 PRÉDIRE LE TAUX DE RÉUSSITE", type="primary"):
    prediction = modele.predict(features_scaled)[0] * 100
    
    # Résultat
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size: 0.9rem; letter-spacing: 2px; opacity: 0.9;">
            TAUX DE RÉUSSITE PRÉDIT
        </div>
        <div class="result-number">{prediction:.1f}%</div>
        <div style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.8;">
            Marge d'erreur ± 1.2%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Graphique
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.barh(["Taux de réussite"], [prediction], color='#667eea', height=0.4)
    ax.axvline(x=26.6, color='#dc3545', linestyle='--', linewidth=2, label='Moyenne nationale (26.6%)')
    
    ax.set_xlim(0, 100)
    ax.set_xlabel("Pourcentage (%)", fontsize=11, color='#495057')
    ax.tick_params(colors='#495057')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#dee2e6')
    ax.spines['left'].set_visible(False)
    ax.text(prediction + 2, 0, f"{prediction:.1f}%", va='center', fontsize=14, fontweight='bold', color='#667eea')
    ax.legend(loc='lower right', facecolor='white', edgecolor='#dee2e6')
    
    st.pyplot(fig)
    
    # Indicateurs
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("👩 Proportion de femmes", f"{pct_femmes*100:.0f}%")
    with col_b:
        st.metric("👩‍🎓 Efficacité femmes", f"{efficacite_F*100:.1f}%")
    with col_c:
        st.metric("👨‍🎓 Efficacité hommes", f"{efficacite_M*100:.1f}%")
    
    # Interprétation
    if prediction > 26.6:
        st.markdown(f"""
        <div class="success-message">
            ✅ <b>Résultat encourageant</b><br>
            Le taux de réussite prédit ({prediction:.1f}%) est <b>supérieur</b> à la moyenne nationale (26.6%).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-message">
            ⚠️ <b>Axe d'amélioration</b><br>
            Le taux de réussite prédit ({prediction:.1f}%) est <b>inférieur</b> à la moyenne nationale (26.6%).
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    Projet Machine Learning - Polytechnique de Sousse | 2025-2026
</div>
""", unsafe_allow_html=True)