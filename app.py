# ============================================================
# APPLICATION STREAMLIT - VERSION SIMPLIFIÉE
# ============================================================

import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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
    .download-btn {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header (sans les badges)
st.markdown("""
<div class="header">
    <div class="title">🎓 Prédiction de Réussite Universitaire</div>
    <div class="subtitle">Prédiction intelligente pour l'enseignement supérieur tunisien</div>
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
# FONCTION POUR GÉNÉRER LE RAPPORT PDF
# ============================================================
def generer_rapport(prediction, taille, pct_femmes, diplomes_f, diplomes_m, 
                    inscrits_f, inscrits_m, efficacite_F, efficacite_M, ecart_genre):
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Style personnalisé pour le titre
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor('#667eea'),
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    # Style pour les sous-titres
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=10
    )
    
    # Contenu du rapport
    story = []
    
    # Titre
    story.append(Paragraph("Rapport de Prédiction - Taux de Réussite Universitaire", title_style))
    story.append(Spacer(1, 12))
    
    # Date
    date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], alignment=TA_CENTER)
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}", date_style))
    story.append(Spacer(1, 20))
    
    # Informations saisies
    story.append(Paragraph("1. INFORMATIONS SAISIES", subtitle_style))
    
    data = [
        ["Paramètre", "Valeur"],
        ["Nombre total d'inscrits", f"{taille} étudiants"],
        ["Pourcentage de femmes", f"{pct_femmes*100:.0f}%"],
        ["Nombre de femmes inscrites", f"{inscrits_f:.0f}"],
        ["Nombre d'hommes inscrits", f"{inscrits_m:.0f}"],
        ["Nombre de diplômées (femmes)", f"{diplomes_f}"],
        ["Nombre de diplômés (hommes)", f"{diplomes_m}"]
    ]
    
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (1, -1), 1, colors.grey)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Résultat de la prédiction
    story.append(Paragraph("2. RÉSULTAT DE LA PRÉDICTION", subtitle_style))
    
    result_data = [
        ["Taux de réussite prédit", f"{prediction:.1f}%"],
        ["Moyenne nationale", "26.6%"],
        ["Comparaison", "Supérieur à la moyenne" if prediction > 26.6 else "Inférieur à la moyenne"]
    ]
    
    t2 = Table(result_data)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#28a745')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, -1), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.lightgreen),
        ('GRID', (0, 0), (1, -1), 1, colors.grey)
    ]))
    story.append(t2)
    story.append(Spacer(1, 20))
    
    # Analyse détaillée
    story.append(Paragraph("3. ANALYSE DÉTAILLÉE", subtitle_style))
    
    analyse_data = [
        ["Indicateur", "Valeur", "Interprétation"],
        ["Efficacité des femmes", f"{efficacite_F*100:.1f}%", "Taux de diplomation des femmes"],
        ["Efficacité des hommes", f"{efficacite_M*100:.1f}%", "Taux de diplomation des hommes"],
        ["Écart de réussite H/F", f"{ecart_genre*100:.1f}%", "Différence entre les genres"]
    ]
    
    t3 = Table(analyse_data)
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.HexColor('#764ba2')),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
        ('ALIGN', (0, 0), (2, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (2, -1), 10),
        ('BACKGROUND', (0, 1), (2, -1), colors.lavender),
        ('GRID', (0, 0), (2, -1), 1, colors.grey)
    ]))
    story.append(t3)
    story.append(Spacer(1, 20))
    
    # Informations sur le modèle
    story.append(Paragraph("4. INFORMATIONS SUR LE MODÈLE", subtitle_style))
    story.append(Paragraph("• Modèle utilisé: Random Forest", styles['Normal']))
    story.append(Paragraph("• R² (coefficient de détermination): 97.7%", styles['Normal']))
    story.append(Paragraph("• MAE (erreur moyenne): 1.2%", styles['Normal']))
    story.append(Paragraph("• Variables les plus importantes: efficacité des femmes (48.8%), efficacité des hommes (45.7%)", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Footer
    story.append(Paragraph("Rapport généré automatiquement par l'application de prédiction", ParagraphStyle('Footer', parent=styles['Normal'], alignment=TA_CENTER, fontSize=8, textColor=colors.grey)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

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
    # BOUTON DE TÉLÉCHARGEMENT DU RAPPORT
    # ============================================================
    st.markdown("---")
    
    # Générer le rapport PDF
    rapport_pdf = generer_rapport(
        prediction, taille, pct_femmes, diplomes_f, diplomes_m,
        inscrits_f, inscrits_m, efficacite_F, efficacite_M, ecart_genre
    )
    
    # Bouton de téléchargement
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        st.download_button(
            label="📥 TÉLÉCHARGER LE RAPPORT (PDF)",
            data=rapport_pdf,
            file_name=f"rapport_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    Projet Machine Learning - Polytechnique de Sousse | 2025-2026
</div>
""", unsafe_allow_html=True)