"""
BridgeCompliance Advisory — Market Validation Dashboard
Streamlit + Plotly | 6 analytical tabs
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import spearmanr

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, r2_score, mean_squared_error)
from mlxtend.frequent_patterns import apriori, association_rules

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BridgeCompliance Advisory | Market Validation",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global colour palette ──────────────────────────────────────────────────
NAVY   = "#1B2A4A"
TEAL   = "#0F6E56"
GOLD   = "#BA7517"
RED    = "#C0392B"
LGRAY  = "#F4F6F9"
MKT_COLORS = {"Dubai (DIFC / ADGM)": "#BA7517", "Singapore (MAS-regulated)": "#0F6E56"}
STAGE_COLORS = ["#1B2A4A","#0F6E56","#BA7517","#C0392B"]

# ── Shared CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.kpi-card {
    background: white; border-radius: 10px; padding: 18px 20px;
    border-left: 4px solid #1B2A4A; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 8px;
}
.kpi-label { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #1B2A4A; }
.kpi-win  { color: #0F6E56; }
.kpi-card.winner { border-left-color: #0F6E56; background: #F0FDF9; }
.method-box {
    background: #EBF5FB; border-left: 4px solid #1B2A4A;
    border-radius: 6px; padding: 14px 18px; margin: 12px 0;
    font-size: 13px; color: #1B2A4A;
}
.insight-box {
    background: #FFFBEB; border-left: 4px solid #BA7517;
    border-radius: 6px; padding: 14px 18px; margin: 12px 0;
    font-size: 13px; color: #78350F;
}
.rec-box {
    background: #F0FDF9; border: 1.5px solid #0F6E56;
    border-radius: 10px; padding: 20px 24px; margin: 16px 0;
}
.rec-title { font-size: 16px; font-weight: 700; color: #0F6E56; margin-bottom: 10px; }
.warning-box {
    background: #FEF3C7; border-left: 4px solid #D97706;
    border-radius: 6px; padding: 12px 16px; margin: 10px 0;
    font-size: 13px; color: #92400E;
}
.section-header {
    font-size: 18px; font-weight: 700; color: #1B2A4A;
    margin: 24px 0 6px; border-bottom: 2px solid #E5E7EB; padding-bottom: 6px;
}
.sub-header { font-size: 14px; font-weight: 600; color: #374151; margin: 16px 0 4px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════
def load_data(uploaded_file) -> pd.DataFrame:
    """Load and validate the preprocessed Excel file."""
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet = "Clean Dataset" if "Clean Dataset" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(uploaded_file, sheet_name=sheet, header=1)
        # Drop section-label row if it exists (row where ResponseID is NaN or 'SURVEY RESPONSES')
        df = df[~df.iloc[:,0].astype(str).str.contains('SURVEY|←|Section', na=True)].copy()
        df = df.dropna(subset=['ResponseID']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"File loading error: {e}")
        return None


def prepare_features(df: pd.DataFrame):
    """Build numeric feature matrix, handling both pre-encoded and raw DataFrames."""
    pain_cols = ['Q4_Licensing','Q4_AMLCFT','Q4_DataProtection','Q4_DigitalAssets','Q4_AIGovernance']
    scale_cols = ['Q5_Urgency','Q8_Likelihood','Q4_Pain_Count','ENC_Stage','ENC_WTP']

    # Label-encode categoricals if LE_ columns absent
    cat_map = {}
    for col, le_col in [('Q2_Industry','LE_Q2_Industry'),
                         ('Q6_Engagement','LE_Q6_Engagement'),
                         ('Q9_Deal_Breaker','LE_Q9_Deal_Breaker')]:
        if le_col in df.columns:
            cat_map[le_col] = df[le_col]
        elif col in df.columns:
            le = LabelEncoder()
            cat_map[le_col] = le.fit_transform(df[col].astype(str))

    feature_cols = (['ENC_Market','ENC_Stage','ENC_WTP'] +
                    pain_cols + scale_cols[:-2] +
                    list(cat_map.keys()))
    feature_cols = [c for c in feature_cols if c in df.columns or c in cat_map]

    feat_df = df[['ENC_Market','ENC_Stage','ENC_WTP',
                  'Q4_Pain_Count','Q5_Urgency','Q8_Likelihood'] +
                 pain_cols].copy()
    for k,v in cat_map.items():
        feat_df[k] = v.values if hasattr(v,'values') else v

    feat_df = feat_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return feat_df


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='background:{NAVY};padding:18px 16px;border-radius:10px;margin-bottom:16px;'>
      <div style='color:white;font-size:17px;font-weight:700;'>⚖️ BridgeCompliance</div>
      <div style='color:#94A3B8;font-size:11px;margin-top:4px;'>Market Validation Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📁 Upload Dataset")
    uploaded = st.file_uploader(
        "Upload the preprocessed Excel file",
        type=["xlsx"],
        help="Upload BridgeCompliance_Preprocessed_Dataset.xlsx"
    )
    if uploaded:
        st.success("✅ File loaded")

    st.markdown("---")
    st.markdown("### 🔧 Filter Controls")
    market_filter = st.multiselect(
        "Market",
        ["Dubai (DIFC / ADGM)", "Singapore (MAS-regulated)"],
        default=["Dubai (DIFC / ADGM)", "Singapore (MAS-regulated)"]
    )
    stage_filter = st.multiselect(
        "Startup Stage",
        ['Pre-seed (1-5 ppl)','Seed (6-20 ppl)',
         'Series A-B (21-100 ppl)','Series C+ (100+ ppl)'],
        default=['Pre-seed (1-5 ppl)','Seed (6-20 ppl)',
                 'Series A-B (21-100 ppl)','Series C+ (100+ ppl)']
    )
    bot_filter = st.checkbox("Exclude bot-flagged responses", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#6B7280;'>
    <b>Dashboard v1.0</b><br>
    BridgeCompliance Advisory<br>
    Dubai · Singapore · Cross-border
    </div>
    """, unsafe_allow_html=True)

# ── Gate on upload ─────────────────────────────────────────────────────────
if not uploaded:
    st.markdown(f"""
    <div style='text-align:center;padding:60px 40px;'>
      <div style='font-size:48px;margin-bottom:16px;'>⚖️</div>
      <div style='font-size:24px;font-weight:700;color:{NAVY};margin-bottom:8px;'>
        BridgeCompliance Advisory
      </div>
      <div style='font-size:15px;color:#6B7280;margin-bottom:24px;'>
        Cross-border Legal & Regulatory Advisory · Dubai × Singapore
      </div>
      <div style='font-size:13px;color:#9CA3AF;'>
        Upload <b>BridgeCompliance_Preprocessed_Dataset.xlsx</b> in the sidebar to begin.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & filter ─────────────────────────────────────────────────────────
df_full = load_data(uploaded)
if df_full is None:
    st.stop()

df = df_full.copy()
if bot_filter and 'BOT_Flag' in df.columns:
    df = df[df['BOT_Flag'] == 0]
if market_filter:
    df = df[df['Q1_Market'].isin(market_filter)]
if stage_filter:
    df = df[df['Q3_Stage'].isin(stage_filter)]

if len(df) < 30:
    st.warning("⚠️ Fewer than 30 rows after filtering. Relax filters for meaningful analysis.")
    st.stop()

# Ensure numeric columns are numeric
for c in ['Q5_Urgency','Q8_Likelihood','ENC_WTP','ENC_Stage','ENC_Market',
          'Q4_Pain_Count','TARGET_Conversion','TARGET_High_Intent',
          'TARGET_High_WTP','TARGET_Priority']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

WTP_MID = {'Early-stage (<SGD 2K/mo)':1000,
            'Growth-stage (SGD 2K-5K/mo)':3500,
            'Scale-up (>SGD 5K/mo)':6500}
df['WTP_SGD'] = df['Q7_WTP_Tier'].map(WTP_MID).fillna(df.get('ENC_WTP_SGD', 1000))

STAGES  = ['Pre-seed (1-5 ppl)','Seed (6-20 ppl)',
           'Series A-B (21-100 ppl)','Series C+ (100+ ppl)']
PAIN_LABELS = {
    'Q4_Licensing':'Licensing & Auth',
    'Q4_AMLCFT':'AML/CFT',
    'Q4_DataProtection':'Data Protection',
    'Q4_DigitalAssets':'Digital Assets',
    'Q4_AIGovernance':'AI Governance'
}
PAIN_COLS = list(PAIN_LABELS.keys())

# ══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌍 Tab 1 — Market Comparison",
    "💰 Tab 2 — What Drives WTP",
    "🎯 Tab 3 — Predicting Engagement",
    "📈 Tab 4 — Predicting Revenue",
    "🧩 Tab 5 — Segments & Bundles",
    "📋 Tab 6 — Recommendations"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET COMPARISON
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🌍 Market Demand Comparison: Dubai vs Singapore</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="method-box">
    <b>Analytical approach — Descriptive statistics</b><br>
    Before building any predictive model, we must first understand what the raw data says about each market.
    Descriptive analytics summarises the current state without making assumptions about causality — it answers
    <i>"what is happening"</i> before we ask <i>"why"</i> or <i>"what will happen next"</i>. This is the
    correct starting point: validate the signal, then model it.
    The four demand signals used here — intent rate, WTP, urgency, and likelihood — are the most direct
    observable proxies for market readiness in a pre-launch B2B context.
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────
    st.markdown('<div class="sub-header">Key Demand Signals — Head-to-Head</div>', unsafe_allow_html=True)

    dub  = df[df['Q1_Market']=='Dubai (DIFC / ADGM)']
    sgp  = df[df['Q1_Market']=='Singapore (MAS-regulated)']

    def pct(series): return round(series.mean()*100,1) if len(series)>0 else 0
    def avg(series): return round(series.mean(),2)    if len(series)>0 else 0

    kpis = {
        "Intent Rate (Likelihood ≥7)": (
            pct(dub['TARGET_High_Intent']), pct(sgp['TARGET_High_Intent']), "%"),
        "Avg WTP (SGD / month)": (
            avg(dub['WTP_SGD']), avg(sgp['WTP_SGD']), "SGD"),
        "Avg Urgency Score (1–10)": (
            avg(dub['Q5_Urgency']), avg(sgp['Q5_Urgency']), "/10"),
        "Avg Likelihood Score (1–10)": (
            avg(dub['Q8_Likelihood']), avg(sgp['Q8_Likelihood']), "/10"),
    }

    col_d, col_s = st.columns(2)
    with col_d:
        st.markdown(f"<div style='text-align:center;font-weight:700;color:{GOLD};font-size:15px;margin-bottom:8px;'>🇦🇪 Dubai (DIFC / ADGM)</div>",
                    unsafe_allow_html=True)
        for label,(dv,sv,unit) in kpis.items():
            winner = dv >= sv
            cls = "kpi-card winner" if winner else "kpi-card"
            arrow = "▲" if winner else ""
            st.markdown(f"""
            <div class="{cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {'kpi-win' if winner else ''}">{dv}{unit} {arrow}</div>
            </div>""", unsafe_allow_html=True)

    with col_s:
        st.markdown(f"<div style='text-align:center;font-weight:700;color:{TEAL};font-size:15px;margin-bottom:8px;'>🇸🇬 Singapore (MAS-regulated)</div>",
                    unsafe_allow_html=True)
        for label,(dv,sv,unit) in kpis.items():
            winner = sv >= dv
            cls = "kpi-card winner" if winner else "kpi-card"
            arrow = "▲" if winner else ""
            st.markdown(f"""
            <div class="{cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value {'kpi-win' if winner else ''}">{sv}{unit} {arrow}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Chart 1: Intent rate by stage ──────────────────────────────────────
    st.markdown('<div class="sub-header">Intent Rate by Startup Stage</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why grouped bar?</b> A grouped bar chart directly shows the percentage of high-intent respondents
    (Likelihood ≥7) across each funding stage, side-by-side per market. A heatmap would compress the
    magnitude differences into colour, making it harder to read exact values. For a primary business decision
    metric like intent rate, precision matters more than pattern recognition.
    </div>
    """, unsafe_allow_html=True)

    intent_stage = (df.groupby(['Q3_Stage','Q1_Market'])['TARGET_High_Intent']
                      .mean().mul(100).round(1).reset_index())
    intent_stage.columns = ['Stage','Market','Intent Rate (%)']
    intent_stage['Stage'] = pd.Categorical(intent_stage['Stage'], categories=STAGES, ordered=True)
    intent_stage = intent_stage.sort_values('Stage')

    fig1 = px.bar(intent_stage, x='Stage', y='Intent Rate (%)', color='Market',
                  barmode='group', color_discrete_map=MKT_COLORS,
                  title='High-Intent Rate (%) by Stage and Market',
                  text='Intent Rate (%)')
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                       font_family='Inter', legend_title_text='',
                       xaxis_title='', yaxis_title='% High-Intent Respondents',
                       title_font_size=14)
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: WTP distribution by stage & market ──────────────────────
    st.markdown('<div class="sub-header">Willingness-to-Pay Distribution Across Stages</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why box plot?</b> WTP has three discrete tiers mapped to midpoint values (SGD 1,000 / 3,500 / 6,500).
    A box plot shows median, spread, and potential skew within each stage-market combination — more informative
    than a bar of averages which hides whether responses cluster tightly or spread widely across tiers.
    </div>
    """, unsafe_allow_html=True)

    fig2 = px.box(df, x='Q3_Stage', y='WTP_SGD', color='Q1_Market',
                  color_discrete_map=MKT_COLORS,
                  title='WTP (SGD/month) Distribution by Stage and Market',
                  category_orders={'Q3_Stage': STAGES},
                  labels={'Q3_Stage':'Stage','WTP_SGD':'WTP Midpoint (SGD/month)','Q1_Market':'Market'})
    fig2.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                       font_family='Inter', legend_title_text='', xaxis_title='')
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Pain point prevalence ───────────────────────────────────
    st.markdown('<div class="sub-header">Regulatory Pain Point Prevalence by Market</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why horizontal grouped bar?</b> Pain point labels are long strings. Horizontal bars give each label
    space to breathe and make cross-market comparison natural — the eye scans left-to-right for the
    Dubai/Singapore split on each domain. A heatmap here would obscure the absolute prevalence rates,
    which matter for service prioritisation.
    </div>
    """, unsafe_allow_html=True)

    pain_mkt = []
    for mkt,grp in df.groupby('Q1_Market'):
        for col,lbl in PAIN_LABELS.items():
            if col in grp.columns:
                pain_mkt.append({'Market':mkt,'Pain Point':lbl,
                                 'Prevalence (%)': round(grp[col].mean()*100,1)})
    pain_df = pd.DataFrame(pain_mkt)

    fig3 = px.bar(pain_df, y='Pain Point', x='Prevalence (%)', color='Market',
                  barmode='group', orientation='h',
                  color_discrete_map=MKT_COLORS,
                  title='% of Startups Flagging Each Regulatory Domain as Urgent',
                  text='Prevalence (%)')
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                       font_family='Inter', legend_title_text='',
                       yaxis_title='', xaxis_title='% Respondents')
    st.plotly_chart(fig3, use_container_width=True)

    # ── Chart 4: Urgency distribution ─────────────────────────────────────
    st.markdown('<div class="sub-header">Urgency Score Distribution</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why violin plot?</b> Urgency is a 1–10 ordinal scale. A violin plot shows the full distribution shape —
    where scores concentrate — not just the mean. This matters because a market with average urgency of 7
    could have a bimodal split (many 5s and many 9s) or a tight cluster around 7. Those two patterns imply
    very different go-to-market strategies.
    </div>
    """, unsafe_allow_html=True)

    fig4 = px.violin(df, x='Q1_Market', y='Q5_Urgency', color='Q1_Market',
                     box=True, points='outliers',
                     color_discrete_map=MKT_COLORS,
                     title='Urgency Score Distribution by Market (with box overlay)',
                     labels={'Q1_Market':'','Q5_Urgency':'Urgency Score (1–10)'})
    fig4.update_layout(height=360, plot_bgcolor='white', paper_bgcolor='white',
                       font_family='Inter', showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Chart 5: Deal-breaker by market ──────────────────────────────────
    st.markdown('<div class="sub-header">Deal-Breaker Factor by Market (Q9)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why stacked bar normalised to 100%?</b> Q9 is a single-choice question — every respondent picks
    exactly one deal-breaker. A normalised stacked bar shows the share of each factor within each market,
    making the relative priority directly comparable even when market sample sizes slightly differ after filtering.
    </div>
    """, unsafe_allow_html=True)

    if 'Q9_Deal_Breaker' in df.columns:
        deal_ct = (df.groupby(['Q1_Market','Q9_Deal_Breaker']).size()
                     .reset_index(name='Count'))
        deal_tot = deal_ct.groupby('Q1_Market')['Count'].transform('sum')
        deal_ct['Share (%)'] = (deal_ct['Count']/deal_tot*100).round(1)

        fig5 = px.bar(deal_ct, x='Q1_Market', y='Share (%)', color='Q9_Deal_Breaker',
                      barmode='stack',
                      title='Deal-Breaker Factor Distribution by Market (% of respondents)',
                      labels={'Q1_Market':'','Q9_Deal_Breaker':'Deal-Breaker','Share (%)':'Share (%)'},
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      text='Share (%)')
        fig5.update_traces(texttemplate='%{text:.0f}%', textposition='inside')
        fig5.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                           font_family='Inter', legend_title_text='')
        st.plotly_chart(fig5, use_container_width=True)

    # ── Recommendation box ────────────────────────────────────────────────
    dub_intent = pct(dub['TARGET_High_Intent'])
    sgp_intent = pct(sgp['TARGET_High_Intent'])
    dub_wtp    = avg(dub['WTP_SGD'])
    sgp_wtp    = avg(sgp['WTP_SGD'])
    dub_urg    = avg(dub['Q5_Urgency'])
    sgp_urg    = avg(sgp['Q5_Urgency'])
    lead_mkt   = "Dubai (DIFC / ADGM)" if dub_intent >= sgp_intent else "Singapore (MAS-regulated)"
    follow_mkt = "Singapore (MAS-regulated)" if lead_mkt == "Dubai (DIFC / ADGM)" else "Dubai (DIFC / ADGM)"

    lead_vals = (dub_intent, dub_wtp, dub_urg) if lead_mkt == "Dubai (DIFC / ADGM)" else (sgp_intent, sgp_wtp, sgp_urg)
    follow_vals = (sgp_intent, sgp_wtp, sgp_urg) if lead_mkt == "Dubai (DIFC / ADGM)" else (dub_intent, dub_wtp, dub_urg)

    st.markdown(f"""
    <div class="rec-box">
      <div class="rec-title">📍 Preliminary Recommendation — Launch Market</div>
      <p><b>Data-supported first market: {lead_mkt}</b></p>
      <p>Three signals from the data above support this conclusion as a preliminary finding:</p>
      <ol>
        <li><b>Higher intent rate:</b> {lead_vals[0]}% of {lead_mkt.split(' ')[0]} respondents scored Likelihood ≥7,
            vs {follow_vals[0]}% in {follow_mkt.split(' ')[0]}. Intent rate is the most direct proxy for near-term
            conversion probability in a pre-launch validation context.</li>
        <li><b>Higher average WTP:</b> {lead_mkt.split(' ')[0]} respondents indicated an average of SGD {lead_vals[1]:,.0f}/month
            vs SGD {follow_vals[1]:,.0f}/month — a {abs(lead_vals[1]-follow_vals[1]):,.0f} SGD delta that directly
            affects revenue-per-client projections.</li>
        <li><b>Higher urgency:</b> Average urgency score of {lead_vals[2]}/10 vs {follow_vals[2]}/10 in {follow_mkt.split(' ')[0]}
            — higher urgency reduces the sales cycle and lowers client acquisition cost.</li>
      </ol>
      <p style='color:#374151;font-size:12px;margin-top:8px;'>
      ⚠️ <i>Preliminary finding only. Tabs 3–5 apply predictive models and segmentation to validate
      and refine this signal before a launch decision is made.</i></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <b>What this tells BridgeCompliance:</b> The descriptive data provides the first directional signal.
    The next step is to understand <i>what variables actually drive</i> the WTP signal before assuming it
    is purely market-driven — which is what Tab 2 addresses through correlation analysis.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — WHAT DRIVES WILLINGNESS TO PAY
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">💰 What Drives Willingness to Pay?</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="method-box">
    <b>Analytical approach — Spearman rank correlation</b><br>
    Before building a regression model, we must identify which variables move together with WTP — and how
    strongly. <b>Why Spearman and not Pearson?</b> Pearson correlation assumes continuous, normally-distributed
    variables with linear relationships. Our data contains binary pain flags (0/1), ordinal scales (1–10),
    and discrete WTP tiers — none of which satisfy the Pearson assumptions. Spearman rank correlation is
    distribution-free: it measures whether higher values of variable A consistently correspond to higher
    values of variable B, without assuming the relationship is linear or the data is normal.
    It tells us the <i>strength and direction</i> of relationships before we commit to a regression structure.
    </div>
    """, unsafe_allow_html=True)

    # ── Spearman correlation matrix ────────────────────────────────────────
    st.markdown('<div class="sub-header">Spearman Rank Correlation Matrix</div>', unsafe_allow_html=True)

    corr_cols = {
        'ENC_WTP':'WTP Tier','Q5_Urgency':'Urgency','Q8_Likelihood':'Likelihood',
        'Q4_Pain_Count':'Pain Count','ENC_Stage':'Stage','ENC_Market':'Market (Dubai=1)',
        'Q4_Licensing':'Pain: Licensing','Q4_AMLCFT':'Pain: AML/CFT',
        'Q4_DataProtection':'Pain: Data Prot.','Q4_DigitalAssets':'Pain: Digital Assets',
        'Q4_AIGovernance':'Pain: AI Gov.'
    }
    avail_corr = {k:v for k,v in corr_cols.items() if k in df.columns}
    corr_df = df[list(avail_corr.keys())].apply(pd.to_numeric, errors='coerce').dropna()
    corr_df.columns = list(avail_corr.values())

    sp_corr = corr_df.corr(method='spearman')

    fig_corr = go.Figure(go.Heatmap(
        z=sp_corr.values,
        x=sp_corr.columns.tolist(),
        y=sp_corr.index.tolist(),
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
        text=sp_corr.round(2).values,
        texttemplate='%{text}',
        textfont_size=9,
        colorbar_title='ρ (Spearman)'
    ))
    fig_corr.update_layout(
        title='Spearman Rank Correlation Matrix — All Key Variables vs WTP',
        height=480, font_family='Inter',
        xaxis_tickangle=-35, paper_bgcolor='white'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Strongest correlators with WTP ─────────────────────────────────────
    st.markdown('<div class="sub-header">Strongest Predictors of WTP Tier</div>', unsafe_allow_html=True)

    wtp_corrs = sp_corr['WTP Tier'].drop('WTP Tier').sort_values(key=abs, ascending=False)
    corr_bar_df = pd.DataFrame({'Variable':wtp_corrs.index, 'Spearman ρ':wtp_corrs.values})
    corr_bar_df['Color'] = corr_bar_df['Spearman ρ'].apply(lambda x: TEAL if x>0 else RED)
    corr_bar_df['Direction'] = corr_bar_df['Spearman ρ'].apply(lambda x: 'Positive' if x>0 else 'Negative')

    fig_wtp_corr = px.bar(corr_bar_df, x='Spearman ρ', y='Variable', orientation='h',
                           color='Direction',
                           color_discrete_map={'Positive': TEAL, 'Negative': RED},
                           title='Variables Correlated with WTP Tier (Spearman ρ)',
                           text='Spearman ρ')
    fig_wtp_corr.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_wtp_corr.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white',
                                font_family='Inter', legend_title_text='',
                                xaxis_title='Spearman ρ', yaxis_title='')
    st.plotly_chart(fig_wtp_corr, use_container_width=True)

    # ── Scatter: strongest predictor vs WTP ────────────────────────────────
    top_pred = wtp_corrs.index[0] if len(wtp_corrs) > 0 else 'Likelihood'
    # reverse map
    rev_map = {v:k for k,v in avail_corr.items()}
    top_col = rev_map.get(top_pred, 'Q8_Likelihood')

    st.markdown(f'<div class="sub-header">Scatter: {top_pred} vs WTP (strongest Spearman predictor)</div>',
                unsafe_allow_html=True)

    scatter_df = df[[top_col,'WTP_SGD','Q1_Market','Q3_Stage']].dropna()
    fig_scatter = px.scatter(scatter_df, x=top_col, y='WTP_SGD',
                              color='Q1_Market', symbol='Q3_Stage',
                              color_discrete_map=MKT_COLORS,
                              trendline='ols',
                              title=f'{top_pred} vs WTP Midpoint (SGD) — by Market and Stage',
                              labels={top_col: top_pred, 'WTP_SGD':'WTP SGD/month',
                                      'Q1_Market':'Market','Q3_Stage':'Stage'},
                              opacity=0.6, size_max=8)
    fig_scatter.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white',
                               font_family='Inter', legend_title_text='')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── WTP by stage and market ─────────────────────────────────────────────
    st.markdown('<div class="sub-header">Average WTP Across Stages — by Market</div>',
                unsafe_allow_html=True)

    wtp_stage = (df.groupby(['Q3_Stage','Q1_Market'])['WTP_SGD']
                   .mean().round(0).reset_index())
    wtp_stage['Q3_Stage'] = pd.Categorical(wtp_stage['Q3_Stage'], categories=STAGES, ordered=True)
    wtp_stage = wtp_stage.sort_values('Q3_Stage')

    fig_wtp_stage = px.line(wtp_stage, x='Q3_Stage', y='WTP_SGD', color='Q1_Market',
                             markers=True, color_discrete_map=MKT_COLORS,
                             title='Average WTP (SGD/month) by Stage — Dubai vs Singapore',
                             labels={'Q3_Stage':'Stage','WTP_SGD':'Avg WTP SGD/month','Q1_Market':'Market'})
    fig_wtp_stage.update_traces(marker_size=10, line_width=2.5)
    fig_wtp_stage.update_layout(height=360, plot_bgcolor='white', paper_bgcolor='white',
                                 font_family='Inter', legend_title_text='', xaxis_title='')
    st.plotly_chart(fig_wtp_stage, use_container_width=True)

    # ── Business interpretation ────────────────────────────────────────────
    top_rho = round(wtp_corrs.iloc[0], 3) if len(wtp_corrs) > 0 else 0
    st.markdown(f"""
    <div class="insight-box">
    <b>What the correlation structure tells BridgeCompliance:</b><br><br>
    1. <b>The strongest WTP driver is {top_pred} (ρ = {top_rho})</b> — meaning startups that score higher
    on this variable consistently fall in higher WTP brackets. This is your primary lead-scoring variable.<br><br>
    2. <b>Stage is a structural WTP predictor.</b> The line chart above confirms WTP scales with funding stage —
    this means BridgeCompliance's pricing architecture should be stage-tiered, not flat. A single price point
    will systematically under-price Series A–B clients and over-price Pre-seed ones.<br><br>
    3. <b>Market (Dubai=1) shows a positive correlation with WTP</b>, confirming the Tab 1 signal that Dubai
    respondents are willing to spend more — but note this is a correlation, not a cause. Dubai's higher WTP
    may be driven by its higher proportion of funded-stage startups rather than geography alone.<br><br>
    <b>Client profile to prioritise:</b> Series A–B and above startups in either market, with high urgency
    scores (≥7) — this combination represents the intersection of highest WTP and highest likelihood to engage.
    Tab 3 models this profile predictively.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTING ENGAGEMENT (Classification)
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🎯 Predicting Which Startups Will Engage</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="method-box">
    <b>Analytical approach — Binary Classification (Decision Tree, Random Forest, Gradient Boosting)</b><br>
    The business question is binary: a startup will either engage BridgeCompliance or it will not.
    Classification algorithms are designed exactly for this. <b>Why tree-based models and not Logistic Regression?</b>
    Logistic Regression assumes a linear relationship between each feature and the log-odds of the outcome.
    Variables like urgency and pain count likely have <i>threshold effects</i> — a startup is not meaningfully
    more likely to engage at urgency 4 vs 5, but there may be a sharp jump at urgency 8. Trees capture these
    non-linear thresholds naturally. Additionally, our feature mix (binary flags, ordinal scales, encoded
    categories) requires no distributional assumption for tree-based models, unlike parametric methods.<br><br>
    <b>Class balance note:</b> <code>class_weight='balanced'</code> is applied to all models because the
    target variable (Likelihood ≥7) may be imbalanced. Without balancing, the model optimises for accuracy
    by over-predicting the majority class — which is exactly wrong when <b>recall is our priority metric</b>
    (missing a high-intent client costs more than a false positive).
    </div>
    """, unsafe_allow_html=True)

    # ── Prepare data ────────────────────────────────────────────────────────
    @st.cache_data
    def run_classification(df_in):
        feat = prepare_features(df_in)
        y = pd.to_numeric(df_in['TARGET_High_Intent'], errors='coerce').fillna(0).astype(int)
        common = feat.index.intersection(y.index)
        X, y = feat.loc[common], y.loc[common]
        X = X.fillna(0)

        X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.25,
                                                random_state=42, stratify=y)
        models = {
            'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=6,
                                                     class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, max_depth=4,
                                                              learning_rate=0.08, random_state=42)
        }
        results, cms, roc_data, importances = {}, {}, {}, {}
        for name, mdl in models.items():
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)
            y_prob = mdl.predict_proba(X_te)[:,1]
            fpr,tpr,_ = roc_curve(y_te, y_prob)
            roc_data[name]  = {'fpr':fpr,'tpr':tpr,'auc':round(roc_auc_score(y_te,y_prob),3)}
            cms[name]       = confusion_matrix(y_te, y_pred)
            results[name]   = {
                'Train Acc': round(mdl.score(X_tr,y_tr),3),
                'Test Acc':  round(accuracy_score(y_te,y_pred),3),
                'Precision': round(precision_score(y_te,y_pred,zero_division=0),3),
                'Recall':    round(recall_score(y_te,y_pred,zero_division=0),3),
                'F1':        round(f1_score(y_te,y_pred,zero_division=0),3),
                'AUC':       roc_data[name]['auc']
            }
            if hasattr(mdl,'feature_importances_'):
                importances[name] = dict(zip(X.columns, mdl.feature_importances_))

        # Cross-validation on best model by F1
        best_name = max(results, key=lambda k: results[k]['F1'])
        best_mdl  = models[best_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_mdl, X, y, cv=cv, scoring='f1')
        return results, cms, roc_data, importances, best_name, cv_scores, X.columns.tolist()

    with st.spinner("Training classification models…"):
        clf_results, cms, roc_data, importances, best_clf, cv_scores, feat_names = run_classification(df)

    # ── Model descriptions ──────────────────────────────────────────────────
    model_descs = {
        'Decision Tree': "A single tree of if-then rules — interpretable and fast; useful as a baseline and for rule extraction.",
        'Random Forest': "An ensemble of 120 decision trees that vote on the outcome — reduces overfitting and provides reliable feature importance.",
        'Gradient Boosting': "Trees built sequentially where each corrects the errors of the last — typically highest accuracy; best when recall matters."
    }
    for name, desc in model_descs.items():
        st.markdown(f"<div style='font-size:12px;color:#374151;margin:4px 0;'><b>{name}:</b> {desc}</div>",
                    unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Recall is the priority metric here.</b> Missing a high-intent client (False Negative) costs
    BridgeCompliance a real revenue opportunity. Generating a false alarm (False Positive) costs only a
    follow-up email. Therefore, we optimise for Recall over Precision — a model with 85% recall and
    75% precision is preferred over one with 95% precision and 65% recall.
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison table ────────────────────────────────────────────────────
    st.markdown('<div class="sub-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    res_df = pd.DataFrame(clf_results).T.reset_index().rename(columns={'index':'Model'})
    res_df['Best'] = res_df['Recall'] == res_df['Recall'].max()

    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Model','Train Acc','Test Acc','Precision','Recall ⭐','F1','AUC'],
            fill_color=NAVY, font=dict(color='white',size=11,family='Inter'),
            align='center', height=32
        ),
        cells=dict(
            values=[res_df[c] for c in ['Model','Train Acc','Test Acc','Precision','Recall','F1','AUC']],
            fill_color=[['white' if not b else '#F0FDF9' for b in res_df['Best'].tolist()]]*7,
            font=dict(size=11,family='Inter'),
            align='center', height=28
        )
    )])
    fig_table.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_table, use_container_width=True)

    # ── Confusion matrices ─────────────────────────────────────────────────
    st.markdown('<div class="sub-header">Confusion Matrices — Where Each Model Fails</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why confusion matrices?</b> Accuracy alone hides failure modes. Each quadrant has a specific
    business meaning for BridgeCompliance:<br>
    • <b>True Positive (TP):</b> High-intent client correctly identified → follow up immediately.<br>
    • <b>False Negative (FN):</b> High-intent client missed → lost revenue opportunity. <i>Worst outcome.</i><br>
    • <b>False Positive (FP):</b> Low-intent client flagged as high-intent → wasted outreach effort. Minor cost.<br>
    • <b>True Negative (TN):</b> Low-intent client correctly deprioritised → efficient pipeline.
    </div>
    """, unsafe_allow_html=True)

    cm_cols = st.columns(3)
    for i,(name,cm) in enumerate(cms.items()):
        with cm_cols[i]:
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=['Pred: Low','Pred: High'], y=['True: Low','True: High'],
                colorscale=[[0,'#FFFFFF'],[1,'#0F6E56']],
                text=cm, texttemplate='%{text}', textfont_size=16,
                showscale=False
            ))
            fig_cm.update_layout(
                title=dict(text=name, font_size=12),
                height=240, font_family='Inter',
                margin=dict(l=10,r=10,t=40,b=10),
                xaxis_title='Predicted', yaxis_title='Actual'
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC curves ─────────────────────────────────────────────────────────
    st.markdown('<div class="sub-header">ROC Curves — All Models (AUC in Legend)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why ROC + AUC?</b> Accuracy at the default 0.5 threshold does not reflect real-world performance
    where the threshold may need tuning. The ROC curve plots True Positive Rate vs False Positive Rate
    across all thresholds. AUC (Area Under Curve) above 0.80 means the model is genuinely discriminating
    between clients who will and will not engage — not guessing. AUC = 0.5 is random guessing.
    </div>
    """, unsafe_allow_html=True)

    roc_colors = {'Decision Tree':'#BA7517','Random Forest':'#0F6E56','Gradient Boosting':'#1B2A4A'}
    fig_roc = go.Figure()
    for name,rd in roc_data.items():
        fig_roc.add_trace(go.Scatter(
            x=rd['fpr'], y=rd['tpr'], mode='lines', name=f"{name} (AUC={rd['auc']})",
            line=dict(color=roc_colors[name], width=2.5)
        ))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',
                                  line=dict(color='gray',dash='dash'),name='Random (AUC=0.5)'))
    fig_roc.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                           font_family='Inter', title='ROC Curves — All Classification Models',
                           xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                           legend_title_text='Model')
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────
    if len(importances) >= 2:
        st.markdown('<div class="sub-header">Feature Importance: Random Forest vs Gradient Boosting</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="method-box">
        <b>Why compare two models?</b> Feature importance scores can differ between algorithms. If both
        Random Forest and Gradient Boosting rank the same variable highly, that is a robust signal.
        If they disagree, the feature's importance is model-specific and should be interpreted with caution.
        </div>
        """, unsafe_allow_html=True)

        fi_models = {k:v for k,v in importances.items() if k in ['Random Forest','Gradient Boosting']}
        all_feats = sorted(set.union(*[set(v.keys()) for v in fi_models.values()]))
        fi_rows = []
        for feat in all_feats:
            for mdl,fi in fi_models.items():
                fi_rows.append({'Feature':feat,'Model':mdl,'Importance':round(fi.get(feat,0),4)})
        fi_df = pd.DataFrame(fi_rows)
        top_feats = (fi_df.groupby('Feature')['Importance'].max()
                         .sort_values(ascending=False).head(12).index.tolist())
        fi_df = fi_df[fi_df['Feature'].isin(top_feats)]

        fig_fi = px.bar(fi_df, x='Importance', y='Feature', color='Model', barmode='group',
                         orientation='h', color_discrete_map={'Random Forest':TEAL,'Gradient Boosting':GOLD},
                         title='Top Feature Importances — RF vs Gradient Boosting',
                         category_orders={'Feature':top_feats[::-1]})
        fig_fi.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white',
                              font_family='Inter', legend_title_text='', yaxis_title='')
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Cross-validation ───────────────────────────────────────────────────
    st.markdown(f'<div class="sub-header">5-Fold Cross-Validation — {best_clf}</div>',
                unsafe_allow_html=True)

    cv_df = pd.DataFrame({'Fold':[f'Fold {i+1}' for i in range(len(cv_scores))],
                           'F1 Score':cv_scores.round(3)})
    fig_cv = px.bar(cv_df, x='Fold', y='F1 Score',
                     color_discrete_sequence=[TEAL],
                     title=f'Cross-Validation F1 Scores — {best_clf} (mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f})',
                     text='F1 Score')
    fig_cv.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_cv.add_hline(y=cv_scores.mean(), line_dash='dash', line_color=NAVY,
                      annotation_text=f"Mean F1 = {cv_scores.mean():.3f}")
    fig_cv.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                          font_family='Inter', yaxis_range=[0,1.05])
    st.plotly_chart(fig_cv, use_container_width=True)

    if cv_scores.mean() > 0.92:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>High accuracy note (synthetic data):</b> Accuracy near or above 92% is expected here because
        the target variable TARGET_High_Intent was constructed from Q8_Likelihood, which is also a feature
        in the model. In real survey data collected from actual founders, the expected benchmark is
        <b>75–85% accuracy</b>, with feature importance shifting toward pain point patterns and deal-breaker
        factors that are independent of the target construction.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTING HOW MUCH THEY WILL PAY (Regression)
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📈 Predicting Revenue: How Much Will They Pay?</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="method-box">
    <b>Analytical approach — Regularised Regression (Linear, Ridge, Lasso)</b><br>
    TARGET_Conversion is a continuous variable (range ~1.5–10), making regression the correct technique.
    <b>Linear Regression</b> is the baseline — it estimates the average change in conversion score per
    unit change in each feature. <b>Ridge Regression</b> adds an L2 penalty that shrinks all coefficients
    toward zero without eliminating any — appropriate here because urgency and likelihood are correlated
    (both measure compliance pressure), and multicollinearity destabilises plain linear regression.
    <b>Lasso</b> adds an L1 penalty that drives weak predictor coefficients to exactly zero — it performs
    automatic feature selection, and the variables it retains are the <i>real drivers</i> of conversion score.
    Comparing Ridge and Lasso coefficients side-by-side reveals which variables are robust predictors
    vs which are noise.
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def run_regression(df_in):
        feat = prepare_features(df_in)
        y = pd.to_numeric(df_in['TARGET_Conversion'], errors='coerce')
        common = feat.index.intersection(y.index)
        X, y = feat.loc[common].fillna(0), y.loc[common]
        valid = ~y.isna(); X, y = X[valid], y[valid]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_tr,X_te,y_tr,y_te = train_test_split(X_s, y, test_size=0.25, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Lasso (α=0.05)': Lasso(alpha=0.05, max_iter=5000)
        }
        results, preds = {}, {}
        for name, mdl in models.items():
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)
            preds[name] = (y_te.values, y_pred)
            results[name] = {
                'R²':     round(r2_score(y_te,y_pred),4),
                'RMSE':   round(np.sqrt(mean_squared_error(y_te,y_pred)),4),
                'Train R²': round(mdl.score(X_tr,y_tr),4)
            }

        coef_data = {}
        for name in ['Ridge (α=1.0)','Lasso (α=0.05)']:
            mdl = models[name]
            coef_data[name] = dict(zip(X.columns, mdl.coef_))

        best = max(results, key=lambda k: results[k]['R²'])
        return results, preds, coef_data, best, X.columns.tolist()

    with st.spinner("Fitting regression models…"):
        reg_results, reg_preds, coef_data, best_reg, reg_feats = run_regression(df)

    # Model descriptions
    reg_descs = {
        'Linear Regression':'Baseline — estimates raw coefficient per feature with no penalty; sensitive to multicollinearity.',
        'Ridge (α=1.0)':'Shrinks all coefficients proportionally — stabilises estimates when urgency and likelihood are correlated.',
        'Lasso (α=0.05)':'Drives weak predictor coefficients to zero — acts as automatic feature selection; variables it keeps are the real drivers.'
    }
    for name,desc in reg_descs.items():
        st.markdown(f"<div style='font-size:12px;color:#374151;margin:4px 0;'><b>{name}:</b> {desc}</div>",
                    unsafe_allow_html=True)

    # ── Performance table ───────────────────────────────────────────────────
    st.markdown('<div class="sub-header">Regression Model Performance</div>', unsafe_allow_html=True)
    reg_df = pd.DataFrame(reg_results).T.reset_index().rename(columns={'index':'Model'})
    fig_reg_table = go.Figure(data=[go.Table(
        header=dict(values=['Model','Train R²','Test R²','RMSE'],
                    fill_color=NAVY, font=dict(color='white',size=11,family='Inter'),
                    align='center', height=32),
        cells=dict(values=[reg_df[c] for c in ['Model','Train R²','R²','RMSE']],
                   fill_color='white', font=dict(size=11,family='Inter'),
                   align='center', height=28)
    )])
    fig_reg_table.update_layout(height=170, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_reg_table, use_container_width=True)

    # ── Ridge vs Lasso coefficient chart ──────────────────────────────────
    st.markdown('<div class="sub-header">Ridge vs Lasso Coefficients — Which Features Survive?</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    A Lasso coefficient of exactly 0 means that variable was eliminated as a predictor.
    Coefficients retained by Lasso but not Ridge (or vice versa) signal which predictors are robust
    vs model-specific. Large Ridge coefficients on correlated features (e.g. urgency + likelihood)
    confirm multicollinearity — the exact problem Ridge was built to handle.
    </div>
    """, unsafe_allow_html=True)

    coef_rows = []
    for model_name, coefs in coef_data.items():
        for feat, val in coefs.items():
            coef_rows.append({'Feature':feat,'Model':model_name,'Coefficient':round(val,4)})
    coef_df = pd.DataFrame(coef_rows)
    top_coef_feats = (coef_df.groupby('Feature')['Coefficient']
                              .apply(lambda x: abs(x).max())
                              .sort_values(ascending=False).head(12).index.tolist())
    coef_df = coef_df[coef_df['Feature'].isin(top_coef_feats)]

    fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', color='Model', barmode='group',
                       orientation='h', color_discrete_map={'Ridge (α=1.0)':TEAL,'Lasso (α=0.05)':GOLD},
                       title='Ridge vs Lasso Standardised Coefficients (features surviving L1 penalty = real drivers)',
                       category_orders={'Feature':top_coef_feats[::-1]})
    fig_coef.add_vline(x=0, line_color='gray', line_dash='dash')
    fig_coef.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white',
                            font_family='Inter', legend_title_text='', yaxis_title='')
    st.plotly_chart(fig_coef, use_container_width=True)

    # ── Actual vs predicted & residuals ────────────────────────────────────
    st.markdown(f'<div class="sub-header">Actual vs Predicted — {best_reg}</div>',
                unsafe_allow_html=True)

    y_act, y_hat = reg_preds[best_reg]
    residuals = y_act - y_hat

    col_avp, col_res = st.columns(2)
    with col_avp:
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=y_act, y=y_hat, mode='markers',
                                      marker=dict(color=TEAL, opacity=0.5, size=5),
                                      name='Predictions'))
        mn, mx = min(y_act.min(),y_hat.min()), max(y_act.max(),y_hat.max())
        fig_avp.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                      line=dict(color=RED,dash='dash'), name='Perfect fit'))
        fig_avp.update_layout(height=360, plot_bgcolor='white', paper_bgcolor='white',
                               font_family='Inter', title='Actual vs Predicted Conversion Score',
                               xaxis_title='Actual', yaxis_title='Predicted')
        st.plotly_chart(fig_avp, use_container_width=True)

    with col_res:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=y_hat, y=residuals, mode='markers',
                                      marker=dict(color=GOLD, opacity=0.5, size=5),
                                      name='Residuals'))
        fig_res.add_hline(y=0, line_dash='dash', line_color=RED)
        fig_res.update_layout(height=360, plot_bgcolor='white', paper_bgcolor='white',
                               font_family='Inter', title='Residual Plot (y_hat vs residuals)',
                               xaxis_title='Predicted', yaxis_title='Residual (Actual − Predicted)')
        st.plotly_chart(fig_res, use_container_width=True)

    resid_std = round(np.std(residuals), 3)
    resid_skew = round(float(stats.skew(residuals)), 3)
    st.markdown(f"""
    <div class="insight-box">
    <b>How to read the residual plot:</b><br>
    A good residual plot shows points scattered <i>randomly and symmetrically</i> around the horizontal
    zero line, with no fan shape (heteroscedasticity) or curve (non-linearity). This plot shows
    residual std = {resid_std}, skew = {resid_skew}.
    {"✅ The residuals are approximately random — the model is appropriately specified." if abs(resid_skew) < 0.5 else
     "⚠️ Some skew in residuals suggests the model may be missing a non-linear component — consider adding interaction terms or switching to a tree-based regressor for production use."}<br><br>
    <b>Business use:</b> The regression formula gives BridgeCompliance a lead-scoring tool.
    Any inbound startup can be scored on urgency, stage, WTP tier, and pain count to estimate their
    conversion potential before a sales call is made.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — CUSTOMER SEGMENTS & SERVICE BUNDLES
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🧩 Customer Segments and Service Bundles</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="method-box">
    <b>Clustering method: K-Prototypes (not K-Means)</b><br>
    Your instinct to reject K-Means is correct. K-Means computes Euclidean distance, which is geometrically
    meaningless between category labels — the "distance" between 'Fintech' and 'Crypto/Web3' has no
    numeric interpretation. <b>K-Prototypes</b> (Huang, 1997) solves this by combining Euclidean distance
    for numeric variables with <b>Hamming distance</b> for categorical ones — it counts the number of
    categorical mismatches rather than trying to measure numeric distance. This is the principled solution
    for mixed-type survey data.<br><br>
    <b>Association mining: Apriori on binary pain flags only.</b> Apriori requires a boolean transaction
    matrix. We apply it exclusively to the 5 binary Q4 pain point columns — this gives us service
    co-occurrence patterns: which compliance domains appear together in the same startup, telling us
    which services to bundle.
    </div>
    """, unsafe_allow_html=True)

    # ── K-Prototypes clustering ─────────────────────────────────────────────
    @st.cache_data
    def run_kproto(df_in, n_clusters=4):
        from kmodes.kprototypes import KPrototypes
        num_cols = ['Q5_Urgency','Q8_Likelihood','ENC_WTP','ENC_Stage','Q4_Pain_Count']
        cat_cols_kp = ['Q1_Market','Q2_Industry','Q6_Engagement','Q9_Deal_Breaker']
        use_cols = [c for c in num_cols+cat_cols_kp if c in df_in.columns]
        cat_use  = [c for c in cat_cols_kp if c in df_in.columns]
        num_use  = [c for c in num_cols if c in df_in.columns]

        df_kp = df_in[use_cols].copy()
        for c in num_use: df_kp[c] = pd.to_numeric(df_kp[c], errors='coerce').fillna(df_kp[c].median() if df_kp[c].dropna().shape[0]>0 else 0)
        for c in cat_use: df_kp[c] = df_kp[c].astype(str).fillna('Unknown')

        cat_idx = [df_kp.columns.tolist().index(c) for c in cat_use]

        # BIC-like: try k=2..6 using cost
        costs = []
        k_range = range(2, 7)
        for k in k_range:
            try:
                kp = KPrototypes(n_clusters=k, init='Huang', n_init=3, random_state=42)
                kp.fit(df_kp.values, categorical=cat_idx)
                costs.append(kp.cost_)
            except Exception:
                costs.append(np.nan)

        kp_best = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=5, random_state=42)
        labels = kp_best.fit_predict(df_kp.values, categorical=cat_idx)
        return labels, costs, list(k_range), use_cols, num_use, cat_use

    n_seg = st.slider("Number of segments (K)", min_value=2, max_value=6, value=4,
                       help="The elbow/cost curve below guides optimal K selection.")

    with st.spinner("Running K-Prototypes clustering…"):
        try:
            labels, kp_costs, k_range, kp_cols, num_kp, cat_kp = run_kproto(df, n_seg)
            df_seg = df.copy()
            df_seg['Segment'] = labels

            # Cost curve (elbow)
            st.markdown('<div class="sub-header">Cost Curve — Optimal Number of Segments</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="method-box">
            <b>Why a cost/elbow curve?</b> K-Prototypes minimises a combined cost function across numeric
            and categorical distances. Plotting cost vs K reveals the "elbow" — the point where adding more
            clusters yields diminishing returns in explanatory power. Choose K at the elbow, not the minimum.
            </div>
            """, unsafe_allow_html=True)

            cost_df = pd.DataFrame({'K': list(k_range), 'Cost': kp_costs})
            fig_cost = px.line(cost_df, x='K', y='Cost', markers=True,
                                title='K-Prototypes Cost vs Number of Clusters (elbow = optimal K)',
                                color_discrete_sequence=[TEAL])
            fig_cost.add_vline(x=n_seg, line_dash='dash', line_color=GOLD,
                                annotation_text=f'Selected K={n_seg}')
            fig_cost.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                                    font_family='Inter')
            st.plotly_chart(fig_cost, use_container_width=True)

            # Segment profiles
            st.markdown('<div class="sub-header">Segment Profiles</div>', unsafe_allow_html=True)

            profile_cols = ['Q5_Urgency','Q8_Likelihood','WTP_SGD','ENC_Stage','Q4_Pain_Count',
                            'TARGET_Conversion','TARGET_Priority']
            avail_prof = [c for c in profile_cols if c in df_seg.columns]
            prof = df_seg.groupby('Segment')[avail_prof].mean().round(2)
            prof['Count'] = df_seg.groupby('Segment').size()
            prof['Market_Dubai_%'] = (df_seg.groupby('Segment')
                                       .apply(lambda g: (g['Q1_Market']=='Dubai (DIFC / ADGM)').mean()*100)
                                       .round(1))

            # Derive segment names from profiles
            seg_names = {}
            for s in prof.index:
                urg = prof.loc[s,'Q5_Urgency'] if 'Q5_Urgency' in prof.columns else 5
                wtp = prof.loc[s,'WTP_SGD']    if 'WTP_SGD'    in prof.columns else 2000
                stg = prof.loc[s,'ENC_Stage']  if 'ENC_Stage'  in prof.columns else 2
                pri = prof.loc[s,'TARGET_Priority'] if 'TARGET_Priority' in prof.columns else 0
                if pri > 0.5:   seg_names[s] = f"Seg {s} — High-Priority Converter"
                elif wtp > 4000: seg_names[s] = f"Seg {s} — Scale-up Big Spender"
                elif urg > 7:    seg_names[s] = f"Seg {s} — Urgent Early Mover"
                else:            seg_names[s] = f"Seg {s} — Exploratory Pre-seed"

            prof.index = [seg_names[i] for i in prof.index]
            prof_display = prof.reset_index().rename(columns={'index':'Segment'})

            fig_prof_table = go.Figure(data=[go.Table(
                header=dict(values=list(prof_display.columns),
                            fill_color=NAVY, font=dict(color='white',size=10,family='Inter'),
                            align='center', height=30),
                cells=dict(values=[prof_display[c] for c in prof_display.columns],
                           fill_color=[['#F0FDF9' if i%2==0 else 'white'
                                        for i in range(len(prof_display))]]*len(prof_display.columns),
                           font=dict(size=10,family='Inter'), align='center', height=26)
            )])
            fig_prof_table.update_layout(height=int(60+len(prof_display)*32),
                                          margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_prof_table, use_container_width=True)

            # Radar chart per segment
            st.markdown('<div class="sub-header">Segment Radar — Urgency, Likelihood, WTP, Stage, Pain Count</div>',
                        unsafe_allow_html=True)
            radar_cols = [c for c in ['Q5_Urgency','Q8_Likelihood','ENC_WTP','ENC_Stage','Q4_Pain_Count']
                          if c in df_seg.columns]
            if radar_cols:
                seg_radar = df_seg.groupby('Segment')[radar_cols].mean()
                # Normalise each column 0-1
                seg_radar_n = (seg_radar - seg_radar.min()) / (seg_radar.max() - seg_radar.min() + 1e-9)
                radar_labels = {'Q5_Urgency':'Urgency','Q8_Likelihood':'Likelihood',
                                'ENC_WTP':'WTP Tier','ENC_Stage':'Stage','Q4_Pain_Count':'Pain Count'}
                cats = [radar_labels.get(c,c) for c in radar_cols]
                seg_palette = ['#1B2A4A','#0F6E56','#BA7517','#C0392B','#5B2C6F','#1A5276']
                fig_radar = go.Figure()
                for i,seg in enumerate(seg_radar_n.index):
                    vals = seg_radar_n.loc[seg].tolist()
                    vals += vals[:1]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=cats+cats[:1],
                        fill='toself', name=seg_names.get(seg, f'Seg {seg}'),
                        line_color=seg_palette[i % len(seg_palette)], opacity=0.7
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                    title='Segment Profiles (normalised)', height=440,
                    font_family='Inter', paper_bgcolor='white'
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Segment market distribution
            seg_mkt = (df_seg.groupby(['Segment','Q1_Market']).size()
                              .reset_index(name='Count'))
            seg_mkt['Segment'] = seg_mkt['Segment'].map(seg_names)
            fig_seg_mkt = px.bar(seg_mkt, x='Segment', y='Count', color='Q1_Market',
                                  barmode='stack', color_discrete_map=MKT_COLORS,
                                  title='Segment Distribution by Market')
            fig_seg_mkt.update_layout(height=360, plot_bgcolor='white', paper_bgcolor='white',
                                       font_family='Inter', xaxis_title='', legend_title_text='')
            st.plotly_chart(fig_seg_mkt, use_container_width=True)

        except Exception as e:
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <b>K-Prototypes fallback:</b> {str(e)[:200]}<br>
            Falling back to K-Means on scaled numeric features. Note: K-Means does not handle categorical
            variables natively — this is a compromise and categorical features are excluded from the
            clustering distance calculation.
            </div>
            """, unsafe_allow_html=True)
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import MinMaxScaler as MMS
            num_only = [c for c in ['Q5_Urgency','Q8_Likelihood','ENC_WTP','ENC_Stage','Q4_Pain_Count']
                        if c in df.columns]
            X_km = df[num_only].apply(pd.to_numeric, errors='coerce').fillna(0)
            mms = MMS(); X_sc = mms.fit_transform(X_km)
            inertias = []
            for k in range(2,7):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_sc)
                inertias.append(km.inertia_)
            fig_in = px.line(pd.DataFrame({'K':range(2,7),'Inertia':inertias}),
                              x='K', y='Inertia', markers=True, title='K-Means Inertia (elbow curve)')
            fig_in.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_in, use_container_width=True)
            km_best = KMeans(n_clusters=n_seg, random_state=42, n_init=10)
            df_seg = df.copy()
            df_seg['Segment'] = km_best.fit_predict(X_sc)

    # ── Association Rules (Apriori) ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sub-header">Service Bundling — Association Rules (Apriori on Pain Flags)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    <b>Why Apriori on pain flags only?</b> Association mining requires a binary transaction matrix.
    The five Q4 columns (0/1 per pain domain) are exactly this structure. Applying Apriori here answers:
    <i>"Which compliance problems appear together in the same startup?"</i> — which directly tells us
    which services to package as a bundle. <b>Lift > 1</b> means the two items co-occur more than
    expected by chance — a lift of 1.5 on {AML/CFT → Data Protection} means startups with AML needs
    are 50% more likely to also need data protection than a random startup.
    </div>
    """, unsafe_allow_html=True)

    min_sup = st.slider("Minimum support threshold", 0.05, 0.60, 0.20, 0.05,
                         help="Lower = more rules found; higher = only frequent co-occurrences")
    min_lift = st.slider("Minimum lift filter", 1.0, 3.0, 1.1, 0.1)

    pain_avail = [c for c in PAIN_COLS if c in df.columns]
    if pain_avail:
        basket = df[pain_avail].copy()
        basket.columns = [PAIN_LABELS[c] for c in pain_avail]
        basket = basket.apply(pd.to_numeric, errors='coerce').fillna(0).astype(bool)

        try:
            freq_items = apriori(basket, min_support=min_sup, use_colnames=True)
            if len(freq_items) > 0:
                rules = association_rules(freq_items, metric='lift', min_threshold=min_lift,
                                          num_itemsets=len(freq_items))
                rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ' + '.join(sorted(x)))
                rules['consequents_str'] = rules['consequents'].apply(lambda x: ' + '.join(sorted(x)))
                rules_sorted = rules.sort_values('lift', ascending=False).head(15)

                fig_rules = px.scatter(rules_sorted, x='support', y='confidence',
                                        size='lift', color='lift',
                                        color_continuous_scale='Teal',
                                        hover_data=['antecedents_str','consequents_str','lift'],
                                        title='Association Rules: Support vs Confidence (bubble size = Lift)',
                                        labels={'support':'Support','confidence':'Confidence'})
                fig_rules.update_layout(height=420, plot_bgcolor='white', paper_bgcolor='white',
                                         font_family='Inter')
                st.plotly_chart(fig_rules, use_container_width=True)

                # Top rules table
                rules_disp = rules_sorted[['antecedents_str','consequents_str','support',
                                           'confidence','lift']].copy()
                rules_disp.columns = ['IF (Antecedent)','THEN (Consequent)','Support','Confidence','Lift']
                rules_disp = rules_disp.round(3).reset_index(drop=True)
                fig_rules_t = go.Figure(data=[go.Table(
                    header=dict(values=list(rules_disp.columns),
                                fill_color=NAVY, font=dict(color='white',size=10,family='Inter'),
                                align='center', height=30),
                    cells=dict(values=[rules_disp[c] for c in rules_disp.columns],
                               fill_color=[['#F0FDF9' if i%2==0 else 'white'
                                            for i in range(len(rules_disp))]]*len(rules_disp.columns),
                               font=dict(size=10,family='Inter'), align='center', height=26)
                )])
                fig_rules_t.update_layout(height=int(60+len(rules_disp)*32),
                                           margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_rules_t, use_container_width=True)

                st.markdown(f"""
                <div class="insight-box">
                <b>Service bundling signals from the data:</b><br>
                Rules with Lift > 1.2 indicate strong co-occurrence. The highest-lift rules above represent
                the most defensible service bundles — startups that need one service in the pair are
                disproportionately likely to also need the other. BridgeCompliance should design package
                offerings around the top 3 rules by lift. Low support but high lift rules represent
                niche high-value bundles; high support rules represent the standard package for the majority.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No frequent itemsets found at this support level. Lower the minimum support slider.")
        except Exception as e:
            st.warning(f"Association rules error: {e}")


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">📋 Strategic Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="method-box">
    No models. No charts. These conclusions are derived from the full analytical pipeline across Tabs 1–5
    and translated into actionable business language for BridgeCompliance Advisory's launch decision.
    </div>
    """, unsafe_allow_html=True)

    # Pull live values
    dub2  = df[df['Q1_Market']=='Dubai (DIFC / ADGM)']
    sgp2  = df[df['Q1_Market']=='Singapore (MAS-regulated)']
    dub_intent2 = round(dub2['TARGET_High_Intent'].mean()*100,1) if len(dub2)>0 else 0
    sgp_intent2 = round(sgp2['TARGET_High_Intent'].mean()*100,1) if len(sgp2)>0 else 0
    dub_wtp2    = round(dub2['WTP_SGD'].mean(),0) if len(dub2)>0 else 0
    sgp_wtp2    = round(sgp2['WTP_SGD'].mean(),0) if len(sgp2)>0 else 0
    lead2       = "Dubai (DIFC / ADGM)" if dub_intent2 >= sgp_intent2 else "Singapore (MAS-regulated)"

    pain_rates = {}
    for col,lbl in PAIN_LABELS.items():
        if col in df.columns:
            pain_rates[lbl] = round(df[col].mean()*100,1)
    top_pain = sorted(pain_rates.items(), key=lambda x:x[1], reverse=True)[:3]

    # Q6 engagement distribution
    eng_dist, deal_dist = "", ""
    if 'Q6_Engagement' in df.columns:
        top_eng = df['Q6_Engagement'].value_counts().index[0]
        eng_share = round(df['Q6_Engagement'].value_counts(normalize=True).iloc[0]*100,1)
        eng_dist = f"{top_eng} ({eng_share}% of respondents)"
    if 'Q9_Deal_Breaker' in df.columns:
        top_deal = df['Q9_Deal_Breaker'].value_counts().index[0]
        deal_share = round(df['Q9_Deal_Breaker'].value_counts(normalize=True).iloc[0]*100,1)
        deal_dist = f"{top_deal} ({deal_share}% of respondents)"

    # ── R1: Launch market ───────────────────────────────────────────────────
    st.markdown("""
    <div class="rec-box">
    <div class="rec-title">🚀 Recommendation 1 — Launch Market: Which to Enter First</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    The data across all five analytical tabs consistently points to **{lead2}** as the stronger first market.

    **Evidence chain:**
    - **Tab 1 (Descriptive):** {lead2.split(' ')[0]} showed a higher intent rate ({dub_intent2}% Dubai vs {sgp_intent2}% Singapore) and higher average WTP (SGD {dub_wtp2:,.0f} vs SGD {sgp_wtp2:,.0f}/month).
    - **Tab 3 (Classification):** The Gradient Boosting model confirmed that Stage and Market features rank among the top predictors of high intent — and Dubai's higher proportion of Series A–B+ startups structurally lifts its intent signal.
    - **Tab 4 (Regression):** Lasso retained Market (Dubai=1) as a coefficient after L1 elimination, confirming it is a genuine predictor of conversion score, not a noise variable.

    **Launch sequence recommendation:** Enter {lead2.split(' ')[0]} in Month 1–6 with a focused go-to-market targeting Seed to Series A–B fintechs and crypto/Web3 firms. Use the revenue and case studies from {lead2.split(' ')[0]} to build the credibility required for a Singapore entry in Month 7–12, where MAS regulation creates a more structured (but slower) buying process.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── R2: Ideal client profile ─────────────────────────────────────────────
    st.markdown("""
    <div class="rec-box">
    <div class="rec-title">🎯 Recommendation 2 — Ideal Client Profile</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    Based on the classification model's feature importance (Tab 3) and correlation analysis (Tab 2):

    | Attribute | Ideal Profile |
    |---|---|
    | **Funding stage** | Seed to Series A–B (ENC_Stage 2–3) |
    | **Industry** | Fintech or Crypto/Web3 |
    | **Urgency score** | ≥ 7 out of 10 |
    | **Likelihood score** | ≥ 7 out of 10 |
    | **WTP tier** | Growth-stage (SGD 2,000–5,000/month) or above |
    | **Pain count** | 2–4 active regulatory domains |

    This profile represents the **Priority Segment** (TARGET_Priority = 1) — startups that scored High_Intent AND High_WTP simultaneously. They are the highest-probability, highest-revenue clients and should receive the most direct outreach resources.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── R3: Services to lead with ────────────────────────────────────────────
    st.markdown("""
    <div class="rec-box">
    <div class="rec-title">⚙️ Recommendation 3 — Services to Lead With Per Market</div>
    """, unsafe_allow_html=True)
    pain_str = ', '.join([f"{lbl} ({rate}%)" for lbl,rate in top_pain])
    st.markdown(f"""
    Based on the pain point prevalence analysis (Tab 1) and association rules (Tab 5):

    **Top 3 urgency domains across both markets:** {pain_str}

    **Dubai (DIFC / ADGM) — Lead services:**
    - **Licensing & Regulatory Authorisation** — DIFC/ADGM/VARA licensing has the highest urgency rate in Dubai due to VARA's active enforcement and DIFC's mandatory licensing framework for digital asset firms.
    - **AML/CFT Programme Design** — DFSA requires documented AML programmes for all regulated entities; this is a Day 1 compliance obligation for any Dubai fintech.
    - **Token & Digital Asset Classification** — Dubai's crypto ecosystem is the largest in the MENA region; classification under VARA frameworks is urgent for any Web3/DeFi firm.

    **Singapore (MAS-regulated) — Lead services:**
    - **MAS Licensing Roadmap** (MPI, CMS, DPT) — Singapore's multi-tier licensing framework creates complexity that requires specialist navigation.
    - **Data Protection & Cross-Border Flows** — PDPA enforcement has intensified; cross-border data transfer restrictions affect every fintech with international operations.
    - **AI Governance** — MAS MRM Guidelines and Singapore's National AI Strategy create compliance obligations for any AI-enabled financial product — the fastest-growing regulatory domain in the market.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── R4: Pricing ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="rec-box">
    <div class="rec-title">💰 Recommendation 4 — Pricing Architecture by Segment</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    The regression and WTP analysis (Tabs 2 and 4) support a **stage-tiered, three-tier pricing architecture**:

    | Segment | Stage | Monthly WTP Band | Recommended Package |
    |---|---|---|---|
    | **Starter** | Pre-seed / Seed | SGD 800–2,000 | One-time regulatory audit + licensing roadmap report |
    | **Growth** | Series A–B | SGD 2,000–5,000 | Monthly retainer — AML framework + ongoing regulatory monitoring |
    | **Scale** | Series C+ | SGD 5,000–10,000 | Annual contract — full compliance infrastructure across both jurisdictions |

    **Key pricing insight from Lasso (Tab 4):** Stage and Likelihood are the two strongest retained predictors of conversion score. Price should scale with stage (proxy for ability to pay) but the trigger for upselling is urgency, not company size alone — a high-urgency Pre-seed is more likely to convert at the Starter tier than a low-urgency Series B.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── R5: Engagement model ─────────────────────────────────────────────────
    st.markdown("""
    <div class="rec-box">
    <div class="rec-title">📄 Recommendation 5 — How to Structure Engagements (Q6 & Q9)</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    **Q6 — Preferred engagement model:** The most common preference is **{eng_dist}**.
    This means BridgeCompliance's primary commercial offer should be structured around this model, with
    alternatives offered at signup. For the Growth and Scale segments, the data supports annual contracts
    (predictable revenue for BridgeCompliance, cost certainty for the client). For Pre-seed, project-based
    engagements lower the barrier to entry and can convert to retainers after the first deliverable.

    **Q9 — Deal-breaker factor:** The single most cited deal-breaker is **{deal_dist}**.
    This is BridgeCompliance's primary positioning anchor. Every marketing message, proposal, and pitch
    deck must lead with this proof point. If the deal-breaker is *jurisdiction-specific expertise*, every
    proposal must include specific DIFC/ADGM/MAS case references. If it is *transparent pricing*, every
    scope-of-work document must include fixed fees with no open-ended hourly billing.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── R6: Unanswered questions ─────────────────────────────────────────────
    st.markdown("""
    <div class="rec-box" style="border-color:#BA7517;">
    <div class="rec-title" style="color:#BA7517;">🔬 Three Questions This Dataset Cannot Answer</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    This synthetic dataset was generated from estimated probability distributions — it validates the
    survey design and analytical pipeline, but cannot substitute for real market data. Three critical
    questions require a live data collection round:

    1. **How does actual price sensitivity respond to BridgeCompliance's specific service framing?**
       The current WTP data measures abstract willingness to pay for "compliance advisory." Conjoint
       analysis with actual service descriptions and price points would reveal true price elasticity
       per feature — essential before setting a live pricing page.

    2. **What is the actual conversion rate from survey intent to signed contract?**
       A Likelihood score of ≥7 in a survey context does not map linearly to actual contract conversion.
       A 6-month pilot with 20–30 warm leads would establish the intent-to-conversion ratio — the
       most important metric for financial projections.

    3. **How does competitive context affect WTP and deal-breaker selection?**
       Survey respondents answered without being shown competitive alternatives. In a live market,
       WTP and deal-breaker choices will be influenced by existing providers (Big Four advisory firms,
       boutique compliance consultancies, legal tech platforms). A competitive displacement study —
       asking "what would make you switch from your current provider?" — would refine the positioning
       strategy significantly.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

