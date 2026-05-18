import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentiment_utils import analyze_single_text, analyze_batch

# --- Page Config ---
st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Light background */
    .stApp {
        background-color: #f5f7fa;
        color: #1a1f2e;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #1a1f2e !important;
        letter-spacing: -0.5px;
    }

    /* Sidebar — slate blue */
    section[data-testid="stSidebar"] {
        background-color: #1e2640 !important;
        border-right: none;
    }
    section[data-testid="stSidebar"] * {
        color: #c8d0e0 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #c8d0e0 !important;
        font-size: 0.95rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #1e2640 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    /* Sentiment pills */
    .score-pill {
        display: inline-block;
        padding: 7px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 12px;
        letter-spacing: 0.3px;
    }
    .pill-positive { background: #dcfce7; color: #15803d; border: 2px solid #86efac; }
    .pill-negative { background: #fee2e2; color: #b91c1c; border: 2px solid #fca5a5; }
    .pill-neutral  { background: #fef9c3; color: #92400e; border: 2px solid #fde68a; }

    /* Buttons */
    .stButton > button {
        background: #1e2640;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.55rem 1.6rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #2e3a5c;
        color: #ffffff;
        border: none;
    }

    /* Text area */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #1a1f2e !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #1e2640 !important;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #94a3b8;
        border-radius: 12px;
        padding: 12px;
    }

    /* Divider */
    hr { border-color: #e2e8f0 !important; }

    /* Alerts */
    .stAlert { border-radius: 10px !important; }

    /* Hero */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #1a1f2e;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .hero-sub {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 28px;
    }
    .badge {
        background: #e0e7ff;
        color: #3730a3;
        border: 1px solid #a5b4fc;
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        display: inline-block;
        margin-bottom: 14px;
        text-transform: uppercase;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def sentiment_pill(label: str) -> str:
    cls = {"Positive": "pill-positive", "Negative": "pill-negative"}.get(label, "pill-neutral")
    return f"<span class='score-pill {cls}'>{label}</span>"

def plotly_light(fig):
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font_color="#475569",
        font_family="Inter",
        title_font_family="Syne",
        title_font_color="#1a1f2e",
        title_font_size=15,
        margin=dict(t=48, b=20, l=20, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#475569")
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)
    st.markdown("## 🧠 SentimentAI")
    st.markdown("<p style='color:#475569;font-size:0.85rem;margin-top:-6px'>Natural language sentiment analysis</p>", unsafe_allow_html=True)
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["📊  Dashboard & Batch", "🔍  Analyze Single Text"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='color:#8896b0;font-size:0.78rem;line-height:1.6'>Powered by NLTK VADER — a lexicon-based sentiment engine optimised for short, social-media style text.</p>", unsafe_allow_html=True)


# ── Single Text ───────────────────────────────────────────────────────────────

if "Single" in mode:
    st.markdown("<div class='badge'>SINGLE TEXT ANALYSIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Analyze a piece of text</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Paste any sentence, review, comment, or message below to instantly score its sentiment.</div>", unsafe_allow_html=True)

    user_input = st.text_area(
        "Text input",
        height=130,
        placeholder="e.g.  The delivery was fast and the packaging was great — really happy with this purchase.",
        label_visibility="collapsed"
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        run = st.button("Analyze →")

    if run:
        if not user_input.strip():
            st.warning("Please enter some text before clicking Analyze.")
        else:
            with st.spinner("Running analysis…"):
                result = analyze_single_text(user_input)

            st.markdown("---")

            # Top row
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Compound Score", f"{result['compound']:+.3f}")
            r2.metric("Positive", f"{result['pos']*100:.1f}%")
            r3.metric("Neutral",  f"{result['neu']*100:.1f}%")
            r4.metric("Negative", f"{result['neg']*100:.1f}%")

            st.markdown(f"**Classification:** {sentiment_pill(result['label'])}", unsafe_allow_html=True)
            st.caption("Compound score ranges from −1.0 (most negative) to +1.0 (most positive). Scores ≥ 0.05 are Positive, ≤ −0.05 are Negative, and everything in between is Neutral.")

            # Gauge chart
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['compound'],
                number={"font": {"family": "Syne", "color": "#1a1f2e", "size": 36}},
                gauge={
                    "axis": {"range": [-1, 1], "tickcolor": "#94a3b8", "tickfont": {"color": "#475569"}},
                    "bar": {"color": "#1e2640"},
                    "bgcolor": "#f1f5f9",
                    "bordercolor": "#e2e8f0",
                    "steps": [
                        {"range": [-1, -0.05], "color": "#fecaca"},
                        {"range": [-0.05, 0.05], "color": "#fef9c3"},
                        {"range": [0.05, 1],    "color": "#bbf7d0"},
                    ],
                    "threshold": {"line": {"color": "#1a1f2e", "width": 3}, "value": result['compound']}
                },
                title={"text": "Sentiment Score", "font": {"family": "Syne", "color": "#475569", "size": 14}}
            ))
            gauge = plotly_light(gauge)
            gauge.update_layout(height=280, paper_bgcolor="#ffffff")
            st.plotly_chart(gauge, use_container_width=True)


# ── Dashboard & Batch ─────────────────────────────────────────────────────────

elif "Dashboard" in mode:
    st.markdown("<div class='badge'>BATCH ANALYSIS & DASHBOARD</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Upload your dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Upload a CSV file that contains a column of text (named <code>text</code>, <code>review</code>, or <code>tweet</code>) and get a full sentiment breakdown.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    # ── Demo data helper ──
    with st.expander("📂  Need a sample file? Generate demo data"):
        if st.button("Generate & Download Demo CSV"):
            demo_data = pd.DataFrame({"text": [
                "I absolutely love this product, it's amazing!",
                "Terrible experience, would not recommend.",
                "It's okay, does the job but nothing special.",
                "Fantastic customer service and fast shipping.",
                "The app keeps crashing, very frustrating.",
                "Best purchase I've made all year.",
                "Too expensive for what you get.",
                "Works exactly as described.",
                "I hate the new update.",
                "Highly recommended to everyone!",
                "The quality exceeded my expectations.",
                "Slow delivery and poor packaging.",
                "Neutral experience overall.",
                "Five stars — absolutely brilliant.",
                "Complete waste of money.",
            ]})
            csv_bytes = demo_data.to_csv(index=False).encode()
            st.download_button("⬇  Download demo_data.csv", csv_bytes, file_name="demo_data.csv", mime="text/csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Auto-detect text column
            text_col = None
            for col in df.columns:
                if any(kw in str(col).lower() for kw in ["text", "review", "tweet", "comment", "message", "content"]):
                    text_col = col
                    break

            if text_col is None:
                st.error("Could not find a text column. Make sure one of your columns is named `text`, `review`, `tweet`, `comment`, `message`, or `content`.")
            else:
                st.success(f"Loaded **{len(df):,}** rows — analysing column: `{text_col}`")

                with st.spinner(f"Analysing {len(df):,} rows…"):
                    results_df = analyze_batch(df[text_col].tolist())

                full_df = pd.concat([df, results_df.drop(columns=["Text"])], axis=1)

                st.markdown("---")

                # ── Key metrics ──
                total     = len(results_df)
                pos_count = (results_df["Label"] == "Positive").sum()
                neg_count = (results_df["Label"] == "Negative").sum()
                neu_count = (results_df["Label"] == "Neutral").sum()
                avg_score = results_df["Compound"].mean()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Entries",    f"{total:,}")
                m2.metric("Positive",         f"{pos_count/total*100:.1f}%")
                m3.metric("Negative",         f"{neg_count/total*100:.1f}%")
                m4.metric("Avg Score",        f"{avg_score:+.3f}")

                st.markdown("---")

                # ── Charts ──
                c1, c2 = st.columns(2)

                with c1:
                    fig_pie = px.pie(
                        results_df,
                        names="Label",
                        title="Sentiment Distribution",
                        color="Label",
                        color_discrete_map={
                            "Positive": "#4ade80",
                            "Neutral":  "#fbbf24",
                            "Negative": "#f87171"
                        },
                        hole=0.5
                    )
                    fig_pie = plotly_light(fig_pie)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    fig_hist = px.histogram(
                        results_df,
                        x="Compound",
                        title="Compound Score Distribution",
                        nbins=30,
                        color_discrete_sequence=["#3b4fd8"]
                    )
                    fig_hist.add_vline(x=avg_score, line_dash="dash", line_color="#1e2640",
                                       annotation_text=f"avg {avg_score:+.2f}", annotation_font_color="#1e2640")
                    fig_hist = plotly_light(fig_hist)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Bar chart breakdown
                label_counts = results_df["Label"].value_counts().reset_index()
                label_counts.columns = ["Sentiment", "Count"]
                fig_bar = px.bar(
                    label_counts,
                    x="Sentiment",
                    y="Count",
                    title="Entry Count by Sentiment",
                    color="Sentiment",
                    color_discrete_map={"Positive": "#4ade80", "Neutral": "#fbbf24", "Negative": "#f87171"}
                )
                fig_bar = plotly_light(fig_bar)
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("---")

                # ── Insights ──
                dominant = "Positive" if pos_count >= neg_count and pos_count >= neu_count else \
                           "Negative" if neg_count >= pos_count and neg_count >= neu_count else "Neutral"

                if dominant == "Positive" and pos_count / total > 0.5:
                    insight = "The overall tone is strongly positive. More than half of entries express satisfaction or approval."
                elif dominant == "Negative" and neg_count / total > 0.5:
                    insight = "The overall tone is predominantly negative. There are clear areas of dissatisfaction worth investigating."
                elif dominant == "Neutral":
                    insight = "Most entries are neutral in tone — factual, mixed, or lacking strong emotional language."
                else:
                    insight = "Sentiment is divided. No single tone dominates, suggesting a broad range of opinions."

                st.markdown("### Summary")
                st.info(
                    f"**{total:,}** entries analysed. "
                    f"**{pos_count}** positive · **{neg_count}** negative · **{neu_count}** neutral. "
                    f"Average compound score: **{avg_score:+.3f}**.\n\n{insight}"
                )

                # ── Raw data ──
                st.markdown("---")
                st.markdown("### Data Table")
                st.markdown("<p style='color:#475569;font-size:0.85rem'>Showing up to 200 rows. Download the full dataset below.</p>", unsafe_allow_html=True)
                st.dataframe(full_df.head(200), use_container_width=True)

                csv_out = full_df.to_csv(index=False).encode()
                st.download_button("⬇  Download full results CSV", csv_out, file_name="sentiment_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        # Empty state illustration
        st.markdown("""
        <div class='empty-state'>
            <div style='font-size:3rem;margin-bottom:12px'>📂</div>
            <div style='font-size:1.1rem;font-weight:700;color:#1a1f2e'>No file uploaded yet</div>
            <div style='font-size:0.9rem;margin-top:6px;color:#64748b'>Upload a CSV above, or expand the demo section to get a sample file.</div>
        </div>
        """, unsafe_allow_html=True)