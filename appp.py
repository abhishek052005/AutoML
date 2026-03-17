import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        background-color: #0a0a0f;
        color: #e8e6ff;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #4f46e5; border-radius: 2px; }

    /* ── Main container ── */
    .main .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1200px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f0f1a !important;
        border-right: 1px solid #1e1e3a;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.2rem;
    }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-tagline {
        font-size: 0.68rem;
        color: #4b5563;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* ── Step nav buttons ── */
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 0.65rem 1rem !important;
        margin-bottom: 0.35rem !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: #1a1a2e !important;
        border-color: #4f46e5 !important;
        color: #a5b4fc !important;
    }
    div[data-testid="stRadio"] [data-checked="true"] + label,
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: #1e1b4b !important;
        border-color: #6366f1 !important;
        color: #c7d2fe !important;
    }

    /* ── Page header ── */
    .page-header {
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #1e1e3a;
    }
    .page-step-label {
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 0.4rem;
    }
    .page-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.03em;
        color: #f0eeff;
        margin: 0;
        line-height: 1.1;
    }
    .page-subtitle {
        margin-top: 0.5rem;
        font-size: 0.78rem;
        color: #6b7280;
    }

    /* ── Stat cards ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #0f0f1a;
        border: 1px solid #1e1e3a;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: linear-gradient(180deg, #6366f1, #8b5cf6);
        border-radius: 3px 0 0 3px;
    }
    .stat-label {
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4b5563;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1.6rem;
        color: #e0e7ff;
        letter-spacing: -0.02em;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        background: #0f0f1a !important;
        border: 2px dashed #2d2d4e !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.8rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
        filter: brightness(1.1) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #059669, #0d9488) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.08em !important;
        box-shadow: 0 4px 20px rgba(16,185,129,0.3) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(16,185,129,0.5) !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background: #0f0f1a !important;
        border: 1px solid #2d2d4e !important;
        border-radius: 8px !important;
        color: #e0e7ff !important;
        font-size: 0.82rem !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e1e3a !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    .stDataFrame > div {
        background: #0f0f1a !important;
    }

    /* ── Alerts ── */
    .stSuccess {
        background: #052e16 !important;
        border: 1px solid #166534 !important;
        border-radius: 8px !important;
        color: #bbf7d0 !important;
        font-size: 0.8rem !important;
    }
    .stWarning {
        background: #1c1a07 !important;
        border: 1px solid #713f12 !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }

    /* ── Subheaders ── */
    h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        font-size: 1.1rem !important;
        color: #c7d2fe !important;
        margin-top: 1.8rem !important;
        margin-bottom: 0.8rem !important;
    }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #1e1e3a;
        margin: 1.5rem 0;
    }

    /* ── Info badge ── */
    .badge {
        display: inline-block;
        background: #1e1b4b;
        border: 1px solid #4338ca;
        color: #a5b4fc;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin-right: 0.4rem;
    }

    /* ── Model result card ── */
    .result-header {
        background: linear-gradient(135deg, #1e1b4b, #1e1033);
        border: 1px solid #4338ca;
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
    }
    .result-header-label {
        font-size: 0.65rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #818cf8;
        margin-bottom: 0.3rem;
    }
    .result-header-value {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: #e0e7ff;
    }

    /* ── Step indicator in sidebar ── */
    .step-indicator {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        margin-bottom: 2rem;
    }
    .step-dot-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.3rem 0;
    }
    .step-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #2d2d4e;
        flex-shrink: 0;
    }
    .step-dot.done { background: #6366f1; }
    .step-dot.active { background: #a78bfa; box-shadow: 0 0 8px #a78bfa; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡ AutoML</div>', unsafe_allow_html=True)

    STEPS = [
        ("📂", "Upload Dataset"),
        ("🔍", "Data Analysis"),
        ("📊", "Visualization"),
        ("🧠", "Train Models"),
        ("💾", "Download Model"),
    ]

    step_labels = [f"{icon}  {label}" for icon, label in STEPS]
    choice_full = st.radio("", step_labels, label_visibility="collapsed")
    choice = choice_full.split("  ", 1)[1]  # strip icon

    st.markdown("<hr style='border-color:#1e1e3a; margin: 1.5rem 0'>", unsafe_allow_html=True)

   # Dataset status
    if st.session_state.df is not None:
        df = st.session_state.df
        filename = st.session_state.get("filename", "Unknown file")

        st.markdown(
            f"""
            <div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;
                        color:#6b7280;margin-bottom:0.6rem;">Active Dataset</div>
            <div style="background:#0f1020;border:1px solid #2d2d4e;border-radius:8px;padding:0.8rem 1rem;">
                <div style="font-size:0.78rem;color:#a5b4fc;font-weight:500;margin-bottom:0.4rem;">
                    {filename}
                </div>
                <div style="font-size:0.65rem;color:#4b5563;">
                    {df.shape[0]:,} rows · {df.shape[1]} cols
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.72rem;color:#4b5563;background:#0f0f1a;'
            'border:1px dashed #2d2d4e;border-radius:8px;padding:0.8rem 1rem;">'
            'No dataset loaded</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.best_model is not None:
        st.markdown(
            """
            <div style="margin-top:1rem;background:#052e16;border:1px solid #166534;
                        border-radius:8px;padding:0.7rem 1rem;font-size:0.72rem;color:#86efac;">
                ✓ Model trained & ready
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
#  HELPER: page header
# ─────────────────────────────────────────────
def page_header(step_num, title, subtitle=""):
    step_labels_map = {
        "Upload Dataset": "01",
        "Data Analysis": "02",
        "Visualization": "03",
        "Train Models": "04",
        "Download Model": "05",
    }
    num = step_labels_map.get(title, "—")
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-step-label">Step {num}</div>
            <div class="page-title">{title}</div>
            {"<div class='page-subtitle'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  1. UPLOAD DATASET
# ─────────────────────────────────────────────
if choice == "Upload Dataset":
    page_header(1, "Upload Dataset", "Start by loading your CSV file into the studio")

    file = st.file_uploader(
        "Drop your CSV here, or click to browse",
        type=["csv"],
        help="Only .csv files are supported.",
    )

    if file:
        
        df = pd.read_csv(file)
        st.session_state.df = df
        st.session_state.filename = file.name

        # Stats row
        num_numeric = df.select_dtypes(include="number").shape[1]
        num_cat = df.select_dtypes(include="object").shape[1]
        missing_pct = round(df.isnull().sum().sum() / df.size * 100, 1)

        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat-card">
                    <div class="stat-label">Rows</div>
                    <div class="stat-value">{df.shape[0]:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Columns</div>
                    <div class="stat-value">{df.shape[1]}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Numeric</div>
                    <div class="stat-value">{num_numeric}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Missing %</div>
                    <div class="stat-value">{missing_pct}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(f"✓  **{file.name}** loaded successfully — {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.subheader("Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.markdown(
            """
            <div style="margin-top:1rem;padding:2rem;background:#0f0f1a;border:1px dashed #2d2d4e;
                        border-radius:16px;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.8rem;">📂</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;
                            color:#4b5563;letter-spacing:-0.01em;">
                    No file selected
                </div>
                <div style="font-size:0.75rem;color:#374151;margin-top:0.4rem;">
                    Supported format: CSV
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
#  2. DATA ANALYSIS
# ─────────────────────────────────────────────
elif choice == "Data Analysis":
    page_header(2, "Data Analysis", "Explore structure, distributions and statistics")

    if st.session_state.df is not None:
        df = st.session_state.df

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📐 Shape & Columns", "📊 Statistics", "🔴 Missing Values"])

        with tab1:
            st.dataframe(df, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Shape")
                st.markdown(
                    f'<div style="font-family:Syne,sans-serif;font-size:2.5rem;font-weight:800;'
                    f'color:#818cf8;">{df.shape[0]:,} <span style="font-size:1.2rem;color:#4b5563;">rows</span>'
                    f'  ×  {df.shape[1]} <span style="font-size:1.2rem;color:#4b5563;">cols</span></div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.subheader("Data Types")
                dtype_df = pd.DataFrame(df.dtypes.astype(str), columns=["dtype"]).reset_index()
                dtype_df.columns = ["Column", "Type"]
                st.dataframe(dtype_df, use_container_width=True, height=220)

        with tab3:
            st.subheader("Summary Statistics")
            st.dataframe(df.describe().T, use_container_width=True)

        with tab4:
            missing = df.isnull().sum()
            missing_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": (missing.values / len(df) * 100).round(2),
            })
            missing_df = missing_df[missing_df["Missing Count"] > 0].reset_index(drop=True)
            if missing_df.empty:
                st.success("✓  No missing values found in the dataset.")
            else:
                st.warning(f"⚠  {len(missing_df)} columns have missing values.")
                st.dataframe(missing_df, use_container_width=True)

    else:
        st.warning("⚠  Upload a dataset in Step 01 first.")

# ─────────────────────────────────────────────
#  3. VISUALIZATION
# ─────────────────────────────────────────────
elif choice == "Visualization":
    page_header(3, "Visualization", "Auto-generate a full report")

    if st.session_state.df is not None:
        df = st.session_state.df


        if st.button("⚡  Generate Full Report"):
            with st.spinner("Analysing dataset…"):
                from ydata_profiling import ProfileReport
                from streamlit_pandas_profiling import st_profile_report
                profile = ProfileReport(df, explorative=True)
                st_profile_report(profile)
    else:
        st.warning("⚠  Upload a dataset in Step 01 first.")

# ─────────────────────────────────────────────
#  4. TRAIN MODELS
# ─────────────────────────────────────────────
elif choice == "Train Models":
    page_header(4, "Train Models")

    if st.session_state.df is not None:
        df = st.session_state.df

        col1, col2 = st.columns([2, 1])
        with col1:
            target = st.selectbox(
                "Select target column",
                df.columns,
                help="The column you want to predict.",
            )
        with col2:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            run = st.button(" Run AutoML")

        if run:
            st.session_state.target_col = target
            with st.spinner("Training & comparing models — this may take a moment…"):
                setup(data=df, target=target, verbose=False)
                best_model = compare_models()
                results = pull()
                st.session_state.best_model = best_model
                st.session_state.model_results = results

        if st.session_state.model_results is not None:
            results = st.session_state.model_results
            best_name = str(type(st.session_state.best_model).__name__)

            st.markdown(
                f"""
                <div class="result-header">
                    <div class="result-header-label">Best Model</div>
                    <div class="result-header-value">⚡ {best_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Model Comparison")
            st.dataframe(results, use_container_width=True)
            st.success("✓  Training complete. Proceed to Step 05 to download the model.")
    else:
        st.warning("⚠  Upload a dataset in Step 01 first.")

# ─────────────────────────────────────────────
#  5. DOWNLOAD MODEL
# ─────────────────────────────────────────────
elif choice == "Download Model":
    page_header(5, "Download Model", "Export the best trained model as a .pkl file")

    if st.session_state.best_model is not None:
        best_name = str(type(st.session_state.best_model).__name__)
        target_col = st.session_state.target_col or "unknown"

        st.markdown(
            f"""
            <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:12px;
                        padding:1.4rem 1.8rem;margin-bottom:1.5rem;display:flex;
                        align-items:center;gap:1.2rem;">
                <div style="font-size:2.5rem;">🧠</div>
                <div>
                    <div style="font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;
                                color:#3b82f6;margin-bottom:0.3rem;">Model ready to export</div>
                    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;
                                color:#bfdbfe;">{best_name}</div>
                    <div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;">
                        Target: <span style="color:#93c5fd;">{target_col}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        save_model(st.session_state.best_model, "best_model")
        with open("best_model.pkl", "rb") as f:
            st.download_button(
                label="⬇  Download best_model.pkl",
                data=f,
                file_name="best_model.pkl",
                mime="application/octet-stream",
            )


    else:
        st.markdown(
            """
            <div style="padding:2rem;background:#0f0f1a;border:1px dashed #2d2d4e;
                        border-radius:16px;text-align:center;">
                <div style="font-size:2rem;margin-bottom:0.8rem;">🔒</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;
                            color:#4b5563;">No model available</div>
                <div style="font-size:0.75rem;color:#374151;margin-top:0.4rem;">
                    Complete Step 04 — Train Models first.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )