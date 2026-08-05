import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
MODEL_PATH = MODEL_DIR / "pure_lstm_model.keras"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"

CONTROL_PATTERN = re.compile(r"[\x00-\x1F\x7F]+")


# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------
def clean_url(value: str) -> str:
    """Apply the same preprocessing used during model training."""
    value = str(value).strip()
    value = CONTROL_PATTERN.sub("", value)
    value = re.sub(r"(?i)^https?://", "", value)
    value = value.strip().lower()

    if len(value) > 1:
        value = value.rstrip("/")

    return value


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold tidak ditemukan: {THRESHOLD_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
        thresholds = json.load(file)

    return model, thresholds


def predict_urls(urls: list[str], threshold_mode: str = "strict") -> pd.DataFrame:
    model, thresholds = load_artifacts()

    original_urls = [str(url) for url in urls]
    cleaned_urls = [clean_url(url) for url in original_urls]

    probabilities = (
        model.predict(
            tf.constant(cleaned_urls),
            batch_size=256,
            verbose=0,
        )
        .reshape(-1)
    )

    balanced_threshold = float(thresholds["balanced_threshold"])
    strict_threshold = float(thresholds["strict_threshold"])

    selected_threshold = (
        strict_threshold if threshold_mode == "strict" else balanced_threshold
    )

    predictions = (probabilities >= selected_threshold).astype(int)

    status = np.where(
        probabilities >= strict_threshold,
        "Phishing",
        np.where(
            probabilities >= balanced_threshold,
            "Perlu ditinjau",
            "Legitimate",
        ),
    )

    return pd.DataFrame(
        {
            "URL": original_urls,
            "Cleaned URL": cleaned_urls,
            "Phishing Probability": probabilities,
            "Threshold Used": selected_threshold,
            "Prediction": predictions,
            "Status": status,
        }
    )


# -----------------------------------------------------------------------------
# Page setup and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LinkSentry — Phishing URL Detector",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

    :root {
        --bg: #302627;
        --panel: rgba(73, 62, 62, 0.76);
        --panel-strong: rgba(63, 53, 53, 0.92);
        --line: rgba(255, 255, 255, 0.16);
        --text: #fff9f3;
        --muted: rgba(255, 249, 243, 0.58);
        --orange: #ff9f12;
        --orange-2: #ffb11f;
        --red: #f30b28;
        --green: #50d890;
        --warning: #ffd166;
    }

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        min-height: 100vh;
        color: var(--text);
        background:
            radial-gradient(circle at 4% 10%, #ffa313 0 15%, transparent 15.2%),
            radial-gradient(circle at 91% 8%, #fff 0 12%, transparent 12.2%),
            radial-gradient(circle at 94% 91%, #f20524 0 13%, transparent 13.2%),
            #302627;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(120deg, rgba(255, 159, 18, 0.08), transparent 32%),
            radial-gradient(circle at 52% 45%, rgba(255,255,255,0.035), transparent 34%);
        z-index: 0;
    }

    header[data-testid="stHeader"],
    footer,
    #MainMenu {
        visibility: hidden;
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1180px;
        padding: 5.2rem 1.25rem 3.5rem;
        position: relative;
        z-index: 2;
    }

    .hero-shell {
        position: relative;
        border-radius: 30px;
        padding: 30px 34px 18px;
        background:
            linear-gradient(135deg, rgba(122, 105, 70, 0.40), rgba(67, 55, 56, 0.86) 42%, rgba(66, 55, 55, 0.90));
        border: 1px solid rgba(255, 255, 255, 0.13);
        box-shadow:
            0 35px 90px rgba(15, 8, 9, 0.42),
            inset 0 1px 0 rgba(255, 255, 255, 0.10);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        overflow: hidden;
        min-height: 620px;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        width: 330px;
        height: 330px;
        left: -130px;
        top: -110px;
        border-radius: 50%;
        background: rgba(255, 166, 17, 0.17);
        filter: blur(2px);
        pointer-events: none;
    }

    .brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
        position: relative;
        z-index: 2;
    }

    .brand-mark {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: grid;
        place-items: center;
        background: linear-gradient(145deg, #ffb01c, #ff8b00);
        color: #322628;
        font-weight: 900;
        box-shadow: 0 10px 28px rgba(255, 159, 18, 0.27);
    }

    .brand-name {
        font-family: "Manrope", sans-serif;
        font-size: 1.42rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .brand-subtitle {
        margin-top: 5px;
        color: var(--muted);
        font-size: 0.70rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .model-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 13px;
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 999px;
        color: rgba(255,255,255,0.72);
        background: rgba(255,255,255,0.055);
        font-size: 0.74rem;
    }

    .model-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 14px rgba(80,216,144,0.8);
    }

    .left-pane {
        padding: 38px 28px 18px 18px;
        min-height: 470px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .eyebrow {
        color: var(--orange-2);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 700;
        font-size: 0.72rem;
        margin-bottom: 14px;
    }

    .hero-title {
        margin: 0;
        max-width: 540px;
        font-family: "Manrope", sans-serif;
        font-weight: 800;
        font-size: clamp(2.2rem, 4vw, 4.25rem);
        line-height: 1.00;
        letter-spacing: -0.06em;
        color: var(--text);
    }

    .hero-title span {
        color: var(--orange-2);
    }

    .hero-copy {
        max-width: 510px;
        margin-top: 18px;
        color: rgba(255,255,255,0.58);
        font-size: 0.94rem;
        line-height: 1.75;
    }

    .visual-wrap {
        position: relative;
        min-height: 230px;
        display: grid;
        place-items: center;
        margin-top: 5px;
    }

    .shield-orbit {
        width: 205px;
        height: 205px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background:
            radial-gradient(circle at 35% 27%, rgba(255,255,255,0.20), transparent 20%),
            linear-gradient(145deg, #6a6061, #282224);
        border: 1px solid rgba(255,255,255,0.17);
        box-shadow:
            0 28px 55px rgba(12,7,8,0.44),
            inset 7px 7px 16px rgba(255,255,255,0.07),
            inset -12px -12px 22px rgba(0,0,0,0.34);
        transform: rotate(-8deg);
    }

    .shield-core {
        width: 118px;
        height: 138px;
        clip-path: polygon(50% 0, 90% 15%, 86% 68%, 50% 100%, 14% 68%, 10% 15%);
        display: grid;
        place-items: center;
        background: linear-gradient(155deg, #cbc4c2, #3b3436 55%, #1e1a1b);
        color: #ffab17;
        font-size: 3.15rem;
        filter: drop-shadow(0 14px 18px rgba(0,0,0,0.42));
    }

    .orbit-ring {
        position: absolute;
        width: 285px;
        height: 125px;
        border: 1px solid rgba(255, 176, 28, 0.36);
        border-radius: 50%;
        transform: rotate(-17deg);
        box-shadow: 0 0 24px rgba(255,159,18,0.08);
    }

    .mini-chip {
        position: absolute;
        padding: 8px 11px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(43,36,37,0.78);
        color: rgba(255,255,255,0.68);
        font-size: 0.68rem;
        backdrop-filter: blur(10px);
    }

    .chip-a { left: 2%; top: 18%; }
    .chip-b { right: 1%; top: 30%; }
    .chip-c { left: 7%; bottom: 12%; }

    .left-footer {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    .stat-card {
        min-width: 132px;
        padding: 12px 14px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.045);
    }

    .stat-value {
        color: #fff;
        font-family: "Manrope", sans-serif;
        font-size: 1.02rem;
        font-weight: 800;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.65rem;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .right-pane {
        height: 100%;
        min-height: 470px;
        padding: 30px 12px 20px 34px;
        border-left: 1px solid rgba(255,255,255,0.12);
    }

    .scanner-heading {
        font-family: "Manrope", sans-serif;
        font-size: 1.62rem;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 4px 0 8px;
    }

    .scanner-copy {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.65;
        margin-bottom: 24px;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stRadio"] > label {
        color: rgba(255,255,255,0.72) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTextInput"] input {
        height: 52px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(28, 23, 24, 0.34);
        color: #fff;
        padding-left: 16px;
        box-shadow: inset 0 1px 6px rgba(0,0,0,0.20);
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(255, 170, 20, 0.78);
        box-shadow: 0 0 0 3px rgba(255, 159, 18, 0.13);
    }

    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        gap: 8px;
        background: rgba(20, 16, 17, 0.21);
        border: 1px solid rgba(255,255,255,0.09);
        padding: 6px;
        border-radius: 14px;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        flex: 1;
        justify-content: center;
        padding: 8px 10px;
        border-radius: 10px;
        margin: 0;
        background: rgba(255,255,255,0.03);
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        min-height: 48px;
        border: 0;
        border-radius: 13px;
        color: #332629;
        background: linear-gradient(90deg, #ffab16, #ff9912);
        font-weight: 800;
        box-shadow: 0 14px 25px rgba(255, 143, 0, 0.20);
        transition: transform .18s ease, box-shadow .18s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        color: #332629;
        border: 0;
        transform: translateY(-1px);
        box-shadow: 0 18px 32px rgba(255, 143, 0, 0.28);
    }

    .threshold-row {
        display: flex;
        gap: 10px;
        margin: 14px 0 20px;
    }

    .threshold-card {
        flex: 1;
        padding: 11px 12px;
        border-radius: 13px;
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.035);
    }

    .threshold-name {
        color: var(--muted);
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .threshold-value {
        margin-top: 4px;
        color: rgba(255,255,255,0.86);
        font-weight: 800;
        font-size: 0.90rem;
    }

    .result-card {
        margin-top: 19px;
        padding: 16px 17px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.11);
        background: rgba(27,22,23,0.30);
    }

    .result-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }

    .result-label {
        color: var(--muted);
        font-size: 0.67rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
    }

    .result-status {
        margin-top: 4px;
        font-family: "Manrope", sans-serif;
        font-size: 1.28rem;
        font-weight: 800;
    }

    .status-legitimate { color: var(--green); }
    .status-review { color: var(--warning); }
    .status-phishing { color: #ff576c; }

    .probability-value {
        color: #fff;
        font-weight: 800;
        font-size: 1.12rem;
    }

    .meter {
        width: 100%;
        height: 8px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        margin: 14px 0 11px;
    }

    .meter-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #50d890, #ffb11f 58%, #f30b28);
    }

    .result-url {
        color: rgba(255,255,255,0.55);
        font-size: 0.73rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    div[data-testid="stExpander"] {
        margin-top: 22px;
        border: 1px solid rgba(255,255,255,0.11);
        border-radius: 15px;
        background: rgba(255,255,255,0.035);
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(20,16,17,0.22);
        border: 1px dashed rgba(255,255,255,0.18);
        border-radius: 14px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    .privacy-note {
        text-align: center;
        color: rgba(255,255,255,0.35);
        font-size: 0.68rem;
        margin-top: 18px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 2rem;
        }

        .hero-shell {
            padding: 22px 18px;
        }

        .right-pane {
            border-left: 0;
            border-top: 1px solid rgba(255,255,255,0.12);
            padding: 28px 2px 4px;
        }

        .left-pane {
            padding: 24px 2px 12px;
        }

        .visual-wrap {
            min-height: 200px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    model, thresholds = load_artifacts()
except Exception as error:
    st.error(
        "Model gagal dimuat. Pastikan folder `model` berisi "
        "`pure_lstm_model.keras` dan `thresholds.json`."
    )
    st.exception(error)
    st.stop()

balanced_threshold = float(thresholds["balanced_threshold"])
strict_threshold = float(thresholds["strict_threshold"])


# -----------------------------------------------------------------------------
# Main glass panel
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-shell">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-mark">
            <div class="brand-icon">↗</div>
            <div>
                <div class="brand-name">LinkSentry</div>
                <div class="brand-subtitle">AI URL Security</div>
            </div>
        </div>
        <div class="model-pill">
            <span class="model-dot"></span>
            Pure LSTM model online
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.12, 0.88], gap="large")

with left_col:
    st.markdown(
        """
        <div class="left-pane">
            <div>
                <div class="eyebrow">Intelligent threat screening</div>
                <h1 class="hero-title">Know the link.<br><span>Before you click.</span></h1>
                <div class="hero-copy">
                    Analisis pola karakter URL menggunakan model LSTM untuk membantu
                    mengenali indikasi phishing secara cepat, aman, dan sederhana.
                </div>
            </div>

            <div class="visual-wrap">
                <div class="orbit-ring"></div>
                <div class="shield-orbit">
                    <div class="shield-core">⌁</div>
                </div>
                <div class="mini-chip chip-a">◉ Character scan</div>
                <div class="mini-chip chip-b">✓ Risk scoring</div>
                <div class="mini-chip chip-c">⌕ URL pattern</div>
            </div>

            <div class="left-footer">
                <div class="stat-card">
                    <div class="stat-value">LSTM</div>
                    <div class="stat-label">Model engine</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">3-level</div>
                    <div class="stat-label">Risk status</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">CSV</div>
                    <div class="stat-label">Batch scanning</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown('<div class="right-pane">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="eyebrow">URL checkout</div>
        <div class="scanner-heading">Check a suspicious link</div>
        <div class="scanner-copy">
            Tempel URL lengkap. Sistem akan membersihkan format URL,
            menghitung probabilitas phishing, dan menampilkan tingkat risikonya.
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold_mode = st.radio(
        "Detection sensitivity",
        options=["strict", "balanced"],
        index=0,
        horizontal=True,
        format_func=lambda value: (
            "Strict · fewer false positives"
            if value == "strict"
            else "Balanced · higher recall"
        ),
    )

    st.markdown(
        f"""
        <div class="threshold-row">
            <div class="threshold-card">
                <div class="threshold-name">Balanced threshold</div>
                <div class="threshold-value">{balanced_threshold:.4f}</div>
            </div>
            <div class="threshold-card">
                <div class="threshold-name">Strict threshold</div>
                <div class="threshold-value">{strict_threshold:.4f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("single_url_form", clear_on_submit=False):
        url_input = st.text_input(
            "URL address",
            placeholder="https://example.com/account/login",
        )
        submitted = st.form_submit_button(
            "Scan URL",
            use_container_width=True,
        )

    if submitted:
        if not url_input.strip():
            st.warning("Masukkan URL terlebih dahulu.")
        else:
            with st.spinner("Menganalisis pola URL..."):
                result = predict_urls(
                    [url_input],
                    threshold_mode=threshold_mode,
                )

            probability = float(result.loc[0, "Phishing Probability"])
            status = str(result.loc[0, "Status"])
            cleaned = str(result.loc[0, "Cleaned URL"])

            status_class = {
                "Legitimate": "status-legitimate",
                "Perlu ditinjau": "status-review",
                "Phishing": "status-phishing",
            }.get(status, "status-review")

            safe_url = html.escape(cleaned)
            meter_width = max(1.0, min(probability * 100, 100.0))

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-top">
                        <div>
                            <div class="result-label">Detection result</div>
                            <div class="result-status {status_class}">{html.escape(status)}</div>
                        </div>
                        <div style="text-align:right">
                            <div class="result-label">Phishing probability</div>
                            <div class="probability-value">{probability:.2%}</div>
                        </div>
                    </div>
                    <div class="meter">
                        <div class="meter-fill" style="width:{meter_width:.2f}%"></div>
                    </div>
                    <div class="result-url">{safe_url}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if status == "Phishing":
                st.error(
                    "Indikasi phishing tinggi. Jangan membuka URL atau "
                    "memasukkan kredensial dan informasi pribadi."
                )
            elif status == "Perlu ditinjau":
                st.warning(
                    "URL berada di area abu-abu. Verifikasi domain dan sumbernya "
                    "sebelum membuka tautan."
                )
            else:
                st.success(
                    "Model menilai URL sebagai legitimate. Tetap periksa domain "
                    "dan konteks pengirim sebelum melanjutkan."
                )

    with st.expander("Batch scan · upload CSV"):
        st.caption("File harus mempunyai kolom bernama `url`.")
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
            except Exception as error:
                st.error("File CSV tidak dapat dibaca.")
                st.exception(error)
                st.stop()

            if "url" not in uploaded_data.columns:
                st.error("CSV harus memiliki kolom bernama `url`.")
            else:
                st.caption(f"{len(uploaded_data):,} URL siap dianalisis.")

                if st.button(
                    "Run batch scan",
                    use_container_width=True,
                ):
                    with st.spinner("Melakukan prediksi batch..."):
                        prediction_result = predict_urls(
                            uploaded_data["url"]
                            .fillna("")
                            .astype(str)
                            .tolist(),
                            threshold_mode=threshold_mode,
                        )

                        final_result = pd.concat(
                            [
                                uploaded_data.reset_index(drop=True),
                                prediction_result[
                                    [
                                        "Cleaned URL",
                                        "Phishing Probability",
                                        "Threshold Used",
                                        "Prediction",
                                        "Status",
                                    ]
                                ],
                            ],
                            axis=1,
                        )

                    st.dataframe(
                        final_result,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download prediction results",
                        data=final_result.to_csv(index=False).encode("utf-8"),
                        file_name="phishing_url_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    st.markdown(
        '<div class="privacy-note">URL diproses langsung oleh model dan tidak disimpan oleh aplikasi.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
