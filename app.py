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

CONTROL_PATTERN = re.compile(r"[\x00-\x1F\x7F-\x9F]+")


def clean_url(value: str) -> str:
    """Apply the same normalization used during model training."""
    value = str(value).strip()
    value = CONTROL_PATTERN.sub("", value)
    value = re.sub(r"(?i)^https?://", "", value)
    value = value.strip().lower()

    if len(value) > 1:
        value = value.rstrip("/")

    return value


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold tidak ditemukan: {THRESHOLD_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with THRESHOLD_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        thresholds = json.load(file)

    return model, thresholds


def predict_urls(
    urls: list[str],
    threshold_mode: str = "strict",
) -> pd.DataFrame:
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

    balanced_threshold = float(
        thresholds["balanced_threshold"]
    )
    strict_threshold = float(
        thresholds["strict_threshold"]
    )

    selected_threshold = (
        strict_threshold
        if threshold_mode == "strict"
        else balanced_threshold
    )

    predictions = (
        probabilities >= selected_threshold
    ).astype(int)

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


st.set_page_config(
    page_title="LinkSentry — Phishing URL Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --background: #0b0d12;
        --surface: #141820;
        --surface-soft: #1a1f29;
        --border: #2a303c;
        --text: #f7f8fa;
        --muted: #98a2b3;
        --accent: #f5a623;
        --accent-hover: #ffb63f;
        --success: #38c985;
        --warning: #f4c95d;
        --danger: #ff5263;
    }

    html,
    body,
    [class*="css"] {
        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .stApp {
        background:
            linear-gradient(
                145deg,
                #090b0f 0%,
                #0d1118 52%,
                #10151d 100%
            );
        color: var(--text);
        min-height: 100vh;
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
        max-width: 1160px;
        padding: 3.2rem 1.25rem 4rem;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 22px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 28px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-badge {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 11px;
        background: var(--accent);
        color: #171109;
        font-size: 20px;
        box-shadow: 0 10px 30px rgba(245, 166, 35, 0.18);
    }

    .brand-name {
        margin: 0;
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .brand-caption {
        margin: 2px 0 0;
        color: var(--muted);
        font-size: 0.76rem;
    }

    .top-status {
        color: var(--muted);
        font-size: 0.82rem;
        border: 1px solid var(--border);
        padding: 8px 12px;
        border-radius: 9px;
        background: rgba(255, 255, 255, 0.02);
    }

    .workspace {
        background: rgba(20, 24, 32, 0.94);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 34px;
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.32);
    }

    .intro-label {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .intro-title {
        margin: 0;
        color: var(--text);
        font-size: clamp(2.1rem, 4vw, 4rem);
        line-height: 1.02;
        letter-spacing: -0.055em;
        font-weight: 850;
        max-width: 720px;
    }

    .intro-text {
        max-width: 670px;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.75;
        margin: 18px 0 28px;
    }

    .feature-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 0 0 30px;
    }

    .feature {
        min-height: 94px;
        padding: 17px;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface-soft);
    }

    .feature-number {
        color: var(--accent);
        font-size: 0.74rem;
        font-weight: 800;
    }

    .feature-title {
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 8px;
    }

    .feature-copy {
        color: var(--muted);
        font-size: 0.76rem;
        line-height: 1.45;
        margin-top: 5px;
    }

    .section-heading {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 750;
        margin: 4px 0 5px;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.84rem;
        margin: 0 0 14px;
    }

    div[data-testid="stTextInput"] input {
        min-height: 53px;
        color: var(--text);
        background: #0f131a;
        border: 1px solid var(--border);
        border-radius: 11px;
        padding-left: 15px;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(245, 166, 35, 0.13);
    }

    div[data-testid="stSelectbox"] > div > div {
        min-height: 53px;
        color: var(--text);
        background: #0f131a;
        border: 1px solid var(--border);
        border-radius: 11px;
    }

    .stButton > button {
        width: 100%;
        min-height: 53px;
        border: 0;
        border-radius: 11px;
        background: var(--accent);
        color: #171109;
        font-weight: 850;
        transition:
            transform 0.16s ease,
            background 0.16s ease;
    }

    .stButton > button:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
        color: #171109;
    }

    .result-card {
        margin-top: 20px;
        padding: 20px;
        border: 1px solid var(--border);
        border-radius: 15px;
        background: #10141b;
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 15px;
        margin-bottom: 15px;
    }

    .result-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 7px 11px;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 800;
    }

    .status-legitimate {
        color: #a7f3d0;
        background: rgba(56, 201, 133, 0.13);
        border: 1px solid rgba(56, 201, 133, 0.25);
    }

    .status-review {
        color: #ffe7a4;
        background: rgba(244, 201, 93, 0.13);
        border: 1px solid rgba(244, 201, 93, 0.25);
    }

    .status-phishing {
        color: #ffc0c7;
        background: rgba(255, 82, 99, 0.13);
        border: 1px solid rgba(255, 82, 99, 0.25);
    }

    .probability-value {
        color: var(--text);
        font-size: 2.35rem;
        line-height: 1;
        font-weight: 850;
        letter-spacing: -0.04em;
    }

    .probability-caption {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 6px;
    }

    .meter {
        width: 100%;
        height: 9px;
        border-radius: 5px;
        background: #252b35;
        overflow: hidden;
        margin-top: 15px;
    }

    .meter-fill {
        height: 100%;
        border-radius: inherit;
        background: var(--accent);
    }

    .cleaned-url {
        margin-top: 14px;
        padding: 12px 13px;
        color: #cdd5df;
        background: #0b0e13;
        border: 1px solid #242a34;
        border-radius: 9px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }

    div[data-testid="stExpander"] {
        margin-top: 24px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    div[data-testid="stFileUploader"] {
        background: #0f131a;
        border-radius: 11px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .footer-note {
        color: #667085;
        text-align: center;
        font-size: 0.75rem;
        margin-top: 24px;
    }

    @media (max-width: 760px) {
        .block-container {
            padding-top: 1.5rem;
        }

        .workspace {
            padding: 22px 17px;
            border-radius: 17px;
        }

        .feature-row {
            grid-template-columns: 1fr;
        }

        .top-status {
            display: none;
        }

        .intro-title {
            font-size: 2.35rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="brand-badge">⌁</div>
            <div>
                <p class="brand-name">LinkSentry</p>
                <p class="brand-caption">LSTM phishing URL detection</p>
            </div>
        </div>
        <div class="top-status">Model ready · Pure character LSTM</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, thresholds = load_artifacts()
except Exception:
    st.error(
        "Model gagal dimuat. Ganti file model lama dengan paket "
        "yang sudah diperbaiki dan pastikan struktur folder GitHub benar."
    )
    st.stop()

st.markdown('<div class="workspace">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="intro-label">URL security analysis</div>
    <h1 class="intro-title">Periksa tautan sebelum kamu mempercayainya.</h1>
    <p class="intro-text">
        Model character-level LSTM menganalisis urutan karakter pada URL
        untuk memperkirakan apakah sebuah tautan legitimate, perlu ditinjau,
        atau terindikasi phishing.
    </p>

    <div class="feature-row">
        <div class="feature">
            <div class="feature-number">01</div>
            <div class="feature-title">Character analysis</div>
            <div class="feature-copy">
                Membaca struktur domain, path, query, angka, dan simbol.
            </div>
        </div>
        <div class="feature">
            <div class="feature-number">02</div>
            <div class="feature-title">Strict threshold</div>
            <div class="feature-copy">
                Mode ketat membantu mengurangi false positive.
            </div>
        </div>
        <div class="feature">
            <div class="feature-number">03</div>
            <div class="feature-title">Batch inspection</div>
            <div class="feature-copy">
                Periksa banyak URL sekaligus melalui file CSV.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns(
    [1.85, 0.85],
    gap="medium",
)

with left_col:
    st.markdown(
        """
        <div class="section-heading">URL yang akan diperiksa</div>
        <div class="section-copy">
            Tempel URL lengkap, misalnya https://example.com/login.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("single_url_form"):
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com/login",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Analisis URL",
            use_container_width=True,
        )

with right_col:
    st.markdown(
        """
        <div class="section-heading">Mode keputusan</div>
        <div class="section-copy">
            Strict disarankan untuk mengurangi salah deteksi.
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold_mode = st.selectbox(
        "Mode",
        options=["strict", "balanced"],
        format_func=lambda mode: (
            "Strict · false positive lebih rendah"
            if mode == "strict"
            else "Balanced · recall lebih tinggi"
        ),
        label_visibility="collapsed",
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

        probability = float(
            result.loc[0, "Phishing Probability"]
        )
        status = str(result.loc[0, "Status"])
        cleaned_url = str(result.loc[0, "Cleaned URL"])

        status_class = {
            "Legitimate": "status-legitimate",
            "Perlu ditinjau": "status-review",
            "Phishing": "status-phishing",
        }[status]

        safe_cleaned_url = (
            cleaned_url
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-header">
                    <div>
                        <div class="result-label">Hasil analisis</div>
                    </div>
                    <span class="status-pill {status_class}">
                        {status}
                    </span>
                </div>

                <div class="probability-value">
                    {probability:.2%}
                </div>
                <div class="probability-caption">
                    Probabilitas URL termasuk phishing
                </div>

                <div class="meter">
                    <div
                        class="meter-fill"
                        style="width: {probability * 100:.2f}%;">
                    </div>
                </div>

                <div class="cleaned-url">
                    {safe_cleaned_url}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Periksa banyak URL melalui CSV"):
    st.caption(
        "Gunakan file CSV yang mempunyai kolom bernama `url`."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"CSV tidak dapat dibaca: {error}")
        else:
            if "url" not in batch_data.columns:
                st.error(
                    "CSV harus mempunyai kolom bernama `url`."
                )
            else:
                st.write(
                    f"Jumlah URL yang ditemukan: {len(batch_data):,}"
                )

                if st.button(
                    "Jalankan analisis batch",
                    use_container_width=True,
                ):
                    with st.spinner("Menganalisis seluruh URL..."):
                        batch_result = predict_urls(
                            batch_data["url"]
                            .fillna("")
                            .astype(str)
                            .tolist(),
                            threshold_mode=threshold_mode,
                        )

                        final_result = pd.concat(
                            [
                                batch_data.reset_index(drop=True),
                                batch_result[
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
                        "Download hasil CSV",
                        data=final_result.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="phishing_url_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

st.markdown(
    """
    <div class="footer-note">
        Prediksi model bukan pengganti pemeriksaan keamanan manual
        atau layanan threat intelligence.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
