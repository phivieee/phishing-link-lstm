import html
import json
import re
from pathlib import Path
from textwrap import dedent

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
    """Apply the same normalization used during training."""
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


def predict_urls(urls: list[str]) -> pd.DataFrame:
    """Predict URLs using the strict deployment threshold."""
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

    predictions = (
        probabilities >= strict_threshold
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
            "Threshold Used": strict_threshold,
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
        max-width: 1120px;
        padding: 2.6rem 1.25rem 4rem;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 22px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 44px;
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

    .intro-label {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .intro-title {
        margin: 0 0 34px;
        color: var(--text);
        font-size: clamp(2.3rem, 4.6vw, 4.35rem);
        line-height: 1.02;
        letter-spacing: -0.055em;
        font-weight: 850;
        max-width: 760px;
    }

    .section-heading {
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 780;
        margin: 0 0 5px;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.55;
        margin: 0 0 16px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(20, 24, 32, 0.92);
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.20);
    }

    div[data-testid="stTextInput"] input {
        min-height: 54px;
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

    .stButton > button,
    .stFormSubmitButton > button {
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

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: var(--accent-hover);
        transform: translateY(-1px);
        color: #171109;
    }

    .result-card {
        margin-top: 22px;
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

    div[data-testid="stFileUploader"] {
        background: #0f131a;
        border-radius: 11px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .batch-note {
        margin: 0 0 16px;
        padding: 13px 14px;
        color: #cbd3df;
        background: #10141b;
        border: 1px solid #282f3a;
        border-radius: 10px;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .footer-note {
        color: #667085;
        text-align: center;
        font-size: 0.75rem;
        margin-top: 24px;
    }

    @media (max-width: 760px) {
        .block-container {
            padding-top: 1.4rem;
        }

        .top-status {
            display: none;
        }

        .intro-title {
            font-size: 2.5rem;
            margin-bottom: 26px;
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
        "Model gagal dimuat. Pastikan folder `model` berisi "
        "`pure_lstm_model.keras` dan `thresholds.json`."
    )
    st.stop()

st.markdown(
    """
    <div class="intro-label">URL security analysis</div>
    <h1 class="intro-title">
        Periksa tautan sebelum kamu mempercayainya.
    </h1>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        """
        <div class="section-heading">Periksa satu URL</div>
        <div class="section-copy">
            Tempel URL lengkap yang ingin dianalisis.
            Sistem otomatis menggunakan strict threshold untuk
            mengurangi false positive.
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

    if submitted:
        if not url_input.strip():
            st.warning("Masukkan URL terlebih dahulu.")
        else:
            with st.spinner("Menganalisis pola URL..."):
                result = predict_urls([url_input])

            probability = float(
                result.loc[0, "Phishing Probability"]
            )
            status = str(result.loc[0, "Status"])
            cleaned_url = html.escape(
                str(result.loc[0, "Cleaned URL"])
            )

            status_class = {
                "Legitimate": "status-legitimate",
                "Perlu ditinjau": "status-review",
                "Phishing": "status-phishing",
            }[status]

            result_html = dedent(
                f"""
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-label">Hasil analisis</div>
                        <span class="status-pill {status_class}">
                            {status}
                        </span>
                    </div>
                    <div class="probability-value">{probability:.2%}</div>
                    <div class="probability-caption">
                        Probabilitas URL termasuk phishing
                    </div>
                    <div class="meter">
                        <div
                            class="meter-fill"
                            style="width: {probability * 100:.2f}%;">
                        </div>
                    </div>
                    <div class="cleaned-url">{cleaned_url}</div>
                </div>
                """
            ).strip()

            st.markdown(
                result_html,
                unsafe_allow_html=True,
            )

with st.container(border=True):
    st.markdown(
        """
        <div class="section-heading">Periksa URL sekaligus</div>
        <div class="section-copy">
            Unggah file CSV untuk menganalisis banyak tautan dalam
            satu proses.
        </div>
        <div class="batch-note">
            File harus memiliki kolom bernama <strong>url</strong>.
            Setiap URL akan diperiksa menggunakan strict threshold,
            lalu hasilnya dapat diunduh kembali sebagai CSV.
        </div>
        """,
        unsafe_allow_html=True,
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
                            .tolist()
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
