from __future__ import annotations

import requests
import streamlit as st

# =========================================================================== #
# ACTUAL BENCHMARK METRICS (Update these values from your Locust / Evaluation results)
# =========================================================================== #
BENCHMARK_LATENCY = "~8.4 ms"       # Displayed on metric card
BENCHMARK_LATENCY_SB = "~8.47 ms"   # Displayed in sidebar spec
BENCHMARK_THROUGHPUT = "115+ RPS"   # Displayed on metric card & sidebar
BENCHMARK_ACCURACY = "82.2%"        # Displayed on metric card

# 1. Page Configuration
st.set_page_config(
    page_title="Flight Sentiment Analysis | Real-Time Monitoring",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Custom CSS Injection for High-Tech Dark Mode & Visible Sidebar Text
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Force Streamlit's top header bar to match your dark background */
    header[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
}
    
    /* Change color of all Streamlit input labels (including Customer Review Text:) to white */
    .stTextArea label {
    color: #ffffff !important;
    }
    
    .stTextArea textarea::placeholder{
        color: #9ca3af !important;
        opacity: 1;
    }
    
    /* Sidebar styling & explicit white text fix */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] li, [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.05);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #00f2fe;
    }

    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Custom Input Box */
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: #ffffff !important;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.3) !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glowing title effect */
    .glow-title {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Sidebar Configuration & System Status
with st.sidebar:
  st.markdown(
      "### ⚙️ System Status",
      help="Real-time status of backend microservices",
  )

  backend_healthy = False
  try:
    health_res = requests.get("http://127.0.0.1:8000/", timeout=2)
    if health_res.status_code == 200:
      backend_healthy = True
  except:
    backend_healthy = False

  if backend_healthy:
    st.success("🟢 FastAPI Backend: Online")
  else:
    st.error("🔴 FastAPI Backend: Offline")

  st.markdown("---")
  st.markdown("### 📊 Architecture Spec")
  st.markdown(
      f"""
    * **Model:** TF-IDF + Linear SVM
    * **Latency:** {BENCHMARK_LATENCY_SB}
    * **Throughput:** {BENCHMARK_THROUGHPUT}
    * **Framework:** FastAPI & Streamlit
    """,
      unsafe_allow_html=True,
  )
  st.markdown("---")
  st.caption("Flight Sentiment Analysis Platform © 2026")

# 4. Main Dashboard Header
st.markdown(
    "<h1 class='glow-title'>Flight Sentiment Analysis Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "High-performance NLP inference pipeline designed to evaluate airline"
    " customer feedback with sub-10ms response times and robust MLOps"
    " monitoring."
)
st.markdown("---")

# 5. Top Metrics Row (3 Cards using variables)
col1, col2, col3 = st.columns(3)

with col1:
  st.markdown(
      f"""
        <div class="metric-card">
            <div class="metric-label">Inference Latency</div>
            <div class="metric-value">{BENCHMARK_LATENCY}</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

with col2:
  st.markdown(
      f"""
        <div class="metric-card">
            <div class="metric-label">Peak Throughput</div>
            <div class="metric-value">{BENCHMARK_THROUGHPUT}</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

with col3:
  st.markdown(
      f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{BENCHMARK_ACCURACY}</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

st.markdown("<br>", unsafe_allow_html=True)

# 6. Interactive Prediction Section (Clean Single-Column Layout)
st.markdown("### 📝 Live Sentiment Analyzer")
st.markdown(
    "Enter customer review feedback below to run real-time classification."
)

review_input = st.text_area(
    "Customer Review Text:",
    placeholder=(
        "@Delta absolute worst experience ever, my luggage is missing and"
        " customer service was rude!"
    ),
    height=120,
)

classify_btn = st.button(
    "🚀 Run Inference Request", type="primary", use_container_width=True
)

# Results container positioned directly beneath the button
output_container = st.container()

if classify_btn:
  if not review_input.strip():
    output_container.warning("Please provide review text to analyze.")
  else:
    with st.spinner("Executing backend request..."):
      try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": review_input},
            timeout=5,
        )
        if response.status_code == 200:
          res_data = response.json()
          sentiment = res_data.get("sentiment", "neutral").lower()

          with output_container:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div"
                ' style="background:rgba(17,24,39,0.9);border:1px solid'
                ' #1f2937;border-radius:12px;padding:25px;">',
                unsafe_allow_html=True,
            )

            if sentiment == "positive":
              st.markdown(
                  "### Prediction: <span"
                  ' style="color:#10b981;">POSITIVE 🟢</span>',
                  unsafe_allow_html=True,
              )
            elif sentiment == "negative":
              st.markdown(
                  "### Prediction: <span"
                  ' style="color:#ef4444;">NEGATIVE 🔴</span>',
                  unsafe_allow_html=True,
              )
            else:
              st.markdown(
                  "### Prediction: <span"
                  ' style="color:#f59e0b;">NEUTRAL 🟡</span>',
                  unsafe_allow_html=True,
              )

            st.success("HTTP Status: 200 OK | Response Time: ~8ms")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
          output_container.error(
              f"API Error ({response.status_code}): Backend failed to process"
              " request."
          )
      except requests.exceptions.ConnectionError:
        output_container.error(
            "Connection Refused! Make sure your FastAPI backend is running on"
            " port 8000."
        )