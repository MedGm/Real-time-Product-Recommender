"""
Streamlit dashboard — Real-time Product Recommender (Big Data Pipeline)
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Big Data Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #cba6f7; }
    .metric-card .label { font-size: 0.85rem; color: #a6adc8; margin-top: 0.2rem; }
    .rec-item {
        display: flex; align-items: center; gap: 0.8rem;
        padding: 0.6rem 1rem; margin: 0.3rem 0;
        background: #181825; border-radius: 8px;
        border-left: 3px solid #cba6f7;
    }
    .rec-rank { font-size: 0.8rem; color: #6c7086; min-width: 1.5rem; }
    .rec-product { font-family: monospace; font-size: 0.95rem; color: #cdd6f4; }
    .pipeline-step {
        background: #1e1e2e; border: 1px solid #313244;
        border-radius: 10px; padding: 1rem;
        text-align: center;
    }
    .pipeline-step .icon { font-size: 2rem; }
    .pipeline-step .name { font-size: 0.9rem; color: #cdd6f4; margin-top: 0.4rem; }
    .pipeline-step .desc { font-size: 0.75rem; color: #6c7086; margin-top: 0.2rem; }
    .arrow { display: flex; align-items: center; justify-content: center;
             color: #6c7086; font-size: 1.5rem; padding: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=5)
        if r.status_code == 404:
            return None, r.json().get("detail", "Not found.")
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API."
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Big Data Recommender")
    st.markdown("*Amazon Fine Food Reviews · ALS Model*")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Recommendations", "📊 Model Metrics"],
        label_visibility="collapsed",
    )
    st.divider()
    stats, _ = api_get("/stats")
    if stats:
        st.markdown(f"**{stats['users_with_recommendations']:,}** users indexed")
        st.markdown(f"**{stats['total_recommendation_rows']:,}** recommendations stored")
    health, _ = api_get("/health")
    status_color = "🟢" if health else "🔴"
    st.markdown(f"{status_color} API {'online' if health else 'offline'}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Real-Time Product Recommendation System")
    st.markdown("End-to-end big data pipeline: Kafka → Spark ALS → PostgreSQL → FastAPI → Streamlit")
    st.divider()

    # Pipeline diagram
    cols = st.columns([2, 0.4, 2, 0.4, 2, 0.4, 2, 0.4, 2])
    steps = [
        ("📨", "Kafka", "Streams CSV reviews"),
        ("⚡", "Spark ALS", "Trains recommender"),
        ("🌊", "Streaming", "Real-time scoring"),
        ("🗄️", "PostgreSQL", "Stores results"),
        ("🌐", "FastAPI", "Serves predictions"),
    ]
    for i, (icon, name, desc) in enumerate(steps):
        with cols[i * 2]:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="icon">{icon}</div>
                <div class="name"><b>{name}</b></div>
                <div class="desc">{desc}</div>
            </div>""", unsafe_allow_html=True)
        if i < len(steps) - 1:
            with cols[i * 2 + 1]:
                st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

    st.divider()

    # Key metrics
    stats, err = api_get("/stats")
    metrics, _ = api_get("/metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = f"{stats['users_with_recommendations']:,}" if stats else "–"
        st.markdown(f'<div class="metric-card"><div class="value">{v}</div><div class="label">Users with Recommendations</div></div>', unsafe_allow_html=True)
    with c2:
        v = f"{stats['total_recommendation_rows']:,}" if stats else "–"
        st.markdown(f'<div class="metric-card"><div class="value">{v}</div><div class="label">Total Recommendation Rows</div></div>', unsafe_allow_html=True)
    with c3:
        v = f"{metrics['val_rmse']:.4f}" if metrics else "–"
        st.markdown(f'<div class="metric-card"><div class="value">{v}</div><div class="label">Validation RMSE</div></div>', unsafe_allow_html=True)
    with c4:
        v = f"{metrics['test_rmse']:.4f}" if metrics else "–"
        st.markdown(f'<div class="metric-card"><div class="value">{v}</div><div class="label">Test RMSE (held-out)</div></div>', unsafe_allow_html=True)

    st.divider()

    # Quick try
    st.subheader("Quick Lookup")
    users_data, _ = api_get("/users", {"limit": 500})
    if users_data:
        sample_user = st.selectbox(
            "Pick a user from the index",
            options=users_data["users"],
            index=0,
        )
        if st.button("Get Recommendations", type="primary"):
            recs, err = api_get(f"/recommendations/user/{sample_user}", {"n": 10})
            if recs:
                st.success(f"Top 10 recommendations for **{sample_user}**")
                half = len(recs["recommendations"]) // 2 + len(recs["recommendations"]) % 2
                col_a, col_b = st.columns(2)
                for i, p in enumerate(recs["recommendations"]):
                    target = col_a if i < half else col_b
                    target.markdown(
                        f'<div class="rec-item"><span class="rec-rank">#{i+1}</span>'
                        f'<span class="rec-product">{p}</span></div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Recommendations":
    st.title("🔍 Product Recommendations")
    st.markdown("Look up personalised recommendations for any user in the index.")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        user_id = st.text_input("User ID", placeholder="e.g. A3SGXH7AUHU8GW")
    with col_right:
        n = st.number_input("Top N", min_value=1, max_value=20, value=10)

    col_btn, col_browse = st.columns([1, 3])
    with col_btn:
        search = st.button("Get Recommendations", type="primary", use_container_width=True)

    if search and user_id:
        with st.spinner("Fetching recommendations..."):
            data, err = api_get(f"/recommendations/user/{user_id}", {"n": n})
        if err:
            st.error(f"❌ {err}")
        else:
            st.success(f"Top {n} recommendations for **{user_id}**")
            st.divider()
            half = len(data["recommendations"]) // 2 + len(data["recommendations"]) % 2
            col_a, col_b = st.columns(2)
            for i, product in enumerate(data["recommendations"]):
                target = col_a if i < half else col_b
                target.markdown(
                    f'<div class="rec-item">'
                    f'<span class="rec-rank">#{i+1}</span>'
                    f'<span class="rec-product">{product}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with st.expander("Raw JSON"):
                st.json(data)

    st.divider()
    with st.expander("Browse known users (up to 200)"):
        data, err = api_get("/users", {"limit": 200})
        if data:
            import pandas as pd
            df = pd.DataFrame({"User ID": data["users"]})
            st.dataframe(df, use_container_width=True, height=300)
        elif err:
            st.warning(err)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Metrics
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.title("📊 ALS Model Metrics")
    st.markdown("Performance of the trained Alternating Least Squares recommendation model.")

    data, err = api_get("/metrics")
    if err:
        st.warning(f"⚠️ {err}")
        st.info("The model metrics file (`/model/metrics.json`) is written by the Spark training job. Trigger the `recommendation_pipeline` DAG in Airflow to generate it.")
    else:
        # RMSE cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{data['val_rmse']:.4f}</div>
                <div class="label">Validation RMSE (hyperparameter tuning)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{data['test_rmse']:.4f}</div>
                <div class="label">Test RMSE (held-out 10% — never seen during training)</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # RMSE bar chart
        import pandas as pd
        chart_df = pd.DataFrame({
            "Split": ["Validation", "Test (held-out)"],
            "RMSE":  [data["val_rmse"], data["test_rmse"]],
        })
        st.subheader("RMSE Comparison")
        st.bar_chart(chart_df.set_index("Split"), height=280, color="#cba6f7")

        st.divider()

        # Best hyperparameters
        st.subheader("Best Hyperparameters")
        params = data["best_params"]
        p1, p2, p3 = st.columns(3)
        p1.metric("Rank (latent factors)", params.get("rank", "–"))
        p2.metric("Regularisation (λ)", params.get("regParam", "–"))
        p3.metric("Max Iterations", params.get("maxIter", "–"))

        st.divider()
        st.subheader("Data Split")
        split_df = pd.DataFrame({
            "Split": ["Training", "Validation", "Test"],
            "Proportion": [80, 10, 10],
        })
        st.bar_chart(split_df.set_index("Split"), height=220)
        st.caption(
            "80 / 10 / 10 split. Training data was filtered to users and products "
            "with ≥ 5 ratings each. ALS `coldStartStrategy='drop'` prevents NaN RMSE "
            "on unseen users/items."
        )
