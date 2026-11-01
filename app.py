# =====================================================================
# Mobile Price Prediction - Streamlit deployment
# Applied Machine Learning (S1-26_DSEZG568) Assignment
# Run with:  streamlit run app.py
# =====================================================================
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "mobile_price_model.pkl"

st.set_page_config(page_title="Mobile Price Range Predictor",
                   page_icon="M", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown('''
<style>
  .main-title    { font-size: 2.0rem; font-weight: 700; margin-bottom: .1rem; }
  .sub-title     { color: #6b7280; margin-bottom: 1.2rem; }
  .result-card   { padding: 1.3rem 1.5rem; border-radius: 12px; color: #fff;
                   text-align: center; margin-bottom: .8rem; }
  .result-band   { font-size: 2.1rem; font-weight: 700; }
  .result-sub    { font-size: 1.0rem; opacity: .9; }
  .metric-box    { background: #f1f5f9; border-radius: 10px; padding: .8rem 1rem;
                   border-left: 4px solid #2563eb; }
</style>
''', unsafe_allow_html=True)

BAND_COLOURS = {0: "#0ea5e9", 1: "#10b981", 2: "#f59e0b", 3: "#ef4444"}


@st.cache_resource
def load_bundle():
    # Load the artefact persisted by the notebook.
    return joblib.load(MODEL_PATH)


def add_engineered_features(df):
    # MUST match Section 4.6 of the notebook exactly.
    out = df.copy()
    out["px_area"] = out["px_height"] * out["px_width"]
    out["screen_area"] = out["sc_h"] * out["sc_w"]
    diag_px = np.sqrt(out["px_height"] ** 2 + out["px_width"] ** 2)
    diag_in = np.sqrt(out["sc_h"] ** 2 + out["sc_w"] ** 2) / 2.54
    out["ppi"] = diag_px / diag_in.replace(0, np.nan)
    out["ppi"] = out["ppi"].fillna(out["ppi"].median() if out["ppi"].notna().any() else 0)
    out["total_camera"] = out["fc"] + out["pc"]
    out["ram_per_core"] = out["ram"] / out["n_cores"]
    out["battery_per_talk_time"] = out["battery_power"] / out["talk_time"]
    return out


def preprocess(raw_df, bundle):
    # Raw specifications -> the exact feature matrix the model was trained on.
    df = raw_df.copy()
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith("Unnamed") or c in ("id", "price_range")],
                 errors="ignore")
    # NOTE: do not test for dtype == object here. Pandas 3 gives string columns the
    # dedicated "str" dtype, so an object check silently skips the mapping and turns
    # every Yes/No flag into 0. Test for "not numeric" instead - correct on all versions.
    for c in bundle["binary_cols"]:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            s = (s.astype(str).str.strip().str.lower()
                 .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}))
        df[c] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    wt = df["mobile_wt"]
    if not pd.api.types.is_numeric_dtype(wt):
        lut = {str(k).strip().lower(): v for k, v in bundle["wt_order"].items()}
        wt = wt.astype(str).str.strip().str.lower().map(lut)
    df["mobile_wt"] = pd.to_numeric(wt, errors="coerce").fillna(1).astype(int)
    for c, med in bundle["impute_medians"].items():
        df[c] = df[c].replace(0, np.nan).fillna(med)
    df = add_engineered_features(df)
    return df[bundle["feature_order"]]


bundle = load_bundle()
model = bundle["model"]
labels = {int(k): v for k, v in bundle["class_labels"].items()}

st.markdown('<div class="main-title">Mobile Price Range Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Applied Machine Learning (S1-26_DSEZG568) &nbsp;|&nbsp; '
    'Model in production: <b>' + bundle["model_name"] + '</b> &nbsp;|&nbsp; '
    'Validation accuracy: <b>' + f'{bundle["metrics"]["Val accuracy"]:.2%}' + '</b></div>',
    unsafe_allow_html=True)

# ----------------------------------------------------------------- sidebar input form
st.sidebar.header("Phone specifications")
rng = bundle["train_ranges"]


def slider(label, key, step=1, default=None):
    lo, hi = rng[key]
    default = (lo + hi) / 2 if default is None else default
    if step == 1:
        return st.sidebar.slider(label, int(lo), int(hi), int(default), 1)
    return st.sidebar.slider(label, float(lo), float(hi), float(default), step)


with st.sidebar:
    st.caption("Performance")
    ram = slider("RAM (MB)", "ram", 1, 2100)
    n_cores = slider("Processor cores", "n_cores", 1, 4)
    clock_speed = slider("Clock speed (GHz)", "clock_speed", 0.1, 1.5)
    int_memory = slider("Internal memory (GB)", "int_memory", 1, 32)

    st.caption("Display")
    px_height = slider("Pixel height", "px_height", 1, 640)
    px_width = slider("Pixel width", "px_width", 1, 1250)
    sc_h = slider("Screen height (cm)", "sc_h", 1, 12)
    sc_w = slider("Screen width (cm)", "sc_w", 1, 6)

    st.caption("Battery and body")
    battery_power = slider("Battery power (mAh)", "battery_power", 1, 1240)
    talk_time = slider("Talk time (hours)", "talk_time", 1, 11)
    m_dep = slider("Mobile depth (cm)", "m_dep", 0.1, 0.5)
    mobile_wt = st.select_slider("Weight band", options=["Low", "Med", "High"], value="Med")

    st.caption("Camera")
    pc = slider("Primary camera (MP)", "pc", 1, 10)
    fc = slider("Front camera (MP)", "fc", 1, 4)

    st.caption("Connectivity")
    c1, c2 = st.columns(2)
    blue = c1.checkbox("Bluetooth", True)
    wifi = c2.checkbox("Wi-Fi", True)
    dual_sim = c1.checkbox("Dual SIM", True)
    touch_screen = c2.checkbox("Touch screen", True)
    three_g = c1.checkbox("3G", True)
    four_g = c2.checkbox("4G", True)

user_row = pd.DataFrame([{
    "battery_power": battery_power, "blue": "Yes" if blue else "No",
    "clock_speed": clock_speed, "dual_sim": "Yes" if dual_sim else "No",
    "fc": fc, "four_g": int(four_g), "int_memory": int_memory, "m_dep": m_dep,
    "mobile_wt": mobile_wt, "n_cores": n_cores, "pc": pc, "px_height": px_height,
    "px_width": px_width, "ram": ram, "sc_h": sc_h, "sc_w": sc_w,
    "talk_time": talk_time, "three_g": int(three_g),
    "touch_screen": "Yes" if touch_screen else "No", "wifi": "Yes" if wifi else "No",
}])

tab1, tab2, tab3 = st.tabs(["Single prediction", "Batch prediction (CSV)", "Model insight"])

# ----------------------------------------------------------------- tab 1
with tab1:
    left, right = st.columns([1, 1.15])

    with left:
        st.subheader("Entered specification")
        st.dataframe(user_row.T.rename(columns={0: "value"}).astype(str), height=430)

    with right:
        st.subheader("Prediction")
        if st.button("Predict price range", type="primary"):
            X = preprocess(user_row, bundle)
            pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]

            st.markdown(
                '<div class="result-card" style="background:' + BAND_COLOURS[pred] + ';">'
                '<div class="result-sub">Predicted price range</div>'
                '<div class="result-band">' + str(pred) + ' &mdash; ' + labels[pred] + '</div>'
                '<div class="result-sub">confidence ' + f'{proba[pred]:.1%}' + '</div></div>',
                unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                "Price range": [f"{i} - {labels[i]}" for i in range(len(proba))],
                "Probability": proba,
            }).set_index("Price range")
            st.bar_chart(prob_df, height=230)
            st.dataframe(prob_df.style.format("{:.2%}"))

            drivers = list(bundle["feature_importance"].items())[:5]
            st.caption("Top drivers of this model: " +
                       ", ".join(f"{k}" for k, _ in drivers))
        else:
            st.info("Set the specifications in the sidebar, then press "
                    "**Predict price range**.")

# ----------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Score a whole file")
    st.caption("Upload a CSV with the same columns as "
               "Mobile_Price_Prediction_test.csv (the price_range column is optional).")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        st.write(f"Loaded **{raw.shape[0]}** rows.")
        try:
            Xb = preprocess(raw, bundle)
            preds = model.predict(Xb)
            probs = model.predict_proba(Xb)
            out = raw.copy()
            out["predicted_price_range"] = preds
            out["predicted_label"] = [labels[int(p)] for p in preds]
            out["confidence"] = probs.max(axis=1).round(4)

            m1, m2, m3 = st.columns(3)
            m1.metric("Rows scored", len(out))
            m2.metric("Mean confidence", f"{out['confidence'].mean():.1%}")
            m3.metric("Most common band", labels[int(pd.Series(preds).mode()[0])])

            st.dataframe(out.head(50))
            st.bar_chart(pd.Series(preds).value_counts().sort_index())

            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Download predictions as CSV", buf.getvalue(),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not score this file: {exc}")

# ----------------------------------------------------------------- tab 3
with tab3:
    st.subheader("How this model was chosen")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Validation accuracy of every model trained**")
        scores = pd.Series(bundle["all_model_scores"], name="Val accuracy").sort_values()
        st.bar_chart(scores)
        st.dataframe(scores.to_frame().style.format("{:.4f}"))
    with c2:
        st.markdown("**Metrics of the deployed model**")
        st.dataframe(pd.Series(bundle["metrics"], name="value").to_frame()
                     .style.format("{:.4f}"))

    st.markdown("**Permutation feature importance** (measured on unseen data)")
    fi = pd.Series(bundle["feature_importance"]).sort_values(ascending=False).head(12)
    st.bar_chart(fi)
    st.caption("Model artefact: " + MODEL_PATH + "  |  scikit-learn " +
               bundle["sklearn_version"])
