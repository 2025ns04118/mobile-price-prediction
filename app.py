# =====================================================================
# Mobile Price Range Workbench - Streamlit deployment
# Applied Machine Learning (S1-26_DSEZG568) Assignment
# Run with:  streamlit run app.py
# =====================================================================
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "mobile_price_model.pkl"

st.set_page_config(page_title="Mobile Price Range Workbench",
                   page_icon="📱", layout="wide")

# ----------------------------------------------------------------- styling
st.markdown('''
<style>
  .block-container { padding-top: 1.6rem; max-width: 1250px; }
  .hero { background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #0ea5e9 100%);
          border-radius: 16px; padding: 1.5rem 1.8rem; color: #fff; margin-bottom: 1.1rem; }
  .hero-title { font-size: 2rem; font-weight: 750; margin: 0; letter-spacing: -.5px; }
  .hero-sub   { opacity: .9; margin-top: .25rem; font-size: .98rem; }
  .pill { display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
          padding: .15rem .75rem; margin: .7rem .4rem 0 0; font-size: .82rem; }
  .section-label { font-size: .78rem; font-weight: 700; letter-spacing: .08em;
                   text-transform: uppercase; color: #64748b; margin: .2rem 0 .4rem 0; }
  .result-card { padding: 1.4rem 1.2rem; border-radius: 14px; color: #fff;
                 text-align: center; margin-bottom: .9rem;
                 box-shadow: 0 6px 18px rgba(0,0,0,.12); }
  .result-sub  { font-size: .95rem; opacity: .92; }
  .result-band { font-size: 2.1rem; font-weight: 750; line-height: 1.2; }
  .mini { background: #f1f5f9; border-radius: 10px; padding: .6rem .8rem;
          border-left: 4px solid #2563eb; }
  .mini-k { font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }
  .mini-v { font-size: 1.15rem; font-weight: 700; color: #0f172a; }
  @media (prefers-color-scheme: dark) {
    .mini { background: #1e293b; } .mini-v { color: #f1f5f9; }
  }
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
train_ranges = bundle["train_ranges"]

# ----------------------------------------------------------------- realistic input ranges
# Ranges are realistic for handsets AND stay within the span of the training data
# (the model has not seen e.g. 8 GB RAM, so values beyond it would be extrapolation).
MEMORY_OPTIONS = [2, 4, 8, 16, 32, 64, 128]

PRESETS = {
    "Budget": dict(ram=512, n_cores=2, clock_speed=1.0, int_memory=8,
                   px_height=600, px_width=700, sc_h=9, sc_w=5,
                   battery_power=800, talk_time=6.0, m_dep=0.7, mobile_wt="High",
                   pc=3, fc=1, blue=True, wifi=False, dual_sim=False,
                   touch_screen=False, three_g=True, four_g=False),
    "Mid-range": dict(ram=2048, n_cores=4, clock_speed=1.5, int_memory=32,
                      px_height=800, px_width=1200, sc_h=12, sc_w=6,
                      battery_power=1300, talk_time=12.0, m_dep=0.5, mobile_wt="Med",
                      pc=10, fc=5, blue=True, wifi=True, dual_sim=True,
                      touch_screen=True, three_g=True, four_g=True),
    "Flagship": dict(ram=3584, n_cores=8, clock_speed=2.5, int_memory=64,
                     px_height=1500, px_width=1800, sc_h=15, sc_w=8,
                     battery_power=1800, talk_time=20.0, m_dep=0.3, mobile_wt="Low",
                     pc=18, fc=12, blue=True, wifi=True, dual_sim=True,
                     touch_screen=True, three_g=True, four_g=True),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


if "ram" not in st.session_state:
    apply_preset("Mid-range")

# ----------------------------------------------------------------- hero header
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">📱 Mobile Price Range Workbench</div>'
    '<div class="hero-sub">Enter a handset specification and get its predicted price band '
    'with class probabilities, or score a whole CSV at once.</div>'
    '<span class="pill">Model: ' + bundle["model_name"] + '</span>'
    '<span class="pill">Validation accuracy: ' + f'{bundle["metrics"]["Val accuracy"]:.2%}' + '</span>'
    '<span class="pill">Applied Machine Learning · S1-26_DSEZG568</span>'
    '</div>',
    unsafe_allow_html=True)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.write("A classifier that places a phone into one of four price bands "
             "from its hardware specifications.")
    st.markdown("**Price bands**")
    for i in sorted(labels):
        st.markdown(
            '<span style="display:inline-block;width:.7rem;height:.7rem;border-radius:50%;'
            'background:' + BAND_COLOURS.get(i, "#888") + ';margin-right:.5rem;"></span>'
            + str(i) + " — " + str(labels[i]), unsafe_allow_html=True)
    st.divider()
    st.markdown("**How to use**")
    st.caption("1. Pick a starting preset or set each spec.\n\n"
               "2. The prediction updates live on the right.\n\n"
               "3. Use the Batch tab to score a CSV file.")
    st.divider()
    st.caption("scikit-learn " + str(bundle["sklearn_version"]))

tab1, tab2, tab3 = st.tabs(["🎯 Single prediction", "📂 Batch prediction (CSV)",
                            "📊 Model insight"])

# ----------------------------------------------------------------- tab 1
with tab1:
    st.markdown('<div class="section-label">Quick start</div>', unsafe_allow_html=True)
    pcols = st.columns(len(PRESETS))
    for col, name in zip(pcols, PRESETS):
        col.button(name, key="preset_" + name, use_container_width=True,
                   on_click=apply_preset, args=(name,))

    form_col, result_col = st.columns([1.55, 1], gap="large")

    with form_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Performance</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            ram = a.slider("RAM (MB)", 256, 4096, step=128, key="ram",
                           help="Typical handsets: 256 MB (basic) to 4 GB (high end).")
            n_cores = b.slider("Processor cores", 1, 8, key="n_cores")
            clock_speed = a.slider("Clock speed (GHz)", 0.5, 3.0, step=0.1, key="clock_speed",
                                   help="Peak CPU frequency.")
            int_memory = b.select_slider("Internal memory (GB)", options=MEMORY_OPTIONS,
                                         key="int_memory")

        with st.container(border=True):
            st.markdown('<div class="section-label">Display</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            px_height = a.slider("Pixel height (px)", 240, 2000, step=10, key="px_height")
            px_width = b.slider("Pixel width (px)", 480, 2000, step=10, key="px_width")
            sc_h = a.slider("Screen height (cm)", 5, 19, step=1, key="sc_h",
                            help="Physical height of the screen, not the whole phone.")
            sc_w = b.slider("Screen width (cm)", 1, 18, step=1, key="sc_w")

        with st.container(border=True):
            st.markdown('<div class="section-label">Battery and body</div>',
                        unsafe_allow_html=True)
            a, b = st.columns(2)
            battery_power = a.slider("Battery capacity (mAh)", 500, 2000, step=50,
                                     key="battery_power")
            talk_time = b.slider("Talk time (hours)", 0.0, 24.0, step=0.5, key="talk_time",
                                 help="Continuous call time on a full charge. A value of 0 "
                                      "is evaluated as 0.5 h to avoid dividing by zero.")
            m_dep = a.slider("Thickness (cm)", 0.1, 1.0, step=0.05, key="m_dep")
            mobile_wt = b.select_slider("Weight band", options=["Low", "Med", "High"],
                                        key="mobile_wt")

        with st.container(border=True):
            st.markdown('<div class="section-label">Camera</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            pc = a.slider("Primary camera (MP)", 0, 20, key="pc")
            fc = b.slider("Front camera (MP)", 0, 20, key="fc")

        with st.container(border=True):
            st.markdown('<div class="section-label">Connectivity and features</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            blue = c1.toggle("Bluetooth", key="blue")
            wifi = c2.toggle("Wi-Fi", key="wifi")
            dual_sim = c3.toggle("Dual SIM", key="dual_sim")
            touch_screen = c1.toggle("Touch screen", key="touch_screen")
            three_g = c2.toggle("3G", key="three_g")
            four_g = c3.toggle("4G", key="four_g")

    user_row = pd.DataFrame([{
        "battery_power": battery_power, "blue": "Yes" if blue else "No",
        "clock_speed": clock_speed, "dual_sim": "Yes" if dual_sim else "No",
        "fc": fc, "four_g": int(four_g), "int_memory": int_memory, "m_dep": m_dep,
        "mobile_wt": mobile_wt, "n_cores": n_cores, "pc": pc, "px_height": px_height,
        "px_width": px_width, "ram": ram, "sc_h": sc_h, "sc_w": sc_w,
        "talk_time": max(talk_time, 0.5), "three_g": int(three_g),
        "touch_screen": "Yes" if touch_screen else "No", "wifi": "Yes" if wifi else "No",
    }])

    with result_col:
        st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
        X = preprocess(user_row, bundle)
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        st.markdown(
            '<div class="result-card" style="background:' + BAND_COLOURS.get(pred, "#2563eb") + ';">'
            '<div class="result-sub">Predicted price range</div>'
            '<div class="result-band">' + str(pred) + ' — ' + str(labels[pred]) + '</div>'
            '<div class="result-sub">confidence ' + f'{proba[pred]:.1%}' + '</div></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="section-label">Class probabilities</div>',
                    unsafe_allow_html=True)
        for i, p in enumerate(proba):
            st.progress(float(p), text=f"{i} — {labels[i]}   ·   {p:.1%}")

        derived = add_engineered_features(user_row.assign(
            mobile_wt=0, blue=0, dual_sim=0, touch_screen=0, wifi=0)).iloc[0]
        diag_in = float(np.sqrt(sc_h ** 2 + sc_w ** 2) / 2.54)
        st.markdown('<div class="section-label" style="margin-top:.8rem;">Derived specs</div>',
                    unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        d1.markdown('<div class="mini"><div class="mini-k">Screen size</div>'
                    '<div class="mini-v">' + f'{diag_in:.1f}"' + '</div></div>',
                    unsafe_allow_html=True)
        d2.markdown('<div class="mini"><div class="mini-k">Pixel density</div>'
                    '<div class="mini-v">' + f'{derived["ppi"]:.0f} ppi' + '</div></div>',
                    unsafe_allow_html=True)
        d3, d4 = st.columns(2)
        d3.markdown('<div class="mini" style="margin-top:.5rem;"><div class="mini-k">Resolution</div>'
                    '<div class="mini-v">' + f'{derived["px_area"] / 1e6:.2f} MP' + '</div></div>',
                    unsafe_allow_html=True)
        d4.markdown('<div class="mini" style="margin-top:.5rem;"><div class="mini-k">RAM per core</div>'
                    '<div class="mini-v">' + f'{derived["ram_per_core"]:.0f} MB' + '</div></div>',
                    unsafe_allow_html=True)

        # Warn when inputs lie outside the range the model was trained on.
        outside = []
        for k in train_ranges:
            if k in user_row.columns and pd.api.types.is_numeric_dtype(user_row[k]):
                lo, hi = train_ranges[k]
                v = float(user_row[k].iloc[0])
                if v < lo or v > hi:
                    outside.append(f"{k} ({v:g}, trained on {lo:g}–{hi:g})")
        if outside:
            st.warning("Outside the model's training range, so the prediction is an "
                       "extrapolation: " + "; ".join(outside))

        drivers = list(bundle["feature_importance"].items())[:5]
        st.caption("Top drivers of this model: " + ", ".join(str(k) for k, _ in drivers))

        with st.expander("Show entered specification"):
            st.dataframe(user_row.T.rename(columns={0: "value"}).astype(str),
                         use_container_width=True)

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

            st.dataframe(out.head(50), use_container_width=True)
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
        st.dataframe(scores.to_frame().style.format("{:.4f}"), use_container_width=True)
    with c2:
        st.markdown("**Metrics of the deployed model**")
        st.dataframe(pd.Series(bundle["metrics"], name="value").to_frame()
                     .style.format("{:.4f}"), use_container_width=True)

    st.markdown("**Permutation feature importance** (measured on unseen data)")
    fi = pd.Series(bundle["feature_importance"]).sort_values(ascending=False).head(12)
    st.bar_chart(fi)
    st.caption("Model artefact: " + MODEL_PATH + "  |  scikit-learn " +
               bundle["sklearn_version"])
