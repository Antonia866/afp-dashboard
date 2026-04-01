import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from afp_pipeline import build_outputs

st.set_page_config(page_title="AFP GAP Dashboard", layout="wide")
st.title("AFP GAP Dashboard — Flujo AFP por papel")


# =========================================================
# HELPERS
# =========================================================
def fmt_pct(x):
    try:
        return f"{x:.2%}"
    except:
        return x


def format_display_df(df):
    out = df.copy()
    for c in ["GAP", "Delta_GAP", "Prob_Compra_AFP_ProxMes"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_pct)
    return out


# =========================================================
# INPUT
# =========================================================
colA, colB = st.columns([1.3, 1])

with colA:
    uploaded = st.file_uploader("Sube el Excel (.xlsx)", type=["xlsx"])
with colB:
    file_path = st.text_input("O ruta local del Excel (.xlsx)", value="")

run = st.button("Cargar y ejecutar")


@st.cache_data(show_spinner=False)
def cached_build(xls_source):
    return build_outputs(xls_source)


if not run:
    st.info("Sube el Excel o pega ruta local, luego presiona **Cargar y ejecutar**.")
    st.stop()


# =========================================================
# LOAD
# =========================================================
if uploaded is not None:
    xls_source = uploaded
elif file_path:
    xls_source = file_path
else:
    st.error("Debes subir archivo o ruta")
    st.stop()


with st.spinner("Procesando..."):
    df_raw, df_model, snap_last, metrics, events, last_date = cached_build(xls_source)


st.success(f"Última fecha usada: {last_date.date()}")


# =========================================================
# CONTROLES
# =========================================================
min_d = df_model["Fecha"].min()
max_d = df_model["Fecha"].max()

date_range = st.slider(
    "Rango histórico",
    min_value=min_d.to_pydatetime(),
    max_value=max_d.to_pydatetime(),
    value=(min_d.to_pydatetime(), max_d.to_pydatetime())
)

dfh = df_model[
    (df_model["Fecha"] >= pd.to_datetime(date_range[0])) &
    (df_model["Fecha"] <= pd.to_datetime(date_range[1]))
]


# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "📈 GAP histórico",
    "✅ Snapshot última fecha",
    "🏁 Ranking",
    "📊 Detalle",
    "🟦 Heatmap"
])


# =========================================================
# TAB 1
# =========================================================
with tabs[0]:
    st.subheader("GAP histórico")

    fig = px.line(dfh, x="Fecha", y="GAP", color="Nemo")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 2
# =========================================================
with tabs[1]:
    st.subheader(f"Snapshot última fecha: {last_date.date()}")

    cols = [
        "Nemo", "Semaforo", "Fase",
        "GAP", "Delta_GAP",
        "Flujo_AFP",
        "CompraVenta_Fuerte",
        "Prob_Compra_AFP_ProxMes",
        "Accion_Tactica"
    ]

    cols = [c for c in cols if c in snap_last.columns]

    st.dataframe(format_display_df(snap_last[cols]), use_container_width=True)


# =========================================================
# TAB 3
# =========================================================
with tabs[2]:
    st.subheader("Ranking")

    if "Delta_GAP" in snap_last.columns:
        df_rank = snap_last.sort_values("Delta_GAP", ascending=False)
    else:
        df_rank = snap_last

    st.dataframe(format_display_df(df_rank), use_container_width=True)


# =========================================================
# TAB 4
# =========================================================
with tabs[3]:
    st.subheader("Detalle por papel")

    paper = st.selectbox("Papel", sorted(dfh["Nemo"].unique()))

    d = dfh[dfh["Nemo"] == paper].sort_values("Fecha")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Fecha"], y=d["GAP"], name="GAP"))
    fig.add_hline(y=0)
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 5
# =========================================================
with tabs[4]:
    st.subheader("Heatmap")

    if "GAP" in dfh.columns:
        pivot = dfh.pivot_table(index="Nemo", columns="Fecha", values="GAP")
        fig = px.imshow(pivot, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
