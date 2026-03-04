import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from afp_pipeline import build_outputs

st.set_page_config(page_title="AFP GAP Dashboard", layout="wide")
st.title("AFP GAP Dashboard — GAP Histórico + Señales + Probabilidad AFP")

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

xls_source = None
if uploaded is not None:
    xls_source = uploaded
elif file_path.strip():
    if os.path.isdir(file_path):
        candidates = [f for f in os.listdir(file_path) if f.lower().endswith(".xlsx")]
        if len(candidates) == 1:
            xls_source = os.path.join(file_path, candidates[0])
        else:
            st.error("Pegaste carpeta. Deja solo 1 .xlsx dentro o pega ruta exacta.")
            st.stop()
    else:
        xls_source = file_path
else:
    st.error("Debes subir el Excel o pegar la ruta local.")
    st.stop()

with st.spinner("Procesando datos..."):
    df_raw, df_model, snap_last, metrics, events, last_date, df_ipsa = cached_build(xls_source)

st.success(
    f"OK | Última fecha (Hola Valores!I2): {last_date.date()} | "
    f"Filas: {metrics['rows']} | AUC: {metrics['AUC_mean']:.3f} | ACC: {metrics['ACC_mean']:.3f}"
)

min_d, max_d = df_model["Fecha"].min(), df_model["Fecha"].max()
default_start = max(min_d, max_d - pd.DateOffset(months=36))

c1, c2 = st.columns([1.2, 1.2])
with c1:
    date_range = st.slider(
        "Rango histórico (contexto)",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(default_start.to_pydatetime(), max_d.to_pydatetime())
    )
with c2:
    tickers = sorted(df_model["Nemo"].unique().tolist())
    sel_tickers = st.multiselect("Papeles a superponer (máx 8)", options=tickers, default=[])

d1 = pd.to_datetime(date_range[0])
d2 = pd.to_datetime(date_range[1])
dfh = df_model[(df_model["Fecha"] >= d1) & (df_model["Fecha"] <= d2)].copy()

tabs = st.tabs([
    "📈 Todos los papeles (gráfico usable)",
    "✅ Snapshot última fecha (Qué hacer)",
    "🏁 Ranking última fecha",
    "📊 Detalle por papel + eventos",
    "🟦 Heatmap (GAP / ΔGAP)"
])

# ----------------------------
# TAB 0
# ----------------------------
with tabs[0]:
    st.subheader("GAP promedio histórico vs IPSA")

    gap_avg = dfh.groupby("Fecha", as_index=False).agg(GAP_prom=("GAP", "mean"))

    if df_ipsa is not None and not df_ipsa.empty:
        ips = df_ipsa[(df_ipsa["Fecha"] >= d1) & (df_ipsa["Fecha"] <= d2)].copy()
        m = pd.merge(gap_avg, ips, on="Fecha", how="left")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=m["Fecha"], y=m["GAP_prom"], mode="lines", name="GAP promedio"), secondary_y=False)
        fig.add_trace(go.Scatter(x=m["Fecha"], y=m["IPSA"], mode="lines", name="IPSA"), secondary_y=True)
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(template="plotly_white", hovermode="x unified",
                          title="GAP promedio (izq) vs IPSA (der)")
        fig.update_yaxes(title_text="GAP promedio", secondary_y=False)
        fig.update_yaxes(title_text="IPSA", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No pude leer serie IPSA desde el Excel (hoja IPSA/alias). Mostrando solo GAP promedio.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gap_avg["Fecha"], y=gap_avg["GAP_prom"], mode="lines", name="GAP promedio"))
        fig.add_hline(y=0, line_width=1)
        fig.update_layout(template="plotly_white", hovermode="x unified", title="GAP promedio histórico")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("GAP por papel (fecha seleccionable) + Promedio histórico")

    available_dates = sorted(dfh["Fecha"].dropna().unique())
    default_idx = len(available_dates) - 1
    if last_date in available_dates:
        default_idx = available_dates.index(last_date)

    sel_date = st.selectbox(
        "Fecha a visualizar (fin de mes)",
        options=available_dates,
        index=max(0, default_idx),
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
    )

    snap_date = dfh[dfh["Fecha"] == pd.to_datetime(sel_date)][["Nemo", "GAP"]].dropna().copy()
    snap_date = snap_date.rename(columns={"GAP": "GAP_Fecha"})
    hist_avg = dfh.groupby("Nemo", as_index=False)["GAP"].mean().rename(columns={"GAP": "GAP_Prom_Hist"})

    comp = pd.merge(snap_date, hist_avg, on="Nemo", how="left")
    comp["Dif_vs_Prom"] = comp["GAP_Fecha"] - comp["GAP_Prom_Hist"]
    comp = comp.sort_values("GAP_Fecha", ascending=False)

    long_ = comp.melt(
        id_vars=["Nemo", "Dif_vs_Prom"],
        value_vars=["GAP_Fecha", "GAP_Prom_Hist"],
        var_name="Serie",
        value_name="GAP"
    )
    long_["Serie"] = long_["Serie"].map({
        "GAP_Fecha": f"GAP {pd.to_datetime(sel_date).date()}",
        "GAP_Prom_Hist": "Promedio histórico (rango)"
    })

    fig_bar = px.bar(
        long_,
        x="Nemo",
        y="GAP",
        color="Serie",
        barmode="group",
        title=f"GAP por papel vs Promedio histórico — {pd.to_datetime(sel_date).date()}",
        hover_data={"Dif_vs_Prom": ":.4f"}
    )
    fig_bar.add_hline(y=0, line_width=1)
    fig_bar.update_layout(template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Universo sin spaghetti: bandas p25–p75 + tickers seleccionados")
    dist = dfh.groupby("Fecha")["GAP"].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    dist.columns = ["Fecha", "p25", "p50", "p75"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dist["Fecha"], y=dist["p50"], mode="lines", name="GAP mediana"))
    fig2.add_trace(go.Scatter(x=dist["Fecha"], y=dist["p75"], mode="lines", name="p75", line=dict(width=1), opacity=0.6))
    fig2.add_trace(go.Scatter(x=dist["Fecha"], y=dist["p25"], mode="lines", name="p25", line=dict(width=1), opacity=0.6, fill="tonexty"))
    fig2.add_hline(y=0, line_width=1)

    if sel_tickers:
        sub = dfh[dfh["Nemo"].isin(sel_tickers)].copy()
        for t in sel_tickers[:8]:
            dt_ = sub[sub["Nemo"] == t]
            fig2.add_trace(go.Scatter(x=dt_["Fecha"], y=dt_["GAP"], mode="lines", name=t))

    fig2.update_layout(template="plotly_white", hovermode="x unified",
                       title="GAP universo (p25–p50–p75) + papeles seleccionados")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# TAB 1
# ----------------------------
with tabs[1]:
    st.subheader(f"Snapshot — Última fecha: {last_date.date()}")

    show = [
        "Semaforo", "Nemo", "Fase",
        "GAP", "Delta_GAP",
        "Flujo_AFP",
        "CompraVenta_Fuerte",
        "Prob_Compra_AFP_ProxMes",
        "Accion_Tactica", "Accion_Relativa",
        "Recomendacion_Timing"
    ]
    show = [c for c in show if c in snap_last.columns]

    order = snap_last.copy()
    order["__sort"] = order.get("Recomendacion_Timing", pd.Series(["MANTENER"]*len(order))).map({
        "COMPRAR en T": 0,
        "COMPRAR / MANTENER": 1,
        "MANTENER": 2,
        "REDUCIR (light)": 3,
        "VENDER / REDUCIR": 4
    }).fillna(9)
    if "Delta_GAP" in order.columns:
        order = order.sort_values(["__sort", "Semaforo", "Delta_GAP"], ascending=[True, True, False]).drop(columns=["__sort"])
    else:
        order = order.sort_values(["__sort", "Semaforo"], ascending=[True, True]).drop(columns=["__sort"])

    st.dataframe(order[show], use_container_width=True, height=650)

# ----------------------------
# TAB 2 (ranking robusto)
# ----------------------------
with tabs[2]:
    st.subheader("Ranking — Última fecha")

    cols = [
        "Semaforo", "Nemo", "Fase",
        "GAP", "Delta_GAP",
        "CompraVenta_Fuerte",
        "Prob_Compra_AFP_ProxMes",
        "Accion_Tactica", "Accion_Relativa"
    ]
    cols = [c for c in cols if c in snap_last.columns]

    left, right = st.columns(2)

    with left:
        st.markdown("**Top oportunidades**")
        sort_cols = [c for c in ["FlowScore_0_100", "Prob_Compra_AFP_ProxMes", "Delta_GAP"] if c in snap_last.columns]
        if not sort_cols:
            sort_cols = [c for c in ["Delta_GAP", "GAP"] if c in snap_last.columns] or ["Nemo"]
        df_rank = snap_last.sort_values(sort_cols, ascending=False)
        st.dataframe(df_rank.head(20)[cols], use_container_width=True, height=520)

    with right:
        st.markdown("**Top riesgo / salida**")
        if "Score_SatRisk" in snap_last.columns:
            df_risk = snap_last.sort_values(["Score_SatRisk", "Delta_GAP"], ascending=[False, True])
        else:
            if "Delta_GAP" in snap_last.columns:
                df_risk = snap_last.sort_values(["Delta_GAP", "GAP"], ascending=[True, True])
            else:
                df_risk = snap_last.copy()
        st.dataframe(df_risk.head(20)[cols], use_container_width=True, height=520)

# ----------------------------
# TAB 3
# ----------------------------
with tabs[3]:
    st.subheader("Detalle por papel + eventos")

    paper = st.selectbox("Selecciona papel", sorted(dfh["Nemo"].unique().tolist()))
    d = dfh[dfh["Nemo"] == paper].sort_values("Fecha").copy()

    last_row = df_model[(df_model["Nemo"] == paper) & (df_model["Fecha"] == last_date)]
    if len(last_row):
        lr = last_row.iloc[0]
        st.markdown(
            f"### {lr.get('Semaforo','')} {paper} — {lr.get('Fase','')} | "
            f"Timing: **{lr.get('Recomendacion_Timing','')}** | "
            f"Flujo: **{lr.get('Flujo_AFP','')}** | "
            f"Fuerte: **{lr.get('CompraVenta_Fuerte','Neutral')}**"
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Fecha"], y=d["GAP"], mode="lines", name="GAP"))
    if "MA_3" in d.columns:
        fig.add_trace(go.Scatter(x=d["Fecha"], y=d["MA_3"], mode="lines", name="MA 3M"))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(template="plotly_white", hovermode="x unified",
                      title=f"{paper} | GAP + MA3")
    st.plotly_chart(fig, use_container_width=True)

    if {"Delta_GAP", "Aceleracion", "Impulso"}.issubset(set(d.columns)):
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Delta_GAP"], mode="lines", name="ΔGAP"))
        fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Aceleracion"], mode="lines", name="Aceleración"))
        fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Impulso"], mode="lines", name="Impulso"))
        fig2.add_hline(y=0, line_width=1)
        fig2.update_layout(template="plotly_white", hovermode="x unified",
                           title=f"{paper} | Flujo / Aceleración / Impulso")
        st.plotly_chart(fig2, use_container_width=True)

    if "P_Up_next" in d.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=d["Fecha"], y=d["P_Up_next"], mode="lines", name="Prob. compra AFP próximo mes"))
        fig3.update_layout(template="plotly_white", hovermode="x unified",
                           title=f"{paper} | Probabilidad de compra AFP (próximo mes)")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📌 Eventos históricos")
    pe = events[events["Nemo"] == paper].sort_values("Fecha").copy()
    st.dataframe(pe, use_container_width=True, height=260)

# ----------------------------
# TAB 4
# ----------------------------
with tabs[4]:
    st.subheader("Heatmap — GAP / ΔGAP")

    metric = st.selectbox("Métrica", ["GAP", "Delta_GAP"], index=0)

    if metric not in dfh.columns:
        st.warning(f"No existe {metric}.")
    else:
        last_vals = dfh[dfh["Fecha"] == dfh["Fecha"].max()][["Nemo", metric]].dropna()
        ordered = last_vals.sort_values(metric, ascending=False)["Nemo"].tolist()

        pivot = dfh.pivot_table(index="Nemo", columns="Fecha", values=metric, aggfunc="last")
        if ordered:
            pivot = pivot.reindex(ordered)

        fig = px.imshow(pivot, aspect="auto", title=f"Heatmap histórico: {metric}")
        fig.update_layout(template="plotly_white", height=650)
        st.plotly_chart(fig, use_container_width=True)
