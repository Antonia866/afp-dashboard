import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from afp_pipeline import build_outputs

st.set_page_config(page_title="AFP GAP Dashboard (Histórico)", layout="wide")
st.title("AFP GAP Dashboard — Histórico + Estados (Largos/Cortos) + Semáforo + Notas")

file_path = st.text_input(
    "Ruta del Excel (.xlsx)",
    value="IPSA Hist.xlsx"
)

run = st.button("Cargar y ejecutar")

@st.cache_data(show_spinner=False)
def cached_build(fp: str):
    return build_outputs(fp)

if not run:
    st.info("Pega la ruta del Excel y presiona **Cargar y ejecutar**.")
    st.stop()

with st.spinner("Procesando datos (features, estados, semáforo, modelo, eventos)..."):
    df_raw, df_model, snap, metrics, events = cached_build(file_path)

st.success(f"OK | Filas: {metrics['rows']} | AUC medio: {metrics['AUC_mean']:.3f} | ACC medio: {metrics['ACC_mean']:.3f}")

# -----------------------------
# Controles globales
# -----------------------------
min_d, max_d = df_model["Fecha"].min(), df_model["Fecha"].max()
cA, cB, cC = st.columns([1.2, 1.2, 1])

with cA:
    date_range = st.slider(
        "Rango histórico",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(min_d.to_pydatetime(), max_d.to_pydatetime())
    )

with cB:
    tickers = sorted(df_model["Nemo"].unique().tolist())
    sel_tickers = st.multiselect("Papeles (multi)", options=tickers, default=[])

with cC:
    top_n = st.number_input("Top N ranking", min_value=5, max_value=50, value=15, step=5)

d1 = pd.to_datetime(date_range[0])
d2 = pd.to_datetime(date_range[1])

dfh = df_model[(df_model["Fecha"] >= d1) & (df_model["Fecha"] <= d2)].copy()

tabs = st.tabs([
    "📌 Overview histórico",
    "🏁 Ranking histórico por mes",
    "📈 Detalle por papel (histórico + eventos)",
    "🗺️ Heatmaps (histórico)",
    "🧪 Backtest (histórico)"
])

# =========================
# TAB 1: Overview histórico
# =========================
with tabs[0]:
    st.subheader("Overview histórico del universo")

    agg = dfh.groupby("Fecha").agg(
        GAP_mean=("GAP", "mean"),
        GAP_median=("GAP", "median"),
        P_up_mean=("P_Up_next", "mean") if "P_Up_next" in dfh.columns else ("GAP", "mean"),
        FlowScore_mean=("FlowScore_0_100", "mean") if "FlowScore_0_100" in dfh.columns else ("GAP", "mean"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["Fecha"], y=agg["GAP_mean"], mode="lines", name="GAP mean"))
    fig.add_trace(go.Scatter(x=agg["Fecha"], y=agg["GAP_median"], mode="lines", name="GAP median"))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(template="plotly_white", hovermode="x unified", title="GAP agregado histórico")
    st.plotly_chart(fig, use_container_width=True)

    if "P_up_mean" in agg.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=agg["Fecha"], y=agg["P_up_mean"], mode="lines", name="Promedio P(ΔGAP>0)"))
        fig2.update_layout(template="plotly_white", hovermode="x unified", title="Predicción agregada histórica: P(ΔGAP>0)")
        st.plotly_chart(fig2, use_container_width=True)

    if sel_tickers:
        sub = dfh[dfh["Nemo"].isin(sel_tickers)].copy()
        fig3 = px.line(sub, x="Fecha", y="GAP", color="Nemo", title="GAP histórico (selección)")
        fig3.update_layout(template="plotly_white", hovermode="x unified")
        fig3.add_hline(y=0, line_width=1)
        st.plotly_chart(fig3, use_container_width=True)

# ==================================
# TAB 2: Ranking histórico por mes
# ==================================
with tabs[1]:
    st.subheader("Ranking histórico (por mes) — Mesa (Semáforo + Estado + Score)")

    months = sorted(dfh["Fecha"].unique())
    month = st.selectbox("Selecciona mes", months, index=len(months) - 1)
    snap_m = dfh[dfh["Fecha"] == month].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Oportunidad (AFP Flow)**")
        cols = ["Semaforo","Nemo","Fase","FlowScore_0_100","GAP","Delta_GAP","Aceleracion","Score_AFPFlow","P_Up_next","Delta_next_hat"]
        cols = [c for c in cols if c in snap_m.columns]
        st.dataframe(
            snap_m.sort_values(["Score_AFPFlow","P_Up_next"], ascending=False).head(int(top_n))[cols],
            use_container_width=True,
            height=520
        )

    with col2:
        st.markdown("**Top Riesgo Saturación**")
        cols2 = ["Semaforo","Nemo","Fase","FlowScore_0_100","GAP","Delta_GAP","Aceleracion","Score_SatRisk","P_Up_next","Delta_next_hat"]
        cols2 = [c for c in cols2 if c in snap_m.columns]
        st.dataframe(
            snap_m.sort_values(["Score_SatRisk","GAP"], ascending=False).head(int(top_n))[cols2],
            use_container_width=True,
            height=520
        )

# ======================================
# TAB 3: Detalle por papel + eventos
# ======================================
with tabs[2]:
    st.subheader("Detalle histórico por papel (uno por uno)")

    paper = st.selectbox("Papel", sorted(dfh["Nemo"].unique().tolist()))
    d = dfh[dfh["Nemo"] == paper].sort_values("Fecha").copy()

    last = d.iloc[-1]
    st.markdown(f"### {last.get('Semaforo','')} {paper} — {last.get('Fase','')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FlowScore (0–100)", f"{last.get('FlowScore_0_100', float('nan')):.0f}" if pd.notna(last.get("FlowScore_0_100", None)) else "NA")
    c2.metric("GAP actual", f"{last.get('GAP', float('nan')):.4f}")
    c3.metric("ΔGAP actual", f"{last.get('Delta_GAP', float('nan')):.4f}" if pd.notna(last.get("Delta_GAP", None)) else "NA")
    c4.metric("P(ΔGAP>0) prox mes", f"{last.get('P_Up_next', float('nan')):.0%}" if pd.notna(last.get("P_Up_next", None)) else "NA")

    # GAP + MA3 + señales
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Fecha"], y=d["GAP"], mode="lines", name="GAP"))
    if "MA_3" in d.columns:
        fig.add_trace(go.Scatter(x=d["Fecha"], y=d["MA_3"], mode="lines", name="MA 3M"))
    fig.add_hline(y=0, line_width=1)

    # flags
    for col, label in [("Acumulacion","Acumulación"), ("Arrastre_T1","Arrastre T+1"), ("Saturacion","Saturación")]:
        if col in d.columns:
            dd = d[d[col] == True]
            fig.add_trace(go.Scatter(x=dd["Fecha"], y=dd["GAP"], mode="markers", name=label))

    fig.update_layout(template="plotly_white", hovermode="x unified", title=f"{paper} | GAP + MA + Señales")
    st.plotly_chart(fig, use_container_width=True)

    # ΔGAP + Aceleración
    fig2 = go.Figure()
    if "Delta_GAP" in d.columns:
        fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Delta_GAP"], mode="lines", name="ΔGAP"))
    if "Aceleracion" in d.columns:
        fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Aceleracion"], mode="lines", name="Aceleración"))
    fig2.add_hline(y=0, line_width=1)
    fig2.update_layout(template="plotly_white", hovermode="x unified", title=f"{paper} | ΔGAP y Aceleración")
    st.plotly_chart(fig2, use_container_width=True)

    # Predicción histórica
    if "P_Up_next" in d.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=d["Fecha"], y=d["P_Up_next"], mode="lines", name="P(ΔGAP>0)"))
        fig3.update_layout(template="plotly_white", hovermode="x unified", title=f"{paper} | Predicción histórica P(ΔGAP>0)")
        st.plotly_chart(fig3, use_container_width=True)

    # Eventos + notas
    st.subheader("📌 Eventos históricos detectados (cambios de estado)")

    paper_events = events[events["Nemo"] == paper].sort_values("Fecha").copy()
    show_e = ["Fecha", "Semaforo", "Fase", "Nota", "GAP", "Delta_GAP", "Aceleracion", "FlowScore_0_100", "P_Up_next"]
    show_e = [c for c in show_e if c in paper_events.columns]
    st.dataframe(paper_events[show_e], use_container_width=True, height=260)

    # Timeline visual
    st.subheader("🗓️ Timeline visual (cambios de estado)")

    if len(paper_events) >= 1:
        paper_events["y"] = 1

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=paper_events["Fecha"],
            y=paper_events["y"],
            mode="markers+text",
            text=paper_events["Semaforo"],
            textposition="top center",
            hovertemplate=
                "<b>%{x|%Y-%m}</b><br>" +
                "Estado: %{customdata[0]}<br>" +
                "Nota: %{customdata[1]}<br>" +
                "GAP: %{customdata[2]:.4f}<br>" +
                "ΔGAP: %{customdata[3]:.4f}<br>" +
                "Acel: %{customdata[4]:.4f}<br>" +
                "FlowScore: %{customdata[5]:.0f}<br>" +
                "P_up: %{customdata[6]:.0%}<extra></extra>",
            customdata=paper_events[["Fase","Nota","GAP","Delta_GAP","Aceleracion","FlowScore_0_100","P_Up_next"]].values
        ))

        fig_t.update_yaxes(visible=False)
        fig_t.update_layout(
            template="plotly_white",
            title=f"{paper} — Timeline de cambios de estado",
            height=220,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No hay eventos para este papel en el rango seleccionado.")

# ==============================
# TAB 4: Heatmaps
# ==============================
with tabs[3]:
    st.subheader("Heatmaps históricos (papel vs mes)")

    metric = st.selectbox("Heatmap de", ["Semaforo", "Fase", "Acumulacion", "Arrastre_T1", "Saturacion", "FlowScore_0_100", "P_Up_next"], index=5)

    if metric not in dfh.columns:
        st.warning(f"No existe la columna {metric} en el dataset.")
    else:
        # Para Semaforo/Fase conviene heatmap con valores numéricos.
        if metric in ["Semaforo", "Fase"]:
            st.info("Para 'Semaforo' y 'Fase' usa mejor la pestaña de detalle por papel. Heatmap numérico recomendado: FlowScore_0_100 o P_Up_next.")
        pivot = dfh.pivot_table(index="Nemo", columns="Fecha", values="FlowScore_0_100" if metric in ["Semaforo","Fase"] else metric, aggfunc="max").fillna(0)
        fig = px.imshow(pivot, aspect="auto", title=f"Heatmap histórico: {metric}")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# TAB 5: Backtest
# ==============================
with tabs[4]:
    st.subheader("Backtest histórico: señales → resultado próximo mes")

    if "Up_next" not in dfh.columns:
        st.warning("No existe Up_next (target) en el dataset.")
    else:
        bt = dfh.dropna(subset=["Up_next"]).copy()

        def hit_rate(mask_col):
            m = bt[bt[mask_col] == True]
            if len(m) == 0:
                return None, 0
            return float(m["Up_next"].mean()), int(len(m))

        rows = []
        for c in ["Acumulacion", "Arrastre_T1", "Saturacion"]:
            if c in bt.columns:
                hr, n = hit_rate(c)
                rows.append({"Señal": c, "HitRate (Up_next=1)": hr, "N": n})

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.download_button(
            "Descargar histórico modelado (CSV)",
            data=dfh.to_csv(index=False).encode("utf-8-sig"),
            file_name="historico_modelado_filtrado.csv",
            mime="text/csv"

        )
