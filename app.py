import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from afp_pipeline import build_outputs

st.set_page_config(page_title="AFP GAP Dashboard", layout="wide")
st.title("AFP GAP Dashboard — Histórico + Señales + Recomendación (Táctico + Relativo)")

# -----------------------------
# Fuente del Excel: uploader (cloud) o path (local)
# -----------------------------
st.caption("Sugerencia: en Streamlit Cloud usa **Subir Excel**. En tu PC puedes usar ruta local.")

colA, colB = st.columns([1.3, 1])
with colA:
    uploaded = st.file_uploader("Sube el Excel (.xlsx) (recomendado para compartir)", type=["xlsx"])
with colB:
    file_path = st.text_input("O ruta local del Excel (.xlsx)", value="")

run = st.button("Cargar y ejecutar")

@st.cache_data(show_spinner=False)
def cached_build(xls_source):
    return build_outputs(xls_source)

if not run:
    st.info("Sube el Excel o pega ruta local, luego presiona **Cargar y ejecutar**.")
    st.stop()

# Decide fuente
xls_source = None
if uploaded is not None:
    xls_source = uploaded
elif file_path.strip():
    # si pegaste carpeta, intenta encontrar el único xlsx dentro
    if os.path.isdir(file_path):
        candidates = [f for f in os.listdir(file_path) if f.lower().endswith(".xlsx")]
        if len(candidates) == 1:
            xls_source = os.path.join(file_path, candidates[0])
        elif len(candidates) > 1:
            st.error("Pegaste una carpeta con varios .xlsx. Pega la ruta exacta del archivo.")
            st.stop()
        else:
            st.error("Pegaste una carpeta pero no hay .xlsx dentro.")
            st.stop()
    else:
        xls_source = file_path
else:
    st.error("Debes subir el Excel o pegar la ruta local.")
    st.stop()

with st.spinner("Procesando datos (features, señales, modelo, acciones, eventos)..."):
    df_raw, df_model, snap_last, metrics, events, last_date = cached_build(xls_source)

st.success(
    f"OK | Última fecha: {last_date.date()} | Filas: {metrics['rows']} | "
    f"AUC: {metrics['AUC_mean']:.3f} | ACC: {metrics['ACC_mean']:.3f}"
)

# -----------------------------
# Controles globales
# -----------------------------
min_d, max_d = df_model["Fecha"].min(), df_model["Fecha"].max()
default_start = max(min_d, max_d - pd.DateOffset(months=36))

c1, c2, c3 = st.columns([1.2, 1.2, 1])
with c1:
    date_range = st.slider(
        "Rango histórico (contexto)",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(default_start.to_pydatetime(), max_d.to_pydatetime())
    )
with c2:
    tickers = sorted(df_model["Nemo"].unique().tolist())
    sel_tickers = st.multiselect("Papeles a superponer (máx 8 recomendado)", options=tickers, default=[])
with c3:
    top_n = st.number_input("Top N", min_value=5, max_value=50, value=15, step=5)

d1 = pd.to_datetime(date_range[0])
d2 = pd.to_datetime(date_range[1])
dfh = df_model[(df_model["Fecha"] >= d1) & (df_model["Fecha"] <= d2)].copy()

tabs = st.tabs([
    "✅ Snapshot última fecha (Qué hacer)",
    "📈 Todos los papeles (gráfico usable)",
    "🧭 Mapa ciclo (GAP vs ΔGAP)",
    "📌 Ranking (última fecha)",
    "📊 Detalle por papel + eventos",
    "🗺️ Heatmaps",
    "🧪 Backtest"
])

# =========================
# TAB 1: Snapshot última fecha (acciones)
# =========================
with tabs[0]:
    st.subheader(f"Snapshot — Última fecha: {last_date.date()} (acciones)")

    show = [
        "Semaforo","Nemo","Fase","FlowScore_0_100",
        "GAP","Delta_GAP","Aceleracion",
        "P_Up_next","Delta_next_hat",
        "Accion_Tactica","Razon_Tactica",
        "Accion_Relativa","Tilt_bps","Razon_Relativa",
        "Etiqueta_Acumulacion","Etiqueta_Arrastre","Etiqueta_Saturacion"
    ]
    show = [c for c in show if c in snap_last.columns]

    # Orden operativo: primero BUY / OW
    order_df = snap_last.copy()
    order_df["__buy_first"] = order_df["Accion_Tactica"].isin(["BUY", "BUY (light)", "BUY (cover)"]).astype(int)
    order_df = order_df.sort_values(["__buy_first","FlowScore_0_100","P_Up_next"], ascending=False).drop(columns=["__buy_first"])

    st.dataframe(order_df[show], use_container_width=True, height=650)

    st.download_button(
        "Descargar snapshot (CSV)",
        data=order_df[show].to_csv(index=False).encode("utf-8-sig"),
        file_name="snapshot_ultima_fecha.csv",
        mime="text/csv"
    )

# =========================
# TAB 2: “Todos los papeles” sin spaghetti
# =========================
with tabs[1]:
    st.subheader("Gráfico universo: mediana + bandas (p25–p75) + selección")

    # Distribución cross-section mensual del GAP (mediana + percentiles)
    dist = dfh.groupby("Fecha")["GAP"].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    dist.columns = ["Fecha", "p25", "p50", "p75"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dist["Fecha"], y=dist["p50"], mode="lines", name="GAP mediana"))
    fig.add_trace(go.Scatter(
        x=dist["Fecha"], y=dist["p75"], mode="lines", name="p75", line=dict(width=1), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=dist["Fecha"], y=dist["p25"], mode="lines", name="p25", line=dict(width=1), opacity=0.6,
        fill="tonexty"
    ))
    fig.add_hline(y=0, line_width=1)

    # Superponer algunos tickers
    if sel_tickers:
        sub = dfh[dfh["Nemo"].isin(sel_tickers)].copy()
        for t in sel_tickers[:8]:
            dt = sub[sub["Nemo"] == t]
            fig.add_trace(go.Scatter(x=dt["Fecha"], y=dt["GAP"], mode="lines", name=t))

    fig.update_layout(template="plotly_white", hovermode="x unified",
                      title="GAP universo (p25–p50–p75) + papeles seleccionados",
                      xaxis_title="Fecha", yaxis_title="GAP")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3: Mapa ciclo (GAP vs ΔGAP)
# =========================
with tabs[2]:
    st.subheader(f"Mapa del ciclo — Última fecha ({last_date.date()})")

    x = snap_last.copy()
    # Scatter: X=GAP (posición), Y=ΔGAP (flujo), tamaño=FlowScore, hover con acción
    fig = px.scatter(
        x,
        x="GAP",
        y="Delta_GAP",
        size="FlowScore_0_100",
        hover_data=["Nemo","Semaforo","Fase","Accion_Tactica","Accion_Relativa","Tilt_bps","P_Up_next"],
        title="Ciclo de posicionamiento AFP: GAP (posición) vs ΔGAP (flujo)"
    )
    fig.update_layout(template="plotly_white")
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 4: Ranking (última fecha)
# =========================
with tabs[3]:
    st.subheader("Ranking — Última fecha (operativo)")

    left, right = st.columns(2)

    with left:
        st.markdown("**Top Oportunidad (Flow / BUY)**")
        cols = ["Semaforo","Nemo","Fase","FlowScore_0_100","P_Up_next","Delta_next_hat","Accion_Tactica","Accion_Relativa","Tilt_bps","GAP","Delta_GAP","Aceleracion"]
        cols = [c for c in cols if c in snap_last.columns]
        st.dataframe(
            snap_last.sort_values(["FlowScore_0_100","P_Up_next"], ascending=False).head(int(top_n))[cols],
            use_container_width=True, height=520
        )

    with right:
        st.markdown("**Top Riesgo (Salida / Saturación)**")
        cols2 = ["Semaforo","Nemo","Fase","Score_SatRisk","P_Up_next","Accion_Tactica","Accion_Relativa","Tilt_bps","GAP","Delta_GAP","Aceleracion"]
        cols2 = [c for c in cols2 if c in snap_last.columns]
        st.dataframe(
            snap_last.sort_values(["Score_SatRisk","GAP"], ascending=False).head(int(top_n))[cols2],
            use_container_width=True, height=520
        )

# =========================
# TAB 5: Detalle por papel + eventos
# =========================
with tabs[4]:
    st.subheader("Detalle por papel (histórico + señales + notas)")

    paper = st.selectbox("Selecciona papel", sorted(dfh["Nemo"].unique().tolist()))
    d = dfh[dfh["Nemo"] == paper].sort_values("Fecha").copy()

    # Estado último mes (si existe)
    last_row = df_model[(df_model["Nemo"] == paper) & (df_model["Fecha"] == last_date)]
    if len(last_row):
        lr = last_row.iloc[0]
        st.markdown(f"### {lr.get('Semaforo','')} {paper} — {lr.get('Fase','')} | "
                    f"Acción táctica: **{lr.get('Accion_Tactica','')}** | "
                    f"Acción relativa: **{lr.get('Accion_Relativa','')}** (Tilt {lr.get('Tilt_bps',0):.0f} bps)")
        cA, cB, cC, cD = st.columns(4)
        cA.metric("FlowScore (0–100)", f"{lr.get('FlowScore_0_100', float('nan')):.0f}" if pd.notna(lr.get("FlowScore_0_100", None)) else "NA")
        cB.metric("GAP", f"{lr.get('GAP', float('nan')):.4f}")
        cC.metric("ΔGAP", f"{lr.get('Delta_GAP', float('nan')):.4f}" if pd.notna(lr.get("Delta_GAP", None)) else "NA")
        cD.metric("P(ΔGAP>0)", f"{lr.get('P_Up_next', float('nan')):.0%}" if pd.notna(lr.get("P_Up_next", None)) else "NA")

    # Gráfico 1: GAP + MA3 + señales
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Fecha"], y=d["GAP"], mode="lines", name="GAP"))
    if "MA_3" in d.columns:
        fig.add_trace(go.Scatter(x=d["Fecha"], y=d["MA_3"], mode="lines", name="MA 3M"))
    fig.add_hline(y=0, line_width=1)

    # puntos señales (manteniendo las originales)
    if "Acumulacion" in d.columns:
        dd = d[d["Acumulacion"] == True]
        fig.add_trace(go.Scatter(x=dd["Fecha"], y=dd["GAP"], mode="markers", name="Compra fuerte (AFP entrando)"))
    if "Arrastre_T1" in d.columns:
        dd = d[d["Arrastre_T1"] == True]
        fig.add_trace(go.Scatter(x=dd["Fecha"], y=dd["GAP"], mode="markers", name="Compra seguidora (rebalanceo)"))
    if "Saturacion" in d.columns:
        dd = d[d["Saturacion"] == True]
        fig.add_trace(go.Scatter(x=dd["Fecha"], y=dd["GAP"], mode="markers", name="Salida / toma de utilidades"))

    fig.update_layout(template="plotly_white", hovermode="x unified",
                      title=f"{paper} | GAP + MA3 + señales",
                      xaxis_title="Fecha", yaxis_title="GAP")
    st.plotly_chart(fig, use_container_width=True)

    # Gráfico 2: ΔGAP y Aceleración
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Delta_GAP"], mode="lines", name="ΔGAP (flujo)"))
    fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Aceleracion"], mode="lines", name="Aceleración (impulso)"))
    fig2.add_hline(y=0, line_width=1)
    fig2.update_layout(template="plotly_white", hovermode="x unified",
                       title=f"{paper} | Flujo e impulso",
                       xaxis_title="Fecha")
    st.plotly_chart(fig2, use_container_width=True)

    # Gráfico 3: Predicción histórica
    if "P_Up_next" in d.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=d["Fecha"], y=d["P_Up_next"], mode="lines", name="P(ΔGAP>0) próximo mes"))
        fig3.update_layout(template="plotly_white", hovermode="x unified",
                           title=f"{paper} | Probabilidad de continuación del flujo",
                           yaxis_title="Probabilidad")
        st.plotly_chart(fig3, use_container_width=True)

    # Eventos (tabla + timeline)
    st.subheader("📌 Eventos históricos (cambios de estado)")

    paper_events = events[events["Nemo"] == paper].sort_values("Fecha").copy()
    show_e = ["Fecha","Semaforo","Fase","Nota","GAP","Delta_GAP","Aceleracion","FlowScore_0_100","P_Up_next"]
    show_e = [c for c in show_e if c in paper_events.columns]
    st.dataframe(paper_events[show_e], use_container_width=True, height=260)

    st.subheader("🗓️ Timeline visual")
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
        fig_t.update_layout(template="plotly_white", height=220,
                            title=f"{paper} — Timeline de cambios de estado")
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No hay eventos para este papel en el rango seleccionado.")

# =========================
# TAB 6: Heatmaps
# =========================
with tabs[5]:
    st.subheader("Heatmaps (papel vs mes)")

    metric = st.selectbox("Heatmap de", ["FlowScore_0_100","P_Up_next","GAP","Delta_GAP","Score_SatRisk"], index=0)
    if metric not in dfh.columns:
        st.warning(f"No existe {metric} en df.")
    else:
        pivot = dfh.pivot_table(index="Nemo", columns="Fecha", values=metric, aggfunc="max").fillna(0)
        fig = px.imshow(pivot, aspect="auto", title=f"Heatmap histórico: {metric}")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 7: Backtest
# =========================
with tabs[6]:
    st.subheader("Backtest histórico: señales → resultado próximo mes (Up_next)")

    if "Up_next" not in dfh.columns:
        st.warning("No existe Up_next.")
    else:
        bt = dfh.dropna(subset=["Up_next"]).copy()

        def hit_rate(mask_col):
            m = bt[bt[mask_col] == True]
            if len(m) == 0:
                return None, 0
            return float(m["Up_next"].mean()), int(len(m))

        rows = []
        for c, label in [
            ("Acumulacion","Compra fuerte (AFP entrando)"),
            ("Arrastre_T1","Compra seguidora (rebalanceo)"),
            ("Saturacion","Salida / toma de utilidades")
        ]:
            if c in bt.columns:
                hr, n = hit_rate(c)
                rows.append({"Señal": label, "HitRate (Up_next=1)": hr, "N": n})

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.download_button(
            "Descargar histórico modelado (CSV)",
            data=dfh.to_csv(index=False).encode("utf-8-sig"),
            file_name="historico_modelado.csv",
            mime="text/csv"
        )
