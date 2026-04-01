import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from afp_pipeline import build_outputs

st.set_page_config(page_title="AFP GAP Dashboard", layout="wide")
st.title("AFP GAP Dashboard — Flujo AFP por papel")

# =========================================================
# HELPERS VISUALES
# =========================================================
def fmt_pct(x):
    try:
        return f"{x:.2%}"
    except Exception:
        return x


def format_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["GAP", "Delta_GAP"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_pct)
    return out


def appendice_general():
    st.markdown(
        """
### Apéndice general
- **GAP** = Peso cartera AFP − Peso IPSA.
- **Delta GAP** = cambio mensual del GAP.
- Ambos se muestran como **%**.
- **Última fecha**: siempre se toma desde `Hola Valores!I2`.
- Universo analizado: papeles donde **AFP = tiene** e **IPSA = tiene**.
"""
    )


def appendice_senales():
    st.markdown(
        """
### Apéndice de señales
**Fase**
- **Largo comprando**: GAP > 0 y Delta GAP > 0
- **Largo vendiendo**: GAP > 0 y Delta GAP < 0
- **Corto aumentando**: GAP < 0 y Delta GAP < 0
- **Corto cubriendo**: GAP < 0 y Delta GAP > 0
- **Manteniendo**: sin señal clara

**Flujo AFP**
- **Entrada activas**: compras que se están fortaleciendo
- **Entrada seguidoras**: compras que continúan, pero más suaves
- **Salida activas**: ventas que se están fortaleciendo
- **Salida seguidoras**: ventas que continúan, pero más suaves

**Compra fuerte / Venta fuerte**
- **Compra fuerte**: Delta GAP muy alto vs historia + flujo fortaleciéndose
- **Venta fuerte**: Delta GAP muy bajo vs historia + flujo deteriorándose
"""
    )


def appendice_acciones():
    st.markdown(
        """
### Apéndice de acciones

**Acción táctica**
- Pensada para decisión mensual de trading.
- **BUY / BUY (light)**: conviene comprar o acompañar
- **HOLD**: conviene mantener
- **SELL/REDUCE / REDUCE**: conviene vender o reducir

**Cómo se calcula**
- Combina:
  - semáforo
  - compra fuerte / venta fuerte
  - dirección del flujo

**Acción relativa**
- Pensada para asignación versus benchmark.
- **OVERWEIGHT**: sobreponderar
- **NEUTRAL**: mantener peso benchmark
- **UNDERWEIGHT**: infraponderar
"""
    )


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

xls_source = None
if uploaded is not None:
    xls_source = uploaded
elif file_path.strip():
    if os.path.isdir(file_path):
        candidates = [f for f in os.listdir(file_path) if f.lower().endswith(".xlsx")]
        if len(candidates) == 1:
            xls_source = os.path.join(file_path, candidates[0])
        else:
            st.error("Pegaste una carpeta. Deja solo 1 archivo .xlsx dentro o pega la ruta exacta.")
            st.stop()
    else:
        xls_source = file_path
else:
    st.error("Debes subir el Excel o pegar la ruta local.")
    st.stop()

with st.spinner("Procesando datos..."):
    out = cached_build(xls_source)
    if len(out) == 6:
        df_raw, df_model, snap_last, metrics, events, last_date = out
    else:
        raise ValueError("El pipeline no devolvió el formato esperado.")

st.success(
    f"OK | Última fecha (Hola Valores!I2): {last_date.date()} | "
    f"Filas modeladas: {metrics['rows']} | AUC: {metrics['AUC_mean']:.3f} | ACC: {metrics['ACC_mean']:.3f}"
)

# Controles
min_d, max_d = df_model["Fecha"].min(), df_model["Fecha"].max()
default_start = max(min_d, max_d - pd.DateOffset(months=36))

c1, c2 = st.columns([1.2, 1.2])
with c1:
    date_range = st.slider(
        "Rango histórico",
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
    "📈 Todos los papeles",
    "✅ Snapshot última fecha",
    "🏁 Ranking última fecha",
    "📊 Detalle por papel + eventos",
    "🟦 Heatmap"
])

# =========================================================
# TAB 1
# =========================================================
with tabs[0]:
    st.subheader("GAP por papel vs promedio histórico")

    available_dates = sorted(dfh["Fecha"].dropna().unique())
    default_idx = len(available_dates) - 1
    if last_date in available_dates:
        default_idx = available_dates.index(last_date)

    sel_date = st.selectbox(
        "Fecha a visualizar",
        options=available_dates,
        index=max(0, default_idx),
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
    )

    snap_date = dfh[dfh["Fecha"] == pd.to_datetime(sel_date)][["Nemo", "GAP", "Semaforo"]].dropna().copy()
    snap_date = snap_date.rename(columns={"GAP": "GAP_Fecha"})
    hist_avg = dfh.groupby("Nemo", as_index=False)["GAP"].mean().rename(columns={"GAP": "GAP_Prom_Hist"})

    comp = pd.merge(snap_date, hist_avg, on="Nemo", how="left")
    comp["Dif_vs_Prom"] = comp["GAP_Fecha"] - comp["GAP_Prom_Hist"]
    comp = comp.sort_values("GAP_Fecha", ascending=False)

    long_ = comp.melt(
        id_vars=["Nemo", "Semaforo", "Dif_vs_Prom"],
        value_vars=["GAP_Fecha", "GAP_Prom_Hist"],
        var_name="Serie",
        value_name="GAP"
    )
    long_["Serie"] = long_["Serie"].map({
        "GAP_Fecha": f"GAP {pd.to_datetime(sel_date).date()}",
        "GAP_Prom_Hist": "Promedio histórico"
    })

    fig_bar = px.bar(
        long_,
        x="Nemo",
        y="GAP",
        color="Serie",
        barmode="group",
        title=f"GAP por papel vs promedio histórico — {pd.to_datetime(sel_date).date()}",
        hover_data={"Dif_vs_Prom": ":.2%"}
    )
    fig_bar.add_hline(y=0, line_width=1)
    fig_bar.update_layout(template="plotly_white", yaxis_tickformat=".1%")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Serie histórica superpuesta")
    if sel_tickers:
        plot_df = dfh[dfh["Nemo"].isin(sel_tickers)].copy()
        if not plot_df.empty:
            fig_lines = px.line(
                plot_df,
                x="Fecha",
                y="GAP",
                color="Nemo",
                title="Histórico GAP por papel"
            )
            fig_lines.add_hline(y=0, line_width=1)
            fig_lines.update_layout(template="plotly_white", yaxis_tickformat=".1%")
            st.plotly_chart(fig_lines, use_container_width=True)

    with st.expander("Apéndice — cómo leer esta pestaña"):
        appendice_general()
        st.markdown(
            """
- La barra **GAP fecha** muestra cómo están paradas las AFP en el mes elegido.
- La barra **Promedio histórico** muestra la referencia promedio del papel en el rango seleccionado.
- Si el GAP actual está muy sobre su promedio, el posicionamiento AFP está más cargado que lo habitual.
"""
        )

# =========================================================
# TAB 2
# =========================================================
with tabs[1]:
    st.subheader(f"Snapshot — Última fecha: {last_date.date()}")

    show = [
        "Semaforo", "Nemo", "Fase",
        "GAP", "Delta_GAP",
        "Flujo_AFP",
        "CompraVenta_Fuerte",
        "Accion_Tactica", "Accion_Relativa",
        "Recomendacion_Timing"
    ]
    show = [c for c in show if c in snap_last.columns]

    order = snap_last.copy()
    order["__sort"] = order["Recomendacion_Timing"].map({
        "COMPRAR en T": 0,
        "COMPRAR / MANTENER": 1,
        "MANTENER": 2,
        "REDUCIR (light)": 3,
        "VENDER / REDUCIR": 4
    }).fillna(9)
    order = order.sort_values(["__sort", "Semaforo", "Delta_GAP"], ascending=[True, True, False]).drop(columns=["__sort"])

    st.dataframe(format_display_df(order[show]), use_container_width=True, height=650)

    with st.expander("Apéndice — señales y acciones"):
        appendice_senales()
        appendice_acciones()

# =========================================================
# TAB 3
# =========================================================
with tabs[2]:
    st.subheader("Ranking — Última fecha")

    cols = [
        "Semaforo", "Nemo", "Fase",
        "GAP", "Delta_GAP",
        "CompraVenta_Fuerte",
        "Accion_Tactica", "Accion_Relativa"
    ]
    cols = [c for c in cols if c in snap_last.columns]

    left, right = st.columns(2)

    with left:
        st.markdown("**Top oportunidades**")
        sort_cols = [c for c in ["FlowScore_0_100", "Delta_GAP", "GAP"] if c in snap_last.columns]
        if not sort_cols:
            sort_cols = ["Nemo"]
        df_rank = snap_last.sort_values(sort_cols, ascending=False)
        st.dataframe(format_display_df(df_rank.head(20)[cols]), use_container_width=True, height=520)

    with right:
        st.markdown("**Top riesgo / salida**")
        if "Delta_GAP" in snap_last.columns:
            df_risk = snap_last.sort_values(["Delta_GAP", "GAP"], ascending=[True, True])
        else:
            df_risk = snap_last.copy()
        st.dataframe(format_display_df(df_risk.head(20)[cols]), use_container_width=True, height=520)

    with st.expander("Apéndice — cómo se ordena el ranking"):
        st.markdown(
            """
- **Top oportunidades** prioriza:
  1. score interno de flujo
  2. Delta GAP
  3. GAP

- **Top riesgo / salida** prioriza:
  1. Delta GAP más negativo
  2. GAP más débil
"""
        )
        appendice_acciones()

# =========================================================
# TAB 4
# =========================================================
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
    if "GAP_MA3" in d.columns:
        fig.add_trace(go.Scatter(x=d["Fecha"], y=d["GAP_MA3"], mode="lines", name="Media 3M"))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=f"{paper} | GAP histórico",
        yaxis_tickformat=".1%"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Delta_GAP"], mode="lines", name="Delta GAP"))
    fig2.add_hline(y=0, line_width=1)
    fig2.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=f"{paper} | Delta GAP",
        yaxis_tickformat=".1%"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Eventos históricos")
    pe = events[events["Nemo"] == paper].sort_values("Fecha").copy()
    show_e = ["Fecha", "Semaforo", "Fase", "Flujo_AFP", "CompraVenta_Fuerte", "GAP", "Delta_GAP", "Nota"]
    show_e = [c for c in show_e if c in pe.columns]
    st.dataframe(format_display_df(pe[show_e]), use_container_width=True, height=260)

    st.subheader("Línea de tiempo con semáforos")
    if len(pe) >= 1:
        pe_plot = pe.copy()
        pe_plot["y"] = 1
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=pe_plot["Fecha"],
            y=pe_plot["y"],
            mode="markers+text",
            text=pe_plot["Semaforo"],
            textposition="top center",
            hovertemplate=
                "<b>%{x|%Y-%m}</b><br>" +
                "Fase: %{customdata[0]}<br>" +
                "Flujo: %{customdata[1]}<br>" +
                "Fuerte: %{customdata[2]}<br>" +
                "GAP: %{customdata[3]:.2%}<br>" +
                "Delta GAP: %{customdata[4]:.2%}<extra></extra>",
            customdata=pe_plot[["Fase", "Flujo_AFP", "CompraVenta_Fuerte", "GAP", "Delta_GAP"]].values
        ))
        fig_t.update_yaxes(visible=False)
        fig_t.update_layout(template="plotly_white", height=240, title=f"{paper} — Timeline de señales")
        st.plotly_chart(fig_t, use_container_width=True)

    with st.expander("Apéndice — cómo leer el detalle por papel"):
        appendice_senales()

# =========================================================
# TAB 5
# =========================================================
with tabs[4]:
    st.subheader("Heatmap — GAP / Delta GAP")

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

    with st.expander("Apéndice — cómo leer el heatmap"):
        st.markdown(
            """
- **GAP**: muestra el posicionamiento relativo AFP vs IPSA.
- **Delta GAP**: muestra el cambio mensual del posicionamiento.
- Tonos más altos suelen indicar mayor sobreponderación o compras más fuertes.
- Tonos más bajos suelen indicar menor sobreponderación o ventas.
"""
        )
