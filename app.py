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
    for c in ["GAP", "Delta_GAP", "Delta_GAP_3M", "Delta_GAP_6M"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_pct)
    return out


def appendice_general():
    st.markdown(
        """
### Apéndice general
- **GAP** = Peso cartera AFP − Peso IPSA.
- **Delta GAP** = cambio mensual del GAP.
- **Delta GAP 3M / 6M** = suma rodante del flujo mensual.
- Todos se muestran como **%**.
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

# columnas auxiliares para nuevos gráficos
df_model = df_model.sort_values(["Nemo", "Fecha"]).copy()
df_model["Delta_GAP_3M"] = (
    df_model.groupby("Nemo")["Delta_GAP"]
    .transform(lambda s: s.rolling(3, min_periods=1).sum())
)
df_model["Delta_GAP_6M"] = (
    df_model.groupby("Nemo")["Delta_GAP"]
    .transform(lambda s: s.rolling(6, min_periods=1).sum())
)

st.success(
    f"OK | Última fecha (Hola Valores!I2): {last_date.date()} | "
    f"Filas modeladas: {metrics['rows']} | AUC: {metrics['AUC_mean']:.3f} | ACC: {metrics['ACC_mean']:.3f}"
)

# =========================================================
# CONTROLES
# =========================================================
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
    tickers = sorted(df_model["Nemo"].dropna().unique().tolist())
    sel_tickers = st.multiselect("Papeles a superponer (máx 8)", options=tickers, default=[])

d1 = pd.to_datetime(date_range[0])
d2 = pd.to_datetime(date_range[1])
dfh = df_model[(df_model["Fecha"] >= d1) & (df_model["Fecha"] <= d2)].copy()

tabs = st.tabs([
    "📈 Todos los papeles",
    "✅ Snapshot última fecha",
    "🏁 Ranking última fecha",
    "📊 Detalle por papel + eventos",
    "🟦 Heatmap",
    "📊 Flujo mensual",
    "📚 Acumulado 3M / 6M",
    "🌐 Breadth AFP",
    "📦 GAP 12M por papel"
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

# =========================================================
# TAB 6
# =========================================================
with tabs[5]:
    st.subheader("Flujo mensual — Top compras / ventas del mes")

    available_dates_flow = sorted(dfh["Fecha"].dropna().unique())
    default_idx_flow = len(available_dates_flow) - 1
    if last_date in available_dates_flow:
        default_idx_flow = available_dates_flow.index(last_date)

    sel_flow_date = st.selectbox(
        "Mes para ver Delta GAP",
        options=available_dates_flow,
        index=max(0, default_idx_flow),
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
        key="sel_flow_date"
    )

    flow_month = dfh[dfh["Fecha"] == pd.to_datetime(sel_flow_date)].copy()
    flow_month = flow_month.dropna(subset=["Nemo", "Delta_GAP"])

    top_n = st.slider("Cantidad de papeles a mostrar", min_value=5, max_value=30, value=12, step=1)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("**Top compras del mes**")
        top_buy = flow_month.sort_values("Delta_GAP", ascending=False).head(top_n).copy()
        top_buy = top_buy.sort_values("Delta_GAP", ascending=True)

        if not top_buy.empty:
            fig_buy = px.bar(
                top_buy,
                x="Delta_GAP",
                y="Nemo",
                orientation="h",
                text="Delta_GAP",
                hover_data=["GAP", "Fase", "Flujo_AFP", "CompraVenta_Fuerte"],
                title=f"Top compras — {pd.to_datetime(sel_flow_date).date()}"
            )
            fig_buy.update_traces(texttemplate="%{x:.2%}")
            fig_buy.update_layout(template="plotly_white", xaxis_tickformat=".1%")
            st.plotly_chart(fig_buy, use_container_width=True)
        else:
            st.info("No hay datos para ese mes.")

    with c_right:
        st.markdown("**Top ventas del mes**")
        top_sell = flow_month.sort_values("Delta_GAP", ascending=True).head(top_n).copy()
        top_sell = top_sell.sort_values("Delta_GAP", ascending=True)

        if not top_sell.empty:
            fig_sell = px.bar(
                top_sell,
                x="Delta_GAP",
                y="Nemo",
                orientation="h",
                text="Delta_GAP",
                hover_data=["GAP", "Fase", "Flujo_AFP", "CompraVenta_Fuerte"],
                title=f"Top ventas — {pd.to_datetime(sel_flow_date).date()}"
            )
            fig_sell.update_traces(texttemplate="%{x:.2%}")
            fig_sell.update_layout(template="plotly_white", xaxis_tickformat=".1%")
            st.plotly_chart(fig_sell, use_container_width=True)
        else:
            st.info("No hay datos para ese mes.")

    with st.expander("Apéndice — cómo leer este gráfico"):
        st.markdown(
            """
- Este gráfico ordena los papeles por **Delta GAP** en el mes elegido.
- **Top compras** = mayores aumentos de GAP en el mes.
- **Top ventas** = mayores caídas de GAP en el mes.
- Sirve para ver rápidamente dónde estuvo el mayor flujo AFP mensual.
"""
        )

# =========================================================
# TAB 7
# =========================================================
with tabs[6]:
    st.subheader("Acumulado de flujo 3M / 6M")

    acc_df = dfh.sort_values(["Nemo", "Fecha"]).copy()

    available_dates_acc = sorted(acc_df["Fecha"].dropna().unique())
    default_idx_acc = len(available_dates_acc) - 1
    if last_date in available_dates_acc:
        default_idx_acc = available_dates_acc.index(last_date)

    c1_acc, c2_acc, c3_acc = st.columns([1.2, 1.2, 1])
    with c1_acc:
        sel_acc_date = st.selectbox(
            "Mes acumulado",
            options=available_dates_acc,
            index=max(0, default_idx_acc),
            format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
            key="sel_acc_date"
        )
    with c2_acc:
        acc_metric = st.selectbox(
            "Ventana",
            options=["Delta_GAP_3M", "Delta_GAP_6M"],
            index=0,
            format_func=lambda x: "Acumulado 3M" if x == "Delta_GAP_3M" else "Acumulado 6M"
        )
    with c3_acc:
        top_acc_n = st.slider("Top N", min_value=5, max_value=30, value=15, step=1, key="top_acc_n")

    acc_month = acc_df[acc_df["Fecha"] == pd.to_datetime(sel_acc_date)].copy()
    acc_month = acc_month.dropna(subset=["Nemo", acc_metric])

    left, right = st.columns(2)

    with left:
        st.markdown("**Mayor acumulado comprador**")
        acc_buy = acc_month.sort_values(acc_metric, ascending=False).head(top_acc_n).copy()
        acc_buy = acc_buy.sort_values(acc_metric, ascending=True)

        if not acc_buy.empty:
            fig_acc_buy = px.bar(
                acc_buy,
                x=acc_metric,
                y="Nemo",
                orientation="h",
                text=acc_metric,
                hover_data=["GAP", "Delta_GAP", "Fase", "Flujo_AFP"],
                title=f"{'3M' if acc_metric == 'Delta_GAP_3M' else '6M'} acumulado comprador — {pd.to_datetime(sel_acc_date).date()}"
            )
            fig_acc_buy.update_traces(texttemplate="%{x:.2%}")
            fig_acc_buy.update_layout(template="plotly_white", xaxis_tickformat=".1%")
            st.plotly_chart(fig_acc_buy, use_container_width=True)
        else:
            st.info("No hay datos acumulados para ese mes.")

    with right:
        st.markdown("**Mayor acumulado vendedor**")
        acc_sell = acc_month.sort_values(acc_metric, ascending=True).head(top_acc_n).copy()
        acc_sell = acc_sell.sort_values(acc_metric, ascending=True)

        if not acc_sell.empty:
            fig_acc_sell = px.bar(
                acc_sell,
                x=acc_metric,
                y="Nemo",
                orientation="h",
                text=acc_metric,
                hover_data=["GAP", "Delta_GAP", "Fase", "Flujo_AFP"],
                title=f"{'3M' if acc_metric == 'Delta_GAP_3M' else '6M'} acumulado vendedor — {pd.to_datetime(sel_acc_date).date()}"
            )
            fig_acc_sell.update_traces(texttemplate="%{x:.2%}")
            fig_acc_sell.update_layout(template="plotly_white", xaxis_tickformat=".1%")
            st.plotly_chart(fig_acc_sell, use_container_width=True)
        else:
            st.info("No hay datos acumulados para ese mes.")

    with st.expander("Apéndice — cómo leer acumulados 3M / 6M"):
        st.markdown(
            """
- **Delta_GAP_3M** = suma del Delta GAP de los últimos 3 meses.
- **Delta_GAP_6M** = suma del Delta GAP de los últimos 6 meses.
- Sirve para filtrar ruido mensual y ver tendencias más persistentes.
- Valores altos sugieren compras acumuladas; valores bajos sugieren ventas acumuladas.
"""
        )

# =========================================================
# TAB 8
# =========================================================
with tabs[7]:
    st.subheader("Breadth AFP del mercado")

    breadth_df = dfh.dropna(subset=["Fecha"]).copy()

    breadth = (
        breadth_df.groupby("Fecha")
        .agg(
            pct_comprando=("Delta_GAP", lambda s: (s > 0).mean()),
            pct_vendiendo=("Delta_GAP", lambda s: (s < 0).mean()),
            pct_gap_pos=("GAP", lambda s: (s > 0).mean()),
            pct_verde=("Semaforo", lambda s: (s == "🟢").mean() if len(s) else 0)
        )
        .reset_index()
    )

    fig_breadth = go.Figure()
    fig_breadth.add_trace(go.Scatter(
        x=breadth["Fecha"], y=breadth["pct_comprando"],
        mode="lines", name="% papeles comprando (Delta_GAP > 0)"
    ))
    fig_breadth.add_trace(go.Scatter(
        x=breadth["Fecha"], y=breadth["pct_vendiendo"],
        mode="lines", name="% papeles vendiendo (Delta_GAP < 0)"
    ))
    fig_breadth.add_trace(go.Scatter(
        x=breadth["Fecha"], y=breadth["pct_gap_pos"],
        mode="lines", name="% papeles con GAP > 0"
    ))
    fig_breadth.add_trace(go.Scatter(
        x=breadth["Fecha"], y=breadth["pct_verde"],
        mode="lines", name="% papeles en verde"
    ))

    fig_breadth.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title="Difusión / Breadth AFP del mercado",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1]
    )
    st.plotly_chart(fig_breadth, use_container_width=True)

    latest_breadth = breadth.sort_values("Fecha").iloc[-1:]
    if not latest_breadth.empty:
        lb = latest_breadth.iloc[0]
        st.markdown(
            f"""
**Último dato**
- % papeles comprando: **{lb['pct_comprando']:.0%}**
- % papeles vendiendo: **{lb['pct_vendiendo']:.0%}**
- % papeles con GAP positivo: **{lb['pct_gap_pos']:.0%}**
- % papeles en verde: **{lb['pct_verde']:.0%}**
"""
        )

    with st.expander("Apéndice — cómo leer el breadth AFP"):
        st.markdown(
            """
- Este gráfico muestra si el flujo AFP está siendo **generalizado** o **concentrado**.
- Si sube el **% de papeles comprando**, el impulso es más amplio.
- Si solo suben pocos papeles, el breadth se ve más débil aunque algunos nombres destaquen.
- El **% con GAP positivo** ayuda a ver si la sobreponderación AFP está extendida.
"""
        )

# =========================================================
# TAB 9
# =========================================================
with tabs[8]:
    st.subheader("GAP últimos 12 meses por papel")

    papers_12m = sorted(dfh["Nemo"].dropna().unique().tolist())
    paper_12m = st.selectbox("Selecciona papel para ver GAP 12M", papers_12m, key="paper_12m")

    d12 = dfh[dfh["Nemo"] == paper_12m].sort_values("Fecha").copy().tail(12)

    if d12.empty:
        st.info("No hay datos para ese papel.")
    else:
        fig_gap12 = px.bar(
            d12,
            x="Fecha",
            y="GAP",
            text="GAP",
            hover_data=["Delta_GAP", "Fase", "Flujo_AFP", "CompraVenta_Fuerte"],
            title=f"{paper_12m} — GAP últimos 12 meses"
        )
        fig_gap12.add_hline(y=0, line_width=1)
        fig_gap12.update_traces(texttemplate="%{y:.2%}")
        fig_gap12.update_layout(template="plotly_white", yaxis_tickformat=".1%")
        st.plotly_chart(fig_gap12, use_container_width=True)

        if "Delta_GAP" in d12.columns:
            fig_delta12 = px.bar(
                d12,
                x="Fecha",
                y="Delta_GAP",
                text="Delta_GAP",
                hover_data=["GAP", "Fase", "Flujo_AFP", "CompraVenta_Fuerte"],
                title=f"{paper_12m} — Delta GAP últimos 12 meses"
            )
            fig_delta12.add_hline(y=0, line_width=1)
            fig_delta12.update_traces(texttemplate="%{y:.2%}")
            fig_delta12.update_layout(template="plotly_white", yaxis_tickformat=".1%")
            st.plotly_chart(fig_delta12, use_container_width=True)

    with st.expander("Apéndice — cómo leer GAP 12M por papel"):
        st.markdown(
            """
- El gráfico muestra el **GAP mensual** del papel en los últimos 12 meses.
- Sirve para ver si la AFP viene aumentando o reduciendo su sobre/subponderación.
- Abajo se muestra también el **Delta GAP** mensual para ver la velocidad del cambio.
"""
        )
