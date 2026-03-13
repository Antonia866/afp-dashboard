import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



st.set_page_config(
    page_title="AFP GAP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

APP_CACHE_VERSION = "v7_fix_unpack_and_debug"

st.markdown("""
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
}
.main-title {
    font-size: 2.0rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 0.98rem;
    color: #475569;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.5rem;
}
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 16px 18px 12px 18px;
    box-shadow: 0 2px 10px rgba(15,23,42,0.04);
    margin-bottom: 12px;
}
.kpi-label {
    font-size: 0.82rem;
    color: #64748B;
    margin-bottom: 0.25rem;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 0.15rem;
}
.small-note {
    font-size: 0.84rem;
    color: #64748B;
}
hr.soft {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 0.8rem 0 1rem 0;
}
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AFP GAP Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Flujos AFP, GAP vs IPSA, señales, probabilidad de compra y ranking institucional.</div>',
    unsafe_allow_html=True
)


def fmt_pct(x, digits=2):
    try:
        return f"{float(x):.{digits}f}%"
    except Exception:
        return ""


def signal_color_map():
    return {
        "Compra fuerte": "#15803d",
        "Entrada activas": "#22c55e",
        "Entrada seguidoras": "#a3e635",
        "Comprando": "#d9f99d",
        "Manteniendo": "#fde68a",
        "Vendiendo": "#fdba74",
        "Salida seguidoras": "#fb7185",
        "Salida activas": "#ef4444",
        "Venta fuerte": "#b91c1c",
    }


def metric_box(label, value, sub=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_df(df, formats=None):
    sty = df.style
    if formats:
        sty = sty.format(formats)
    return sty


def build_gap_vs_hist(df_model: pd.DataFrame, sel_date) -> pd.DataFrame:
    sel_date = pd.to_datetime(sel_date)

    hist = (
        df_model[df_model["Fecha"] < sel_date]
        .groupby("Nemo", as_index=False)
        .agg(
            Gap_Promedio_Historico=("Gap", "mean"),
            Gap_P25=("Gap", lambda x: x.quantile(0.25)),
            Gap_P75=("Gap", lambda x: x.quantile(0.75)),
            N_Hist=("Gap", "count"),
        )
    )

    cur = (
        df_model[df_model["Fecha"] == sel_date][["Nemo", "Gap"]]
        .rename(columns={"Gap": "Gap_Actual"})
        .copy()
    )

    out = pd.merge(cur, hist, on="Nemo", how="left")
    out["Gap_Promedio_Historico"] = out["Gap_Promedio_Historico"].fillna(0)
    out["Gap_P25"] = out["Gap_P25"].fillna(out["Gap_Promedio_Historico"])
    out["Gap_P75"] = out["Gap_P75"].fillna(out["Gap_Promedio_Historico"])
    out["N_Hist"] = out["N_Hist"].fillna(0)

    out = out.sort_values("Gap_Actual", ascending=False).reset_index(drop=True)
    return out


def plot_gap_vs_hist(df_comp: pd.DataFrame, title_fecha: str):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_comp["Nemo"],
        y=df_comp["Gap_Actual"],
        name=f"GAP {title_fecha}",
        marker_color="#1565C0"
    ))

    fig.add_trace(go.Bar(
        x=df_comp["Nemo"],
        y=df_comp["Gap_Promedio_Historico"],
        name="Promedio histórico",
        marker_color="#90CAF9"
    ))

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title=f"GAP por papel vs Promedio histórico — {title_fecha}",
        xaxis_title="",
        yaxis_title="GAP",
        yaxis_tickformat=".2%",
        height=460,
        legend_title_text="Serie",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.add_hline(y=0, line_width=1, line_color="#6B7280")
    return fig


colA, colB = st.columns([1.2, 1])

with colA:
    uploaded = st.file_uploader("Sube el Excel (.xlsx)", type=["xlsx"])

with colB:
    file_path = st.text_input("O pega la ruta local del Excel (.xlsx)", value="")

run = st.button("Cargar y ejecutar", type="primary")


@st.cache_data(show_spinner=False)
def cached_build(xls_source, cache_version):
    return build_outputs(xls_source)


if not run:
    st.info("Sube el Excel o pega la ruta local, luego presiona Cargar y ejecutar.")
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
            st.error("Pegaste una carpeta. Deja solo 1 archivo .xlsx dentro o pega la ruta exacta del Excel.")
            st.stop()
    else:
        xls_source = file_path
else:
    st.error("Debes subir el Excel o pegar la ruta local.")
    st.stop()

try:
    with st.spinner("Procesando datos..."):
        result = cached_build(xls_source, APP_CACHE_VERSION)

    if not isinstance(result, tuple):
        raise ValueError(f"build_outputs devolvió un tipo inesperado: {type(result)}")

    if len(result) != 13:
        raise ValueError(f"build_outputs devolvió {len(result)} elementos y app.py esperaba 13.")

    (
        df_raw,
        df_model,
        snap_last,
        metrics,
        events,
        last_date,
        df_ipsa,
        top_compras,
        top_ventas,
        ranking_entrada,
        ranking_salida,
        appendix,
        primera_fecha,
    ) = result

except Exception as e:
    st.error("Se produjo un error al construir el dashboard.")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    st.stop()

st.success(
    f"OK | Última fecha tomada desde Hola Valores!I2: {last_date.date()} | "
    f"Primera fecha: {primera_fecha.date() if primera_fecha is not None else 'N/D'} | "
    f"Filas modelo: {metrics['rows']} | "
    f"AUC: {metrics['AUC_mean']:.3f} | ACC: {metrics['ACC_mean']:.3f}"
)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    metric_box("Papeles", f"{len(snap_last):,.0f}", "universo último mes")
with k2:
    metric_box("Prob. compra promedio", fmt_pct(snap_last["Prob_Compra_AFP_ProxMes"].mean()), "estimación modelo")
with k3:
    metric_box("Compras fuertes", f"{int((snap_last['CompraVenta_Fuerte'] == 'Compra fuerte').sum())}", "señales fuertes")
with k4:
    metric_box("Ventas fuertes", f"{int((snap_last['CompraVenta_Fuerte'] == 'Venta fuerte').sum())}", "señales fuertes")
with k5:
    metric_box("Flow Rank promedio", f"{snap_last['AFP_Flow_Rank'].mean():.1f}", "ranking institucional")

min_d, max_d = df_model["Fecha"].min(), df_model["Fecha"].max()
default_start = max(min_d, max_d - pd.DateOffset(months=36))

st.markdown('<hr class="soft">', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.2, 1.1, 1.1])

with c1:
    date_range = st.slider(
        "Rango histórico",
        min_value=min_d.to_pydatetime(),
        max_value=max_d.to_pydatetime(),
        value=(default_start.to_pydatetime(), max_d.to_pydatetime())
    )

with c2:
    tickers = sorted(df_model["Nemo"].unique().tolist())
    sel_tickers = st.multiselect("Papeles a comparar (máx 8)", options=tickers, default=[])

with c3:
    available_dates = sorted(df_model["Fecha"].dropna().unique())
    default_idx = max(0, len(available_dates) - 1)
    if last_date in available_dates:
        default_idx = available_dates.index(last_date)

    sel_date = st.selectbox(
        "Fecha de análisis",
        options=available_dates,
        index=default_idx,
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
    )

d1 = pd.to_datetime(date_range[0])
d2 = pd.to_datetime(date_range[1])

dfh = df_model[(df_model["Fecha"] >= d1) & (df_model["Fecha"] <= d2)].copy()
snap_date = df_model[df_model["Fecha"] == pd.to_datetime(sel_date)].copy()
gap_vs_hist = build_gap_vs_hist(df_model, sel_date)

tabs = st.tabs([
    "🏠 Dashboard",
    "📊 Snapshot última fecha",
    "📈 Ranking entradas / salidas",
    "💰 Top compras / Top ventas",
    "🧾 Detalle por papel",
    "🟦 Heatmap",
    "📘 Apéndice metodología",
])

with tabs[0]:
    st.markdown('<div class="section-title">Vista general</div>', unsafe_allow_html=True)

    e1, e2, e3 = st.columns([1.1, 1.1, 1.2])

    buy_count = int((snap_last["Senal"].isin(["Compra fuerte", "Entrada activas", "Entrada seguidoras", "Comprando"])).sum())
    hold_count = int((snap_last["Senal"] == "Manteniendo").sum())
    sell_count = int((snap_last["Senal"].isin(["Venta fuerte", "Salida activas", "Salida seguidoras", "Vendiendo"])).sum())

    with e1:
        metric_box("Sesgo comprador", f"{buy_count}", "papeles con señal favorable")
    with e2:
        metric_box("Mantener", f"{hold_count}", "zona intermedia")
    with e3:
        metric_box("Sesgo vendedor", f"{sell_count}", "papeles con señal de salida")

    r0c1, r0c2 = st.columns([1.35, 1])

    with r0c1:
        st.plotly_chart(
            plot_gap_vs_hist(gap_vs_hist, pd.to_datetime(sel_date).strftime("%Y-%m-%d")),
            use_container_width=True
        )

    with r0c2:
        st.markdown("#### Top ranking institucional")
        rank_view = snap_last.sort_values(
            ["AFP_Flow_Rank", "Prob_Compra_AFP_ProxMes", "Delta_Gap"],
            ascending=False
        ).head(15).copy()

        fig_rank = px.bar(
            rank_view.sort_values("AFP_Flow_Rank", ascending=True),
            x="AFP_Flow_Rank",
            y="Nemo",
            orientation="h",
            color="Senal",
            color_discrete_map=signal_color_map(),
            title="AFP Flow Rank"
        )
        fig_rank.update_layout(
            template="plotly_white",
            height=460,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    r1c1, r1c2 = st.columns([1, 1])

    with r1c1:
        st.markdown("#### Distribución del universo")
        dist = dfh.groupby("Fecha")["Gap"].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
        dist.columns = ["Fecha", "p25", "p50", "p75"]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(x=dist["Fecha"], y=dist["p50"], mode="lines", name="Mediana"))
        fig_dist.add_trace(go.Scatter(x=dist["Fecha"], y=dist["p75"], mode="lines", name="p75", opacity=0.45))
        fig_dist.add_trace(go.Scatter(x=dist["Fecha"], y=dist["p25"], mode="lines", name="p25", opacity=0.45, fill="tonexty"))

        for t in sel_tickers[:8]:
            dt_ = dfh[dfh["Nemo"] == t]
            fig_dist.add_trace(go.Scatter(x=dt_["Fecha"], y=dt_["Gap"], mode="lines", name=t))

        fig_dist.add_hline(y=0, line_width=1, line_color="#94A3B8")
        fig_dist.update_layout(
            template="plotly_white",
            hovermode="x unified",
            title="p25 / mediana / p75 + papeles seleccionados",
            height=450,
            yaxis_tickformat=".2%",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with r1c2:
        st.markdown("#### Resumen ejecutivo del mes")
        top_signal = snap_last[[
            "Nemo", "Senal", "Prob_Compra_AFP_ProxMes",
            "Prob_Entrada_AFP", "Prob_Salida_AFP", "Gap", "Delta_Gap"
        ]].copy()

        top_signal = top_signal.sort_values(
            ["Prob_Compra_AFP_ProxMes", "Prob_Entrada_AFP", "Delta_Gap"],
            ascending=False
        ).head(18)

        st.dataframe(
            style_df(
                top_signal,
                {
                    "Prob_Compra_AFP_ProxMes": "{:.2f}%",
                    "Prob_Entrada_AFP": "{:.2f}%",
                    "Prob_Salida_AFP": "{:.2f}%",
                    "Gap": "{:.2%}",
                    "Delta_Gap": "{:.2%}",
                }
            ),
            use_container_width=True,
            height=450
        )

with tabs[1]:
    st.markdown('<div class="section-title">Snapshot última fecha</div>', unsafe_allow_html=True)
    st.caption(f"Fecha de corte: {last_date.date()}")

    show = [
        "Semaforo", "Nemo", "PesoAFP", "PesoIPSA", "Gap", "Delta_Gap",
        "Posicion", "Movimiento", "Senal", "Flujo_AFP", "CompraVenta_Fuerte",
        "Prob_Compra_AFP_ProxMes", "Prob_Entrada_AFP", "Prob_Salida_AFP",
        "AFP_Flow_Rank", "Accion_Tactica", "Accion_Relativa", "Recomendacion_Timing",
    ]
    show = [c for c in show if c in snap_last.columns]

    order = snap_last.copy()
    order["__sort"] = order["Recomendacion_Timing"].map({
        "Comprar en T": 0,
        "Comprar / Mantener": 1,
        "Mantener": 2,
        "Reducir suave": 3,
        "Vender / Reducir": 4,
    }).fillna(9)

    order = order.sort_values(
        ["__sort", "Prob_Compra_AFP_ProxMes", "AFP_Flow_Rank", "Delta_Gap"],
        ascending=[True, False, False, False]
    ).drop(columns=["__sort"])

    st.dataframe(
        style_df(
            order[show],
            {
                "PesoAFP": "{:.2%}",
                "PesoIPSA": "{:.2%}",
                "Gap": "{:.2%}",
                "Delta_Gap": "{:.2%}",
                "Prob_Compra_AFP_ProxMes": "{:.2f}%",
                "Prob_Entrada_AFP": "{:.2f}%",
                "Prob_Salida_AFP": "{:.2f}%",
                "AFP_Flow_Rank": "{:.1f}",
            }
        ),
        use_container_width=True,
        height=720
    )

with tabs[2]:
    st.markdown('<div class="section-title">Ranking institucional</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        cols_in = [
            "Nemo", "Semaforo", "Senal", "Flujo_AFP",
            "Prob_Compra_AFP_ProxMes", "Prob_Entrada_AFP",
            "AFP_Flow_Rank", "Gap", "Delta_Gap", "Accion_Tactica"
        ]
        cols_in = [c for c in cols_in if c in ranking_entrada.columns]

        st.dataframe(
            style_df(
                ranking_entrada.head(25)[cols_in],
                {
                    "Prob_Compra_AFP_ProxMes": "{:.2f}%",
                    "Prob_Entrada_AFP": "{:.2f}%",
                    "AFP_Flow_Rank": "{:.1f}",
                    "Gap": "{:.2%}",
                    "Delta_Gap": "{:.2%}",
                }
            ),
            use_container_width=True,
            height=620
        )

    with right:
        cols_out = [
            "Nemo", "Semaforo", "Senal", "Flujo_AFP",
            "Prob_Salida_AFP", "Gap", "Delta_Gap",
            "CompraVenta_Fuerte", "Accion_Tactica"
        ]
        cols_out = [c for c in cols_out if c in ranking_salida.columns]

        st.dataframe(
            style_df(
                ranking_salida.head(25)[cols_out],
                {
                    "Prob_Salida_AFP": "{:.2f}%",
                    "Gap": "{:.2%}",
                    "Delta_Gap": "{:.2%}",
                }
            ),
            use_container_width=True,
            height=620
        )

with tabs[3]:
    st.markdown('<div class="section-title">Top compras / Top ventas del mes</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        cols_buy = [
            "Nemo", "Semaforo", "Senal", "Flujo_AFP",
            "Prob_Compra_AFP_ProxMes", "Prob_Entrada_AFP",
            "AFP_Flow_Rank", "Gap", "Delta_Gap", "Recomendacion_Timing"
        ]
        cols_buy = [c for c in cols_buy if c in top_compras.columns]

        st.dataframe(
            style_df(
                top_compras[cols_buy],
                {
                    "Prob_Compra_AFP_ProxMes": "{:.2f}%",
                    "Prob_Entrada_AFP": "{:.2f}%",
                    "AFP_Flow_Rank": "{:.1f}",
                    "Gap": "{:.2%}",
                    "Delta_Gap": "{:.2%}",
                }
            ),
            use_container_width=True,
            height=560
        )

    with right:
        cols_sell = [
            "Nemo", "Semaforo", "Senal", "Flujo_AFP",
            "Prob_Salida_AFP", "Gap", "Delta_Gap",
            "CompraVenta_Fuerte", "Recomendacion_Timing"
        ]
        cols_sell = [c for c in cols_sell if c in top_ventas.columns]

        st.dataframe(
            style_df(
                top_ventas[cols_sell],
                {
                    "Prob_Salida_AFP": "{:.2f}%",
                    "Gap": "{:.2%}",
                    "Delta_Gap": "{:.2%}",
                }
            ),
            use_container_width=True,
            height=560
        )

with tabs[4]:
    st.markdown('<div class="section-title">Detalle por papel</div>', unsafe_allow_html=True)

    paper = st.selectbox("Selecciona papel", sorted(dfh["Nemo"].unique().tolist()))
    d = dfh[dfh["Nemo"] == paper].sort_values("Fecha").copy()

    last_row = df_model[(df_model["Nemo"] == paper) & (df_model["Fecha"] == last_date)]
    if len(last_row):
        lr = last_row.iloc[0]
        cA, cB, cC = st.columns(3)
        with cA:
            metric_box("Papel", paper, "")
        with cB:
            metric_box("Señal", lr.get("Senal", ""), "")
        with cC:
            metric_box("Timing", lr.get("Recomendacion_Timing", ""), "")

    p1, p2 = st.columns(2)

    with p1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=d["Fecha"], y=d["Gap"], mode="lines", name="Gap"))
        if "MA_3" in d.columns:
            fig1.add_trace(go.Scatter(x=d["Fecha"], y=d["MA_3"], mode="lines", name="Media 3M"))
        fig1.add_hline(y=0, line_width=1, line_color="#94A3B8")
        fig1.update_layout(
            template="plotly_white",
            hovermode="x unified",
            title=f"{paper} | Gap histórico",
            height=420,
            yaxis_tickformat=".2%",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with p2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=d["Fecha"], y=d["Delta_Gap"], mode="lines", name="Delta Gap"))
        fig2.add_hline(y=0, line_width=1, line_color="#94A3B8")
        fig2.update_layout(
            template="plotly_white",
            hovermode="x unified",
            title=f"{paper} | Cambio mensual del Gap",
            height=420,
            yaxis_tickformat=".2%",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=d["Fecha"],
        y=d["Prob_Compra_AFP_ProxMes"],
        mode="lines",
        name="Prob. compra AFP próximo mes"
    ))
    fig3.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=f"{paper} | Probabilidad de compra AFP próximo mes",
        yaxis_title="%",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

with tabs[5]:
    st.markdown('<div class="section-title">Heatmap histórico</div>', unsafe_allow_html=True)

    metric = st.selectbox("Métrica", ["Gap", "Delta_Gap", "Prob_Compra_AFP_ProxMes"], index=0)

    if metric not in dfh.columns:
        st.warning(f"No existe {metric}.")
    else:
        last_vals = dfh[dfh["Fecha"] == dfh["Fecha"].max()][["Nemo", metric]].dropna()
        ordered = last_vals.sort_values(metric, ascending=False)["Nemo"].tolist()

        pivot = dfh.pivot_table(index="Nemo", columns="Fecha", values=metric, aggfunc="last")
        if ordered:
            pivot = pivot.reindex(ordered)

        fig = px.imshow(
            pivot,
            aspect="auto",
            title=f"Heatmap histórico: {metric}",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(template="plotly_white", height=720, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tabs[6]:
    st.markdown('<div class="section-title">Apéndice metodología</div>', unsafe_allow_html=True)
    st.dataframe(appendix["reglas_senales"], use_container_width=True, height=360)
    st.dataframe(appendix["probabilidades"], use_container_width=True, height=260)
    st.dataframe(appendix["acciones"], use_container_width=True, height=360)
    st.dataframe(appendix["metricas"], use_container_width=True, height=260)
    st.dataframe(appendix["rangos"], use_container_width=True, height=220)

