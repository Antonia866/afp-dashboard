"""
app.py — Dashboard Sura Flows
=============================
Streamlit app con:
- Panorama Ejecutivo (landing): tabla tipo research con barras AFP y FFMM.
- Toggle global AFP / FFMM / Ambos en sidebar.
- 11 tabs analíticas.
- Export Excel.
"""

import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sura_pipeline import build_outputs

st.set_page_config(page_title="Sura Flows Dashboard", layout="wide")
st.title("📊 Sura Flows — AFP & FFMM vs IPSA")

# ============================================================
# HELPERS VISUALES
# ============================================================
def fmt_pct(x, dec=2):
    try:
        return f"{x:.{dec}%}"
    except Exception:
        return x

def fmt_bps(x):
    try:
        return f"{x*10000:.0f}"
    except Exception:
        return x

SIGNAL_ORDER = ["BUY_FUERTE", "BUY", "BUY_LIGHT", "HOLD", "SELL_LIGHT", "SELL", "SELL_FUERTE"]
SIGNAL_RANK = {s: i for i, s in enumerate(SIGNAL_ORDER)}

SIGNAL_COLOR = {
    "BUY_FUERTE": "#0a7a0a",
    "BUY": "#2ecc71",
    "BUY_LIGHT": "#a5e8c6",
    "HOLD": "#bdc3c7",
    "SELL_LIGHT": "#f5b7b1",
    "SELL": "#e74c3c",
    "SELL_FUERTE": "#922b21",
}

# ============================================================
# INPUT
# ============================================================
with st.sidebar:
    st.markdown("### 📁 Archivo")
    uploaded = st.file_uploader("Sube Request_Sura.xlsx", type=["xlsx"])
    file_path = st.text_input("O ruta local", value="")
    run = st.button("Cargar y ejecutar", type="primary")

# Mantener el archivo entre reruns usando session_state
if "xls_loaded" not in st.session_state:
    st.session_state["xls_loaded"] = None

# Si clickea el botón, guardar el source
if run:
    if uploaded is not None:
        st.session_state["xls_loaded"] = uploaded
    elif file_path.strip():
        st.session_state["xls_loaded"] = file_path.strip()

xls_source = st.session_state["xls_loaded"]

if xls_source is None:
    st.info("👈 Sube el Excel en el sidebar, luego presiona **Cargar y ejecutar**.")
    st.stop()

@st.cache_data(show_spinner=False)
def cached_build(_xls_source):
    return build_outputs(_xls_source)

with st.spinner("Procesando datos..."):
    try:
        out = cached_build(xls_source)
    except Exception as e:
        st.error(f"Error al procesar el Excel: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

df = out["df"]
snap_last = out["snap_last"].copy()
events_afp = out["events_afp"]
events_ffmm = out["events_ffmm"]
last_date = out["last_date"]
panel = out["panel"]

# ============================================================
# SIDEBAR: TOGGLE GLOBAL + RESUMEN
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown(f"**Última fecha:** `{last_date.date()}`")
    st.markdown(f"**Tickers:** `{snap_last['Ticker'].nunique()}`")

    st.markdown("### 🎚️ Universo")
    universo = st.radio(
        "Fondo a visualizar",
        options=["AFP", "FFMM", "Ambos"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("### 📋 Resumen ejecutivo")
    if universo == "AFP":
        top_buy = snap_last[snap_last["Senal_AFP"].isin(["BUY_FUERTE", "BUY"])].nlargest(3, "Delta_GAP_AFP")
        top_sell = snap_last[snap_last["Senal_AFP"].isin(["SELL_FUERTE", "SELL"])].nsmallest(3, "Delta_GAP_AFP")
        st.markdown("**🟢 Top BUY AFP:**")
        if len(top_buy):
            for _, r in top_buy.iterrows():
                st.markdown(f"- `{r['Ticker']}` ΔGAP `{fmt_bps(r['Delta_GAP_AFP'])}bps`")
        else:
            st.caption("— sin BUY fuerte este mes")
        st.markdown("**🔴 Top SELL AFP:**")
        if len(top_sell):
            for _, r in top_sell.iterrows():
                st.markdown(f"- `{r['Ticker']}` ΔGAP `{fmt_bps(r['Delta_GAP_AFP'])}bps`")
        else:
            st.caption("— sin SELL fuerte este mes")
    elif universo == "FFMM":
        top_buy = snap_last[snap_last["Senal_FFMM"].isin(["BUY_FUERTE", "BUY"])].nlargest(3, "Delta_GAP_FFMM")
        top_sell = snap_last[snap_last["Senal_FFMM"].isin(["SELL_FUERTE", "SELL"])].nsmallest(3, "Delta_GAP_FFMM")
        st.markdown("**🟢 Top BUY FFMM:**")
        if len(top_buy):
            for _, r in top_buy.iterrows():
                st.markdown(f"- `{r['Ticker']}` ΔGAP `{fmt_bps(r['Delta_GAP_FFMM'])}bps`")
        else:
            st.caption("— sin BUY fuerte este mes")
        st.markdown("**🔴 Top SELL FFMM:**")
        if len(top_sell):
            for _, r in top_sell.iterrows():
                st.markdown(f"- `{r['Ticker']}` ΔGAP `{fmt_bps(r['Delta_GAP_FFMM'])}bps`")
        else:
            st.caption("— sin SELL fuerte este mes")
    else:
        st.markdown("**🔀 Top divergencias AFP-FFMM:**")
        div = snap_last.copy()
        div["abs_div"] = div["Divergencia_GAP"].abs()
        for _, r in div.nlargest(5, "abs_div").iterrows():
            st.markdown(f"- `{r['Ticker']}` AFP `{fmt_pct(r['GAP_AFP'])}` / FFMM `{fmt_pct(r['GAP_FFMM'])}`")

        st.markdown("**⚡ Liderazgo del mes:**")
        LID_LABELS = {
            "Consenso_Compra": "🟢 Consenso compra",
            "Consenso_Venta": "🔴 Consenso venta",
            "Lidera_AFP": "🏎️ AFP lidera",
            "Lidera_FFMM": "🏎️ FFMM lidera",
            "Divergencia_Flujos": "⚔️ Divergencia",
        }
        for lid_key, lid_name in LID_LABELS.items():
            tickers_lid = snap_last[snap_last["Liderazgo_del_mes"] == lid_key]["Ticker"].tolist()
            if len(tickers_lid) == 0:
                continue
            st.markdown(f"**{lid_name}** ({len(tickers_lid)}):")
            # Mostrar hasta 8 tickers, si hay más agregar "...+N"
            to_show = tickers_lid[:8]
            extra = len(tickers_lid) - len(to_show)
            line = ", ".join([f"`{t}`" for t in to_show])
            if extra > 0:
                line += f" +{extra}"
            st.markdown(line)

    st.markdown("---")
    # Export Excel
    def _export_excel():
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            snap_last.to_excel(w, sheet_name="Snapshot", index=False)
            events_afp.to_excel(w, sheet_name="Events_AFP", index=False)
            events_ffmm.to_excel(w, sheet_name="Events_FFMM", index=False)
        return buf.getvalue()

    st.download_button(
        "💾 Export Excel",
        data=_export_excel(),
        file_name=f"sura_snapshot_{last_date.date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# CONTROLES GLOBALES
# ============================================================
min_d, max_d = df["Fecha"].min(), df["Fecha"].max()
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
    tickers_all = sorted(df["Ticker"].dropna().unique().tolist())
    sel_tickers = st.multiselect("Papeles a superponer (máx 8)", options=tickers_all, default=[], max_selections=8)

d1 = pd.to_datetime(date_range[0])
d2 = pd.to_datetime(date_range[1])
dfh = df[(df["Fecha"] >= d1) & (df["Fecha"] <= d2)].copy()

# ============================================================
# PANORAMA EJECUTIVO (landing - siempre visible, arriba de las tabs)
# ============================================================
st.markdown(f"## 🧭 Panorama Ejecutivo — {last_date.date()}")

panorama = snap_last[["Ticker", "Sector", "Peso_IPSA", "Peso_AFP", "GAP_AFP",
                      "Peso_FFMM", "GAP_FFMM"]].copy()
# Solo tickers con peso IPSA>0 (miembros activos) o presencia AFP/FFMM significativa
panorama = panorama[
    (panorama["Peso_IPSA"] > 0) |
    (panorama["Peso_AFP"].abs() > 0.001) |
    (panorama["Peso_FFMM"].abs() > 0.001)
].copy()

# Ordenar por peso IPSA descendente
panorama = panorama.sort_values("Peso_IPSA", ascending=False).reset_index(drop=True)

# Convertir GAPs a bps
panorama["Diff_AFP_bps"] = (panorama["GAP_AFP"] * 10000).round(0).astype(int)
panorama["Diff_FFMM_bps"] = (panorama["GAP_FFMM"] * 10000).round(0).astype(int)

# Columnas a mostrar
panorama_show = panorama[[
    "Ticker", "Sector", "Peso_IPSA", "Peso_AFP", "Diff_AFP_bps", "Peso_FFMM", "Diff_FFMM_bps"
]].rename(columns={
    "Peso_IPSA": "IPSA Weight",
    "Peso_AFP": "Pension Fund",
    "Diff_AFP_bps": "Diff AFP (bps)",
    "Peso_FFMM": "Mutual Fund",
    "Diff_FFMM_bps": "Diff FFMM (bps)"
})

# Render con barras embebidas tipo research
# Cast explícito a int() puro de Python para evitar TypeError en json.dumps
max_abs_afp = int(max(panorama_show["Diff AFP (bps)"].abs().max(), 1))
max_abs_ffmm = int(max(panorama_show["Diff FFMM (bps)"].abs().max(), 1))

# Asegurar que Diff también sean int puros
panorama_show["Diff AFP (bps)"] = panorama_show["Diff AFP (bps)"].astype(int)
panorama_show["Diff FFMM (bps)"] = panorama_show["Diff FFMM (bps)"].astype(int)

st.dataframe(
    panorama_show,
    use_container_width=True,
    height=620,
    hide_index=True,
    column_config={
        "IPSA Weight": st.column_config.NumberColumn("IPSA Weight", format="%.1f%%"),
        "Pension Fund": st.column_config.NumberColumn("Pension Fund", format="%.1f%%"),
        "Mutual Fund": st.column_config.NumberColumn("Mutual Fund", format="%.1f%%"),
        "Diff AFP (bps)": st.column_config.ProgressColumn(
            "Diff AFP (bps)",
            format="%d",
            min_value=-max_abs_afp,
            max_value=max_abs_afp,
        ),
        "Diff FFMM (bps)": st.column_config.ProgressColumn(
            "Diff FFMM (bps)",
            format="%d",
            min_value=-max_abs_ffmm,
            max_value=max_abs_ffmm,
        ),
    }
)
# Los pesos están en escala 0-1 decimal; Streamlit con "%.1f%%" ya los renderiza como %.

st.caption("📌 Panorama ejecutivo: diferencia absoluta AFP vs IPSA y FFMM vs IPSA en **bps**, ordenado por peso IPSA descendente. Barras verdes = sobreponderado, rojas = infraponderado.")

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tabs = st.tabs([
    "💰 Flujo Agregado",
    "📈 Posicionamiento vs historia",
    "✅ Snapshot",
    "🏁 Ranking",
    "📊 Detalle por papel",
    "🟦 Heatmap",
    "📊 Flujo mensual / 3M / 6M",
    "🌐 Breadth",
    "🎯 Scatter AFP vs FFMM",
    "⚡ Liderazgo",
    "🏢 Sectorial",
    "🔄 Persistencia"
])


# ------------------------------------------------------------
# Error boundary por tab: si una tab falla no rompe las demás
# ------------------------------------------------------------
from contextlib import contextmanager

@contextmanager
def safe_tab(tab_ctx, tab_name):
    """Envuelve el contenido de una tab para que un error no tumbe el app entero."""
    with tab_ctx:
        try:
            yield
        except Exception as e:
            st.error(f"❌ Error en tab **{tab_name}**: {e}")
            import traceback
            with st.expander("Ver traceback completo"):
                st.code(traceback.format_exc())


# ------------------------------------------------------------
# Helper: obtener cols del universo activo
# ------------------------------------------------------------
def cols_u(suffix):
    return {
        "GAP": f"GAP_{suffix}",
        "Peso": f"Peso_{suffix}" if suffix != "AFP" else "Peso_AFP",
        "Delta": f"Delta_GAP_{suffix}",
        "Delta_3M": f"Delta_GAP_3M_{suffix}",
        "Delta_6M": f"Delta_GAP_6M_{suffix}",
        "Z": f"GAP_Z6_{suffix}",
        "Persist": f"Persistencia_{suffix}",
        "Pos": f"Posicionamiento_{suffix}",
        "Dir": f"Direccion_{suffix}",
        "Senal": f"Senal_{suffix}",
        "Sem": f"Sem_{suffix}",
        "Score": f"FlowScore_{suffix}",
    }


# ============================================================
# TAB 0 — 💰 Flujo Agregado (NEW)
# ============================================================
with safe_tab(tabs[0], "Flujo Agregado"):
    st.subheader("💰 Flujo agregado al IPSA — AFP y FFMM")
    st.caption(
        "Vista macro: los fondos en su conjunto, ¿están aumentando o reduciendo exposición al IPSA? "
        "Agrega los flujos de todos los tickers en uno solo."
    )

    # Construir serie agregada por universo
    agg = dfh.groupby("Fecha").agg(
        SumMMUSD_AFP=("MMUSD_AFP", "sum"),
        SumMMUSD_FFMM=("MMUSD_FFMM", "sum"),
    ).reset_index().sort_values("Fecha").reset_index(drop=True)

    agg["DMMUSD_AFP"] = agg["SumMMUSD_AFP"].diff()
    agg["DMMUSD_FFMM"] = agg["SumMMUSD_FFMM"].diff()

    # Selector de vista
    vista = st.radio(
        "Vista",
        options=["Stock total (MMUSD)", "Flujo mensual (ΔMMUSD)", "Flujo acumulado desde fecha base"],
        index=1,
        horizontal=True,
        key="flujo_agg_vista",
        help=(
            "• **Stock total**: cuántos MMUSD tiene cada fondo invertido en el IPSA en cada mes.\n\n"
            "• **Flujo mensual**: cambio mes a mes del stock. Mezcla flujo real + efecto precio del IPSA.\n\n"
            "• **Flujo acumulado**: suma de flujos mensuales desde una fecha base — suaviza ruido y muestra tendencia."
        )
    )

    # ===== Gráfico 1: Serie temporal =====
    st.markdown("### Serie temporal")

    if "Stock total" in vista:
        col_afp, col_ffmm, y_title, fmt_y = "SumMMUSD_AFP", "SumMMUSD_FFMM", "Stock total (MMUSD)", ",.0f"
        plot_df = agg.copy()
    elif "Flujo mensual" in vista:
        col_afp, col_ffmm, y_title, fmt_y = "DMMUSD_AFP", "DMMUSD_FFMM", "Flujo mensual (ΔMMUSD)", ",.0f"
        plot_df = agg.dropna(subset=[col_afp, col_ffmm]).copy()
    else:
        # Acumulado desde fecha base
        fechas_disp = agg["Fecha"].tolist()
        default_idx_base = max(0, len(fechas_disp) - 25)  # ~24 meses atrás
        base_date = st.selectbox(
            "Fecha base (acumulado desde aquí)",
            options=fechas_disp,
            index=default_idx_base,
            format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
            key="flujo_agg_base"
        )
        sub = agg[agg["Fecha"] >= pd.to_datetime(base_date)].copy()
        sub["Acum_AFP"] = sub["DMMUSD_AFP"].fillna(0).cumsum()
        sub["Acum_FFMM"] = sub["DMMUSD_FFMM"].fillna(0).cumsum()
        col_afp, col_ffmm, y_title, fmt_y = "Acum_AFP", "Acum_FFMM", f"Flujo acumulado desde {pd.to_datetime(base_date).strftime('%Y-%m')} (MMUSD)", ",.0f"
        plot_df = sub

    # Construir figura
    if "Flujo mensual" in vista:
        # Barras para flujo mensual (más claro que líneas)
        fig_ts = go.Figure()
        colors_afp = ["#27ae60" if v > 0 else "#c0392b" for v in plot_df[col_afp]]
        colors_ffmm = ["#2980b9" if v > 0 else "#8e44ad" for v in plot_df[col_ffmm]]
        fig_ts.add_trace(go.Bar(
            x=plot_df["Fecha"], y=plot_df[col_afp],
            name="AFP", marker_color="#27ae60", opacity=0.85
        ))
        fig_ts.add_trace(go.Bar(
            x=plot_df["Fecha"], y=plot_df[col_ffmm],
            name="FFMM", marker_color="#3498db", opacity=0.85
        ))
        fig_ts.add_hline(y=0, line_color="black")
        fig_ts.update_layout(barmode="group")
    else:
        # Líneas para stock y acumulado
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=plot_df["Fecha"], y=plot_df[col_afp], mode="lines+markers",
            name="AFP", line=dict(color="#27ae60", width=2.5),
            marker=dict(size=5),
            fill="tozeroy" if "Acumulado" in y_title else None,
            fillcolor="rgba(39, 174, 96, 0.1)" if "Acumulado" in y_title else None
        ))
        fig_ts.add_trace(go.Scatter(
            x=plot_df["Fecha"], y=plot_df[col_ffmm], mode="lines+markers",
            name="FFMM", line=dict(color="#3498db", width=2.5),
            marker=dict(size=5),
            fill="tozeroy" if "Acumulado" in y_title else None,
            fillcolor="rgba(52, 152, 219, 0.1)" if "Acumulado" in y_title else None
        ))
        fig_ts.add_hline(y=0, line_dash="dash", line_color="gray")

    fig_ts.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=y_title,
        yaxis_tickformat=fmt_y,
        yaxis_title=y_title,
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Interpretación dinámica
    last_d_afp = agg["DMMUSD_AFP"].iloc[-1] if pd.notna(agg["DMMUSD_AFP"].iloc[-1]) else 0
    last_d_ffmm = agg["DMMUSD_FFMM"].iloc[-1] if pd.notna(agg["DMMUSD_FFMM"].iloc[-1]) else 0
    direccion_afp = "entraron" if last_d_afp > 0 else "salieron"
    direccion_ffmm = "entraron" if last_d_ffmm > 0 else "salieron"
    st.info(
        f"📌 **Último mes ({pd.to_datetime(last_date).strftime('%Y-%m')})**: "
        f"**AFP {direccion_afp} {abs(last_d_afp):,.0f} MMUSD** del IPSA. "
        f"**FFMM {direccion_ffmm} {abs(last_d_ffmm):,.0f} MMUSD**."
    )

    # ===== Tabla comparativa último mes + Δ =====
    st.markdown("### Tabla comparativa")

    last_i = len(agg) - 1

    def _val_at(col, offset):
        idx = last_i - offset
        return agg[col].iloc[idx] if 0 <= idx < len(agg) else None

    def _fmt_mm(v):
        if v is None or pd.isna(v):
            return "—"
        return f"{v:,.0f}"

    def _sum_last_n(col, n):
        """Suma de últimos n meses de delta."""
        if last_i - n + 1 < 0:
            return None
        s = agg[col].iloc[last_i - n + 1:last_i + 1].sum()
        return s

    rows = []
    for label, col_stock, col_delta in [
        ("AFP", "SumMMUSD_AFP", "DMMUSD_AFP"),
        ("FFMM", "SumMMUSD_FFMM", "DMMUSD_FFMM"),
    ]:
        rows.append({
            "Fondo": label,
            "Stock actual (MMUSD)": _fmt_mm(_val_at(col_stock, 0)),
            "Flujo 1M": _fmt_mm(_val_at(col_delta, 0)),
            "Flujo 3M (suma)": _fmt_mm(_sum_last_n(col_delta, 3)),
            "Flujo 6M (suma)": _fmt_mm(_sum_last_n(col_delta, 6)),
            "Flujo 12M (suma)": _fmt_mm(_sum_last_n(col_delta, 12)),
            "Prom mensual 12M": _fmt_mm(_sum_last_n(col_delta, 12) / 12 if _sum_last_n(col_delta, 12) is not None else None),
        })
    tabla_comp = pd.DataFrame(rows)
    st.dataframe(tabla_comp, use_container_width=True, hide_index=True)

    # ===== Gráfico 3: Barras acumuladas por año =====
    st.markdown("### Flujo acumulado por año calendario")
    st.caption("Suma del ΔMMUSD por año calendario. Útil para ver tendencias largas sin ruido mensual.")

    agg_yr = agg.copy()
    agg_yr["Año"] = agg_yr["Fecha"].dt.year
    year_sums = agg_yr.groupby("Año").agg(
        Flujo_AFP=("DMMUSD_AFP", "sum"),
        Flujo_FFMM=("DMMUSD_FFMM", "sum")
    ).reset_index()
    # Excluir años con todo NaN/0 iniciales
    year_sums = year_sums[(year_sums["Flujo_AFP"].abs() + year_sums["Flujo_FFMM"].abs()) > 0]

    fig_yr = go.Figure()
    fig_yr.add_trace(go.Bar(
        x=year_sums["Año"], y=year_sums["Flujo_AFP"],
        name="AFP", marker_color="#27ae60",
        text=[f"{v:,.0f}" for v in year_sums["Flujo_AFP"]],
        textposition="outside"
    ))
    fig_yr.add_trace(go.Bar(
        x=year_sums["Año"], y=year_sums["Flujo_FFMM"],
        name="FFMM", marker_color="#3498db",
        text=[f"{v:,.0f}" for v in year_sums["Flujo_FFMM"]],
        textposition="outside"
    ))
    fig_yr.add_hline(y=0, line_color="black")
    fig_yr.update_layout(
        template="plotly_white",
        title="Flujo anual acumulado (ΔMMUSD sumado en el año)",
        barmode="group",
        height=400,
        yaxis_title="ΔMMUSD anual",
        xaxis_title="Año",
        xaxis=dict(type="category"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_yr, use_container_width=True)

    # ===== Gráfico 4: Descomposición del flujo 6M en 3 tramos no solapados =====
    st.markdown("### 🎯 Descomposición del flujo 6M — ¿de dónde viene el movimiento?")
    st.caption(
        "Para cada papel, el flujo acumulado de 6 meses se descompone en **3 tramos no solapados que suman exactamente el total**: "
        "primeros 3 meses del semestre (meses -6 a -4), siguientes 2 meses (meses -3 a -2), y último mes (mes 0). "
        "Si los 3 tramos van en el mismo sentido y son parejos → flujo sostenido. "
        "Si tienen signos opuestos → cambio de tendencia. "
        "Solo se usa ΔGAP (rebalanceo vs IPSA) porque es la métrica limpia — el ΔMMUSD mezcla flujo y efecto precio."
    )

    top_n_anidado = st.slider("Top N tickers (por magnitud 6M)", 5, 30, 15, key="anidado_topn")

    fechas_sorted = sorted(dfh["Fecha"].unique())

    def _sum_tramo_by_ticker(suffix, start_offset, end_offset):
        """
        Suma de Delta_GAP por ticker en el tramo definido por [start_offset, end_offset]
        donde offset = 0 es el último mes, -1 el anterior, etc.
        Devuelve la suma de ΔGAP en ese tramo (sin solapamiento con otros tramos).
        """
        n = len(fechas_sorted)
        # Convertir offsets en índices absolutos
        idx_end = n - 1 + end_offset       # inclusivo
        idx_start = n - 1 + start_offset   # inclusivo
        if idx_start < 0 or idx_end < 0 or idx_start > idx_end:
            return pd.Series(dtype=float)
        fechas_tramo = fechas_sorted[idx_start:idx_end + 1]
        sub = dfh[dfh["Fecha"].isin(fechas_tramo)]
        return sub.groupby("Ticker")[f"Delta_GAP_{suffix}"].sum()

    def _render_descomposicion(suffix, color_tramo1, color_tramo2, color_tramo3):
        # Tramo 1: meses -5 a -3 (los primeros 3 meses del semestre)
        # Tramo 2: meses -2 a -1 (los 2 meses anteriores al último)
        # Tramo 3: mes 0 (último mes)
        # Total = Tramo1 + Tramo2 + Tramo3 = Flujo_6M completo
        t1 = _sum_tramo_by_ticker(suffix, -5, -3)
        t2 = _sum_tramo_by_ticker(suffix, -2, -1)
        t3 = _sum_tramo_by_ticker(suffix, 0, 0)

        if len(t1) == 0 or len(t2) == 0 or len(t3) == 0:
            st.warning(f"No hay suficiente historia para {suffix} (se necesitan ≥ 6 meses)")
            return

        comp = pd.DataFrame({
            "Tramo1_mes_minus_6_a_4": t1,
            "Tramo2_mes_minus_3_a_2": t2,
            "Tramo3_ultimo_mes": t3,
        }).fillna(0).reset_index()
        comp["Total_6M"] = comp["Tramo1_mes_minus_6_a_4"] + comp["Tramo2_mes_minus_3_a_2"] + comp["Tramo3_ultimo_mes"]
        comp = comp[comp["Total_6M"].abs() > 0]
        comp["abs_total"] = comp["Total_6M"].abs()
        comp = comp.nlargest(top_n_anidado, "abs_total").sort_values("Total_6M", ascending=True)

        if len(comp) == 0:
            st.warning(f"Sin movimientos significativos en {suffix}")
            return

        # Convertir a bps
        mult = 10000.0

        # Stacked bar: para cada ticker, 3 segmentos apilados
        # Plotly maneja bien stacking con valores negativos por separado
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(
            y=comp["Ticker"], x=comp["Tramo1_mes_minus_6_a_4"] * mult,
            name="Meses -6 a -4 (primeros 3M)",
            marker_color=color_tramo1,
            orientation="h",
            hovertemplate="<b>%{y}</b><br>Tramo 1 (meses -6 a -4): %{x:.0f} bps<extra></extra>"
        ))
        fig_d.add_trace(go.Bar(
            y=comp["Ticker"], x=comp["Tramo2_mes_minus_3_a_2"] * mult,
            name="Meses -3 a -2 (mid)",
            marker_color=color_tramo2,
            orientation="h",
            hovertemplate="<b>%{y}</b><br>Tramo 2 (meses -3 a -2): %{x:.0f} bps<extra></extra>"
        ))
        fig_d.add_trace(go.Bar(
            y=comp["Ticker"], x=comp["Tramo3_ultimo_mes"] * mult,
            name="Último mes",
            marker_color=color_tramo3,
            orientation="h",
            hovertemplate="<b>%{y}</b><br>Tramo 3 (último mes): %{x:.0f} bps<extra></extra>"
        ))
        fig_d.add_vline(x=0, line_color="black", line_width=0.8)
        fig_d.update_layout(
            template="plotly_white",
            title=f"{suffix} — Descomposición del flujo 6M en 3 tramos (los 3 suman al total 6M)",
            barmode="relative",   # stack pero respeta signos
            height=max(450, 28 * len(comp)),
            xaxis_title=f"Flujo {suffix} (bps de ΔGAP)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_d, use_container_width=True)

        # Tabla resumen interpretativa
        comp_tbl = comp[["Ticker", "Tramo1_mes_minus_6_a_4", "Tramo2_mes_minus_3_a_2",
                         "Tramo3_ultimo_mes", "Total_6M"]].copy()
        comp_tbl.columns = ["Ticker", "Tramo 1 (M-6 a M-4)", "Tramo 2 (M-3 a M-2)", "Último mes", "Total 6M"]
        # Convertir a bps
        for c in comp_tbl.columns[1:]:
            comp_tbl[c] = (comp_tbl[c] * 10000).round(0).astype(int)
        # Diagnóstico de tendencia
        def _diagnostico(r):
            t1_pos = r["Tramo 1 (M-6 a M-4)"] > 0
            t2_pos = r["Tramo 2 (M-3 a M-2)"] > 0
            t3_pos = r["Último mes"] > 0
            if t1_pos and t2_pos and t3_pos:
                return "🟢 Compra sostenida"
            if not t1_pos and not t2_pos and not t3_pos:
                return "🔴 Venta sostenida"
            if t1_pos and not t3_pos:
                return "⚠️ Era compra, ahora vende"
            if not t1_pos and t3_pos:
                return "↗️ Era venta, ahora compra"
            return "↔️ Mixto"
        comp_tbl["Tendencia"] = comp_tbl.apply(_diagnostico, axis=1)
        st.dataframe(comp_tbl, use_container_width=True, hide_index=True)

    # Colores para los 3 tramos (de claro a oscuro = del pasado al presente)
    if universo == "AFP":
        _render_descomposicion("AFP", "#a5c4e6", "#5b8fd1", "#1f4e8f")
    elif universo == "FFMM":
        _render_descomposicion("FFMM", "#aed6f1", "#5dade2", "#1b4f72")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### AFP")
            _render_descomposicion("AFP", "#a5c4e6", "#5b8fd1", "#1f4e8f")
        with col2:
            st.markdown("#### FFMM")
            _render_descomposicion("FFMM", "#aed6f1", "#5dade2", "#1b4f72")

    st.info(
        "📌 **Cómo leer**: las 3 barras de cada papel **suman exactamente el flujo total de 6 meses**. "
        "Si los 3 tramos van en la misma dirección y de tamaño similar → flujo sostenido y parejo. "
        "Si el último mes (color más oscuro) tiene signo opuesto a los primeros tramos → cambio de tendencia. "
        "La columna **Tendencia** de la tabla resume cada caso."
    )

    with st.expander("📘 Cómo leer esta tab"):
        st.markdown("""
**¿Qué responde esta vista?**

La pregunta macro: **¿los fondos en su conjunto están entrando o saliendo del IPSA?**

**Tres vistas:**

1. **Stock total (MMUSD)**: cuántos MMUSD tiene invertido cada fondo en el IPSA en cada mes. Muestra el nivel absoluto.

2. **Flujo mensual (ΔMMUSD)**: cambio de un mes al siguiente. Barras verdes = entraron dólares, rojas = salieron. Es la métrica más "de mesa" para ver dinámica.

3. **Flujo acumulado desde fecha base**: suma de flujos mensuales desde una fecha que vos elegís. Suaviza el ruido y muestra si la tendencia larga es entrada o salida neta.

**Advertencia sobre el ΔMMUSD**: mezcla flujo real (compras/ventas) con efecto precio (si el IPSA sube 5%, el MMUSD sube aunque no compren nada). Para separar flujo puro del efecto precio habría que tener retornos del IPSA — no están en el Excel.

**Las dos líneas AFP vs FFMM superpuestas** permiten detectar si ambos fondos actúan igual (consenso) o divergen.

**Ejemplo de lectura**:
- Si AFP tiene +500 MMUSD este mes y FFMM tiene +200 → consenso de entrada al IPSA.
- Si AFP tiene -800 MMUSD y FFMM tiene +100 → AFP rotando afuera, FFMM comprando (desacuerdo institucional).
""")


# ============================================================
# TAB 1 — Posicionamiento vs historia
# ============================================================
with safe_tab(tabs[1], "Posicionamiento vs historia"):
    st.subheader("GAP por papel vs promedio histórico")

    # Selector de fecha
    available_dates = sorted(dfh["Fecha"].dropna().unique())
    default_idx = len(available_dates) - 1
    if last_date in available_dates:
        default_idx = available_dates.index(last_date)

    sel_date = st.selectbox(
        "Fecha a visualizar",
        options=available_dates,
        index=max(0, default_idx),
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"),
        key="tab1_date"
    )

    def _render_pos_hist(suffix):
        c = cols_u(suffix)

        # Snap para la fecha seleccionada
        snap_date = dfh[dfh["Fecha"] == pd.to_datetime(sel_date)][
            ["Ticker", c["GAP"], c["Z"]]
        ].dropna(subset=[c["GAP"]])
        snap_date = snap_date.rename(columns={c["GAP"]: "GAP_Actual"})

        # Promedio histórico (en el rango seleccionado, no toda la historia)
        hist_avg = dfh.groupby("Ticker", as_index=False)[c["GAP"]].mean().rename(
            columns={c["GAP"]: "GAP_Prom"}
        )

        comp = snap_date.merge(hist_avg, on="Ticker", how="left")
        comp["Desvio"] = comp["GAP_Actual"] - comp["GAP_Prom"]
        comp = comp.sort_values("GAP_Actual", ascending=False)

        if len(comp) == 0:
            st.warning(f"No hay datos {suffix} para {sel_date}")
            return

        # Barras agrupadas: GAP_Actual (oscuro) vs GAP_Prom (claro)
        fecha_str = pd.to_datetime(sel_date).strftime("%Y-%m-%d")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp["Ticker"],
            y=comp["GAP_Actual"],
            name=f"GAP {fecha_str}",
            marker_color="#1f4e8f",  # azul oscuro
            hovertemplate="<b>%{x}</b><br>GAP actual: %{y:.2%}<extra></extra>",
            customdata=comp["Desvio"]
        ))
        fig.add_trace(go.Bar(
            x=comp["Ticker"],
            y=comp["GAP_Prom"],
            name="Promedio histórico",
            marker_color="#8fb4d8",  # azul claro
            hovertemplate="<b>%{x}</b><br>GAP promedio: %{y:.2%}<extra></extra>"
        ))
        fig.add_hline(y=0, line_width=1, line_color="black")
        fig.update_layout(
            template="plotly_white",
            title=f"{suffix}: GAP por papel vs promedio histórico — {fecha_str}",
            barmode="group",
            height=500,
            yaxis_tickformat=".1%",
            yaxis_title="GAP",
            xaxis_title="",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(b=120)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Z-score barras (se conserva - sigue siendo útil para detectar extremos)
        zs = comp[["Ticker", c["Z"]]].dropna().sort_values(c["Z"], ascending=True)
        if len(zs):
            colors = ["#e74c3c" if v < -1.5 else "#f5b7b1" if v < 0
                      else "#a5e8c6" if v < 1.5 else "#27ae60" for v in zs[c["Z"]]]
            fig2 = go.Figure(go.Bar(x=zs[c["Z"]], y=zs["Ticker"],
                                    orientation="h", marker_color=colors))
            fig2.add_vline(x=1.5, line_dash="dash", line_color="gray")
            fig2.add_vline(x=-1.5, line_dash="dash", line_color="gray")
            fig2.add_vline(x=0, line_color="black")
            fig2.update_layout(
                template="plotly_white",
                title=f"{suffix}: Z-score del GAP actual vs historia propia del ticker",
                height=max(400, 18 * len(zs)),
                xaxis_title="Z-score (±1.5 = extremo)"
            )
            st.plotly_chart(fig2, use_container_width=True)

    if universo == "AFP":
        _render_pos_hist("AFP")
    elif universo == "FFMM":
        _render_pos_hist("FFMM")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### AFP")
            _render_pos_hist("AFP")
        with col2:
            st.markdown("### FFMM")
            _render_pos_hist("FFMM")

    # Líneas superpuestas si hay tickers seleccionados
    if sel_tickers:
        st.markdown("#### Serie histórica superpuesta")
        sub = dfh[dfh["Ticker"].isin(sel_tickers)]
        fig_l = go.Figure()
        gap_cols_plot = ["GAP_AFP"] if universo == "AFP" else \
                        ["GAP_FFMM"] if universo == "FFMM" else \
                        ["GAP_AFP", "GAP_FFMM"]
        for t in sel_tickers:
            subt = sub[sub["Ticker"] == t].sort_values("Fecha")
            for gc in gap_cols_plot:
                fig_l.add_trace(go.Scatter(x=subt["Fecha"], y=subt[gc], mode="lines",
                                           name=f"{t} {gc.split('_')[1]}"))
        fig_l.add_hline(y=0, line_width=1)
        fig_l.update_layout(template="plotly_white", yaxis_tickformat=".1%", height=400)
        st.plotly_chart(fig_l, use_container_width=True)


# ============================================================
# TAB 2 — Snapshot
# ============================================================
with safe_tab(tabs[2], "Snapshot"):
    st.subheader(f"Snapshot — {last_date.date()}")

    def _snap_table(suffix):
        c = cols_u(suffix)
        peso_col = f"Peso_{suffix}"
        d = snap_last.copy()
        # Sparkline (filtrar NaN porque LineChartColumn no los acepta)
        spark = {}
        for t in d["Ticker"].unique():
            hist = dfh[(dfh["Ticker"] == t)].sort_values("Fecha").tail(12)[c["GAP"]].dropna().tolist()
            spark[t] = [float(x) for x in hist] if hist else [0.0]
        d["Sparkline"] = d["Ticker"].map(spark)

        # Persistencia formateada
        d["Persist_str"] = d[c["Persist"]].apply(
            lambda v: f"{int(v):+d}M" if pd.notna(v) and v != 0 else "0M"
        )

        d["rank"] = d[c["Senal"]].map(SIGNAL_RANK).fillna(99)
        d["abs_d"] = d[c["Delta"]].abs()
        d = d.sort_values(["rank", "abs_d"], ascending=[True, False])

        show = d[[c["Sem"], "Ticker", "Sector", peso_col, "Peso_IPSA",
                  c["GAP"], c["Delta"], c["Delta_3M"],
                  c["Pos"], c["Dir"], "Persist_str", "Sparkline", c["Senal"]]].copy()
        show.columns = ["Sem", "Ticker", "Sector", f"Peso_{suffix}", "Peso_IPSA",
                        "GAP", "ΔGAP", "ΔGAP 3M",
                        "Posicion.", "Dirección", "Persist.", "Spark 12M", "Señal"]

        st.dataframe(
            show,
            use_container_width=True,
            height=600,
            hide_index=True,
            column_config={
                f"Peso_{suffix}": st.column_config.NumberColumn(format="%.2f%%"),
                "Peso_IPSA": st.column_config.NumberColumn(format="%.2f%%"),
                "GAP": st.column_config.NumberColumn(format="%.2f%%"),
                "ΔGAP": st.column_config.NumberColumn(format="%.2f%%"),
                "ΔGAP 3M": st.column_config.NumberColumn(format="%.2f%%"),
                "Spark 12M": st.column_config.LineChartColumn("GAP 12M"),
            }
        )

    if universo == "AFP":
        _snap_table("AFP")
    elif universo == "FFMM":
        _snap_table("FFMM")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### AFP")
            _snap_table("AFP")
        with col2:
            st.markdown("#### FFMM")
            _snap_table("FFMM")

    with st.expander("📘 Matriz de Señal (auditable)"):
        st.markdown("""
| Posicionamiento | Dirección | Señal |
|---|---|---|
| UW_Extremo | Acumulando_Fuerte | **BUY_FUERTE** |
| UW | Acumulando_Fuerte | BUY |
| UW_Extremo | Acumulando | BUY |
| UW | Acumulando | BUY_LIGHT |
| Neutral | Acumulando_Fuerte | BUY_LIGHT |
| OW | Acumulando_Fuerte | BUY_LIGHT |
| OW / OW_Ext | Acumulando | HOLD |
| OW_Extremo | Reduciendo_Fuerte | SELL |
| OW | Reduciendo_Fuerte | SELL |
| OW_Extremo | Reduciendo | SELL_LIGHT |
| Neutral | Reduciendo_Fuerte | SELL_LIGHT |
| UW / UW_Ext | Reduciendo_Fuerte | **SELL_FUERTE** |
| Resto / Plano | | HOLD |

**Umbral de "Fuerte"**: `|ΔGAP| > max(5 bps, 0.5·σ_ticker)` **y** `|ΔGAP| ≥ percentil 85 histórico del propio ticker`.
**Persistencia**: # meses consecutivos con mismo signo de ΔGAP (requiere ≥3 para `_Fuerte`).
""")


# ============================================================
# TAB 3 — Ranking
# ============================================================
with safe_tab(tabs[3], "Ranking"):
    st.subheader(f"Ranking — {last_date.date()}")

    def _render_cards(d, suffix, kind):
        c = cols_u(suffix)
        for _, r in d.iterrows():
            sig = r[c["Senal"]]
            col_bg = SIGNAL_COLOR.get(sig, "#ccc")
            st.markdown(
                f"""
<div style="border-left: 6px solid {col_bg}; padding: 6px 10px; margin-bottom: 4px; background: #f8f9fa; border-radius: 4px;">
<b>{r[c['Sem']]} {r['Ticker']}</b> — <code>{sig}</code><br>
<small>GAP: <b>{fmt_pct(r[c['GAP']])}</b> | ΔGAP: <b>{fmt_bps(r[c['Delta']])} bps</b> | Persist: <b>{int(r[c['Persist']]):+d}M</b></small>
</div>
""",
                unsafe_allow_html=True
            )

    def _rankings(suffix):
        c = cols_u(suffix)
        # Separar BUY y SELL ordenado por Delta
        buys = snap_last[snap_last[c["Senal"]].isin(["BUY_FUERTE", "BUY", "BUY_LIGHT"])] \
            .nlargest(5, c["Delta"])
        sells = snap_last[snap_last[c["Senal"]].isin(["SELL_FUERTE", "SELL", "SELL_LIGHT"])] \
            .nsmallest(5, c["Delta"])

        if len(buys) == 0:
            buys = snap_last.nlargest(5, c["Delta"])
        if len(sells) == 0:
            sells = snap_last.nsmallest(5, c["Delta"])

        cl, cr = st.columns(2)
        with cl:
            st.markdown(f"**🟢 Top 5 BUY {suffix}**")
            _render_cards(buys, suffix, "BUY")
        with cr:
            st.markdown(f"**🔴 Top 5 SELL {suffix}**")
            _render_cards(sells, suffix, "SELL")

        # Tabla top 20
        st.markdown(f"**Top 20 movimientos {suffix}**")
        d20 = snap_last.copy()
        d20["abs_d"] = d20[c["Delta"]].abs()
        d20 = d20.nlargest(20, "abs_d")
        show = d20[[c["Sem"], "Ticker", c["GAP"], c["Delta"], c["Pos"], c["Dir"], c["Persist"], c["Senal"]]]
        show.columns = ["Sem", "Ticker", "GAP", "ΔGAP", "Pos.", "Dir.", "Persist.", "Señal"]
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={
                         "GAP": st.column_config.NumberColumn(format="%.2f%%"),
                         "ΔGAP": st.column_config.NumberColumn(format="%.2f%%"),
                     })

    if universo == "AFP":
        _rankings("AFP")
    elif universo == "FFMM":
        _rankings("FFMM")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### AFP")
            _rankings("AFP")
        with col2:
            st.markdown("### FFMM")
            _rankings("FFMM")

        st.markdown("---")
        st.markdown("### 🔀 Top 10 divergencias AFP vs FFMM")
        div = snap_last.copy()
        div["abs_div"] = div["Divergencia_GAP"].abs()
        div_top = div.nlargest(10, "abs_div")
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(y=div_top["Ticker"], x=div_top["GAP_AFP"], name="GAP AFP",
                                 orientation="h", marker_color="#27ae60"))
        fig_div.add_trace(go.Bar(y=div_top["Ticker"], x=div_top["GAP_FFMM"], name="GAP FFMM",
                                 orientation="h", marker_color="#3498db"))
        fig_div.update_layout(template="plotly_white", barmode="group", height=400,
                              xaxis_tickformat=".1%")
        st.plotly_chart(fig_div, use_container_width=True)


# ============================================================
# TAB 4 — Detalle por papel (siempre muestra ambos)
# ============================================================
with safe_tab(tabs[4], "Detalle por papel"):
    st.subheader("Detalle por papel — AFP y FFMM en paralelo")
    paper = st.selectbox("Ticker", sorted(dfh["Ticker"].unique()))
    sub = dfh[dfh["Ticker"] == paper].sort_values("Fecha")
    last_row = df[(df["Ticker"] == paper) & (df["Fecha"] == last_date)]

    if len(last_row):
        lr = last_row.iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
**{lr['Sem_AFP']} AFP**
- Posicionamiento: `{lr['Posicionamiento_AFP']}`
- Dirección: `{lr['Direccion_AFP']}`
- Persist: `{int(lr['Persistencia_AFP']):+d}M`
- Señal: **{lr['Senal_AFP']}**
- GAP: `{fmt_pct(lr['GAP_AFP'])}`
""")
        with c2:
            st.markdown(f"""
**{lr['Sem_FFMM']} FFMM**
- Posicionamiento: `{lr['Posicionamiento_FFMM']}`
- Dirección: `{lr['Direccion_FFMM']}`
- Persist: `{int(lr['Persistencia_FFMM']):+d}M`
- Señal: **{lr['Senal_FFMM']}**
- GAP: `{fmt_pct(lr['GAP_FFMM'])}`
""")

    # Series
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Fecha"], y=sub["GAP_AFP"], mode="lines", name="GAP AFP", line=dict(color="#27ae60")))
    fig.add_trace(go.Scatter(x=sub["Fecha"], y=sub["GAP_FFMM"], mode="lines", name="GAP FFMM", line=dict(color="#3498db")))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(template="plotly_white", title=f"{paper} — GAP histórico",
                      yaxis_tickformat=".1%", hovermode="x unified", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Delta bars
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=sub["Fecha"], y=sub["Delta_GAP_AFP"], name="ΔGAP AFP", marker_color="#27ae60"))
    fig2.add_trace(go.Bar(x=sub["Fecha"], y=sub["Delta_GAP_FFMM"], name="ΔGAP FFMM", marker_color="#3498db"))
    fig2.update_layout(template="plotly_white", title=f"{paper} — ΔGAP mensual",
                       yaxis_tickformat=".2%", barmode="group", height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Events
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Eventos AFP**")
        ea = events_afp[events_afp["Ticker"] == paper].sort_values("Fecha", ascending=False)
        st.dataframe(ea[["Fecha", "Posicionamiento", "Direccion", "Senal", "Persistencia"]],
                     use_container_width=True, hide_index=True, height=300)
    with c2:
        st.markdown("**Eventos FFMM**")
        ef = events_ffmm[events_ffmm["Ticker"] == paper].sort_values("Fecha", ascending=False)
        st.dataframe(ef[["Fecha", "Posicionamiento", "Direccion", "Senal", "Persistencia"]],
                     use_container_width=True, hide_index=True, height=300)

    # Lead-lag del papel
    st.markdown("**Correlación cruzada ΔGAP_AFP vs ΔGAP_FFMM (últimos 24m)**")
    a = sub["Delta_GAP_AFP"].dropna().tail(24).values
    b = sub["Delta_GAP_FFMM"].dropna().tail(24).values
    n = min(len(a), len(b))
    if n >= 8:
        a, b = a[-n:], b[-n:]
        lags, corrs = [], []
        for lag in range(-6, 7):
            if lag < 0:
                x, y = a[:lag], b[-lag:]
            elif lag > 0:
                x, y = a[lag:], b[:-lag]
            else:
                x, y = a, b
            if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
                lags.append(lag); corrs.append(0)
                continue
            lags.append(lag); corrs.append(np.corrcoef(x, y)[0, 1])
        fig_ll = go.Figure(go.Bar(x=lags, y=corrs, marker_color=["#e74c3c" if c < 0 else "#27ae60" for c in corrs]))
        fig_ll.update_layout(template="plotly_white", height=300,
                             title="Lag negativo: AFP lidera. Lag positivo: FFMM lidera.",
                             xaxis_title="Lag (meses)", yaxis_title="Correlación")
        st.plotly_chart(fig_ll, use_container_width=True)


# ============================================================
# TAB 5 — Heatmap
# ============================================================
with safe_tab(tabs[5], "Heatmap"):
    st.subheader("Heatmap — últimos 24 meses")
    metric_choice = st.selectbox("Métrica", ["GAP", "ΔGAP", "GAP_Z6"])

    def _metric_col(suffix, choice):
        return {"GAP": f"GAP_{suffix}", "ΔGAP": f"Delta_GAP_{suffix}", "GAP_Z6": f"GAP_Z6_{suffix}"}[choice]

    def _render_heatmap(suffix):
        mcol = _metric_col(suffix, metric_choice)
        last24 = dfh[dfh["Fecha"] >= last_date - pd.DateOffset(months=24)]
        piv = last24.pivot_table(index="Ticker", columns="Fecha", values=mcol, aggfunc="last")
        # Orden por valor último mes
        last_vals = piv.iloc[:, -1].sort_values(ascending=False)
        piv = piv.reindex(last_vals.index)
        fig = px.imshow(piv, aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        title=f"{suffix}: {metric_choice}")
        fig.update_layout(template="plotly_white", height=max(400, 12 * len(piv)))
        st.plotly_chart(fig, use_container_width=True)

    if universo == "AFP":
        _render_heatmap("AFP")
    elif universo == "FFMM":
        _render_heatmap("FFMM")
    else:
        # Heatmap de divergencia
        last24 = dfh[dfh["Fecha"] >= last_date - pd.DateOffset(months=24)].copy()
        piv = last24.pivot_table(index="Ticker", columns="Fecha", values="Divergencia_GAP", aggfunc="last")
        last_vals = piv.iloc[:, -1].sort_values(ascending=False)
        piv = piv.reindex(last_vals.index)
        fig = px.imshow(piv, aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        title="Divergencia: GAP_AFP − GAP_FFMM (rojo=FFMM más OW, verde=AFP más OW)")
        fig.update_layout(template="plotly_white", height=max(400, 12 * len(piv)))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 6 — Flujo mensual / 3M / 6M
# ============================================================
with safe_tab(tabs[6], "Flujo"):
    st.subheader("Flujos — mensual / acumulado")
    ventana = st.radio("Ventana", ["1M", "3M", "6M"], horizontal=True)
    top_n = st.slider("Top N", 5, 30, 12)
    fechas = sorted(dfh["Fecha"].unique())
    sel_f = st.selectbox("Mes", fechas, index=len(fechas) - 1,
                         format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m"))

    def _render_flow(suffix):
        col_map = {"1M": f"Delta_GAP_{suffix}", "3M": f"Delta_GAP_3M_{suffix}", "6M": f"Delta_GAP_6M_{suffix}"}
        col = col_map[ventana]
        m = dfh[dfh["Fecha"] == sel_f].dropna(subset=[col])

        c1, c2 = st.columns(2)
        with c1:
            buys = m.nlargest(top_n, col).sort_values(col)
            fig = px.bar(buys, x=col, y="Ticker", orientation="h",
                         title=f"Top {top_n} compras {suffix} ({ventana})",
                         color_discrete_sequence=["#27ae60"])
            fig.update_layout(template="plotly_white", xaxis_tickformat=".2%", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sells = m.nsmallest(top_n, col).sort_values(col, ascending=False)
            fig = px.bar(sells, x=col, y="Ticker", orientation="h",
                         title=f"Top {top_n} ventas {suffix} ({ventana})",
                         color_discrete_sequence=["#e74c3c"])
            fig.update_layout(template="plotly_white", xaxis_tickformat=".2%", height=420)
            st.plotly_chart(fig, use_container_width=True)

    if universo in ("AFP", "Ambos"):
        st.markdown("### AFP" if universo == "Ambos" else "")
        _render_flow("AFP")
    if universo in ("FFMM", "Ambos"):
        st.markdown("### FFMM" if universo == "Ambos" else "")
        _render_flow("FFMM")


# ============================================================
# TAB 7 — Breadth (siempre ambos)
# ============================================================
with safe_tab(tabs[7], "Breadth"):
    st.subheader("Breadth AFP y FFMM")
    br = dfh.groupby("Fecha").agg(
        pct_buy_afp=("Delta_GAP_AFP", lambda s: (s > 0).mean()),
        pct_ow_afp=("GAP_AFP", lambda s: (s > 0).mean()),
        pct_buy_ffmm=("Delta_GAP_FFMM", lambda s: (s > 0).mean()),
        pct_ow_ffmm=("GAP_FFMM", lambda s: (s > 0).mean()),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=br["Fecha"], y=br["pct_buy_afp"], mode="lines",
                             name="% comprando AFP", line=dict(color="#27ae60")))
    fig.add_trace(go.Scatter(x=br["Fecha"], y=br["pct_buy_ffmm"], mode="lines",
                             name="% comprando FFMM", line=dict(color="#3498db")))
    fig.update_layout(template="plotly_white", title="% papeles con ΔGAP > 0",
                      yaxis_tickformat=".0%", height=350)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=br["Fecha"], y=br["pct_ow_afp"], mode="lines",
                              name="% OW AFP", line=dict(color="#27ae60")))
    fig2.add_trace(go.Scatter(x=br["Fecha"], y=br["pct_ow_ffmm"], mode="lines",
                              name="% OW FFMM", line=dict(color="#3498db")))
    fig2.update_layout(template="plotly_white", title="% papeles con GAP > 0",
                       yaxis_tickformat=".0%", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    last_br = br.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("% compra AFP", f"{last_br['pct_buy_afp']:.0%}")
    c2.metric("% OW AFP", f"{last_br['pct_ow_afp']:.0%}")
    c3.metric("% compra FFMM", f"{last_br['pct_buy_ffmm']:.0%}")
    c4.metric("% OW FFMM", f"{last_br['pct_ow_ffmm']:.0%}")


# ============================================================
# TAB 8 — Scatter AFP vs FFMM (siempre ambos)
# ============================================================
with safe_tab(tabs[8], "Scatter AFP vs FFMM"):
    st.subheader("🎯 Scatter AFP vs FFMM")
    snap = snap_last.copy()
    snap = snap.dropna(subset=["GAP_AFP", "GAP_FFMM"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snap["GAP_AFP"], y=snap["GAP_FFMM"], mode="markers+text",
        text=snap["Ticker"], textposition="top center",
        marker=dict(
            size=(snap["Peso_IPSA"] * 500).clip(5, 50),
            color=snap["Delta_GAP_AFP"],
            colorscale="RdYlGn", cmid=0,
            colorbar=dict(title="ΔGAP AFP"),
            line=dict(width=1, color="#333")
        ),
        hovertemplate="<b>%{text}</b><br>GAP_AFP: %{x:.2%}<br>GAP_FFMM: %{y:.2%}<extra></extra>"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    # Cuadrant labels
    fig.add_annotation(x=snap["GAP_AFP"].max() * 0.7, y=snap["GAP_FFMM"].max() * 0.7,
                       text="Consenso OW", showarrow=False, font=dict(color="#27ae60", size=12))
    fig.add_annotation(x=snap["GAP_AFP"].min() * 0.7, y=snap["GAP_FFMM"].min() * 0.7,
                       text="Consenso UW", showarrow=False, font=dict(color="#e74c3c", size=12))
    fig.add_annotation(x=snap["GAP_AFP"].max() * 0.7, y=snap["GAP_FFMM"].min() * 0.7,
                       text="Solo AFP OW", showarrow=False, font=dict(color="#f39c12", size=12))
    fig.add_annotation(x=snap["GAP_AFP"].min() * 0.7, y=snap["GAP_FFMM"].max() * 0.7,
                       text="Solo FFMM OW", showarrow=False, font=dict(color="#3498db", size=12))

    fig.update_layout(template="plotly_white",
                      xaxis_title="GAP AFP", yaxis_title="GAP FFMM",
                      xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top 10 divergencias AFP vs FFMM**")
    snap["abs_div"] = snap["Divergencia_GAP"].abs()
    div_top = snap.nlargest(10, "abs_div")[["Ticker", "GAP_AFP", "GAP_FFMM", "Divergencia_GAP",
                                             "Senal_AFP", "Senal_FFMM"]]
    st.dataframe(div_top, use_container_width=True, hide_index=True,
                 column_config={
                     "GAP_AFP": st.column_config.NumberColumn(format="%.2f%%"),
                     "GAP_FFMM": st.column_config.NumberColumn(format="%.2f%%"),
                     "Divergencia_GAP": st.column_config.NumberColumn(format="%.2f%%"),
                 })


# ============================================================
# TAB 9 — Liderazgo (siempre ambos)
# ============================================================
with safe_tab(tabs[9], "Liderazgo"):
    st.subheader("⚡ Liderazgo AFP vs FFMM")

    lid_tbl = snap_last[["Ticker", "Sector", "Corr_6M", "Lead_Lag", "Liderazgo_del_mes",
                         "Divergencia_GAP", "Divergencia_Z6"]].copy()
    lid_tbl = lid_tbl.sort_values("Lead_Lag", na_position="last")
    st.dataframe(lid_tbl, use_container_width=True, hide_index=True,
                 column_config={
                     "Corr_6M": st.column_config.NumberColumn(format="%.2f"),
                     "Lead_Lag": st.column_config.NumberColumn(format="%.0f"),
                     "Divergencia_GAP": st.column_config.NumberColumn(format="%.2f%%"),
                     "Divergencia_Z6": st.column_config.NumberColumn(format="%.2f"),
                 })

    # Barras de Lead_Lag
    ll = lid_tbl.dropna(subset=["Lead_Lag"]).sort_values("Lead_Lag")
    if len(ll):
        colors = ["#27ae60" if v < 0 else "#3498db" if v > 0 else "#95a5a6" for v in ll["Lead_Lag"]]
        fig = go.Figure(go.Bar(y=ll["Ticker"], x=ll["Lead_Lag"], orientation="h", marker_color=colors))
        fig.add_vline(x=0, line_color="black")
        fig.update_layout(template="plotly_white", height=max(400, 18 * len(ll)),
                          title="Lead-Lag (negativo = AFP lidera / positivo = FFMM lidera)",
                          xaxis_title="Lag óptimo (meses)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Conteo de Liderazgo del mes:**")
    lid_counts = snap_last["Liderazgo_del_mes"].value_counts().reset_index()
    lid_counts.columns = ["Tipo", "Cantidad"]
    st.dataframe(lid_counts, hide_index=True, use_container_width=True)


# ============================================================
# TAB 10 — Sectorial
# ============================================================
with safe_tab(tabs[10], "Sectorial"):
    st.subheader("🏢 Vista sectorial")

    def _render_sectorial(suffix):
        gap_col = f"GAP_{suffix}"
        last24 = dfh[dfh["Fecha"] >= last_date - pd.DateOffset(months=24)]
        sect = last24.groupby(["Sector", "Fecha"])[gap_col].sum().reset_index()
        piv = sect.pivot(index="Sector", columns="Fecha", values=gap_col)
        last_vals = piv.iloc[:, -1].sort_values(ascending=False)
        piv = piv.reindex(last_vals.index)

        fig = px.imshow(piv, aspect="auto", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        title=f"{suffix}: OW/UW sectorial (24M)")
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Waterfall: cambio de OW sectorial del último mes
        prev_date = sorted(dfh["Fecha"].unique())[-2] if len(dfh["Fecha"].unique()) >= 2 else None
        if prev_date is not None:
            curr = dfh[dfh["Fecha"] == last_date].groupby("Sector")[gap_col].sum()
            prev = dfh[dfh["Fecha"] == prev_date].groupby("Sector")[gap_col].sum()
            delta = (curr - prev).sort_values()

            fig2 = go.Figure(go.Bar(
                x=delta.values, y=delta.index, orientation="h",
                marker_color=["#e74c3c" if v < 0 else "#27ae60" for v in delta.values]
            ))
            fig2.add_vline(x=0, line_color="black")
            fig2.update_layout(template="plotly_white",
                               title=f"{suffix}: ΔOW sectorial último mes",
                               xaxis_tickformat=".2%", height=450)
            st.plotly_chart(fig2, use_container_width=True)

    if universo == "AFP":
        _render_sectorial("AFP")
    elif universo == "FFMM":
        _render_sectorial("FFMM")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### AFP")
            _render_sectorial("AFP")
        with c2:
            st.markdown("### FFMM")
            _render_sectorial("FFMM")


# ============================================================
# TAB 11 — Persistencia
# ============================================================
with safe_tab(tabs[11], "Persistencia"):
    st.subheader("🔄 Persistencia de flujo")

    def _render_persist(suffix):
        c = cols_u(suffix)
        d = snap_last.copy()
        d = d[d[c["Persist"]].abs() >= 3].sort_values(c["Persist"])
        if len(d) == 0:
            st.info(f"No hay tickers con persistencia ≥ 3M en {suffix}.")
            return
        colors = ["#27ae60" if v > 0 else "#e74c3c" for v in d[c["Persist"]]]
        fig = go.Figure(go.Bar(y=d["Ticker"], x=d[c["Persist"]], orientation="h", marker_color=colors))
        fig.add_vline(x=0, line_color="black")
        fig.update_layout(template="plotly_white",
                          title=f"{suffix}: meses consecutivos con mismo signo de ΔGAP",
                          xaxis_title="Persistencia (signo + magnitud)",
                          height=max(400, 22 * len(d)))
        st.plotly_chart(fig, use_container_width=True)

    if universo == "AFP":
        _render_persist("AFP")
    elif universo == "FFMM":
        _render_persist("FFMM")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### AFP")
            _render_persist("AFP")
        with c2:
            st.markdown("### FFMM")
            _render_persist("FFMM")
