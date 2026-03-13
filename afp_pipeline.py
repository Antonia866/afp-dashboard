import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

SHEET_HOLA = "Hola Valores"
SHEET_IPSA = "IPSA"


# =========================================================
# Excel utils
# =========================================================
def _excel_sheet_names(xls_source) -> list[str]:
    xf = pd.ExcelFile(xls_source, engine="openpyxl")
    return list(xf.sheet_names)


def _pick_sheet(available: list[str], want: str, aliases: list[str]) -> Optional[str]:
    if want in available:
        return want

    low_map = {s.lower(): s for s in available}
    if want.lower() in low_map:
        return low_map[want.lower()]

    for a in aliases:
        if a in available:
            return a
        if a.lower() in low_map:
            return low_map[a.lower()]

    for s in available:
        sl = s.lower()
        if want.lower() in sl:
            return s
        for a in aliases:
            if a.lower() in sl:
                return s
    return None


def _read_excel(xls_source, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(xls_source, sheet_name=sheet_name, engine="openpyxl")


# =========================================================
# Fecha override desde Hola Valores!I2
# =========================================================
def _find_last_date_override_from_I2(xls_source, sheet_name: str) -> Optional[pd.Timestamp]:
    try:
        raw = pd.read_excel(xls_source, sheet_name=sheet_name, header=None, engine="openpyxl")
        if raw.shape[0] < 2 or raw.shape[1] < 9:
            return None
        v = raw.iat[1, 8]  # I2
        dt = pd.to_datetime(v, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return pd.Timestamp(dt) + pd.offsets.MonthEnd(0)
        return None
    except Exception:
        return None


# =========================================================
# Carga hoja Hola Valores
# =========================================================
def load_hola_valores(xls_source) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp], dict]:
    sheets = _excel_sheet_names(xls_source)

    sh = _pick_sheet(
        sheets,
        SHEET_HOLA,
        aliases=["hola valores", "hola_valores"]
    )
    if sh is None:
        raise ValueError(f"No encuentro la hoja '{SHEET_HOLA}'. Hojas disponibles: {sheets}")

    last_date_override = _find_last_date_override_from_I2(xls_source, sh)

    raw = _read_excel(xls_source, sh)
    if raw.shape[1] < 10:
        raise ValueError(
            f"La hoja '{sh}' no tiene la estructura esperada de 10 columnas. "
            f"Columnas detectadas: {list(raw.columns)}"
        )

    df = raw.iloc[:, :10].copy()
    df.columns = [
        "Codigo", "Fecha", "Nemo", "PesoAFP", "PesoIPSA", "Gap",
        "TieneAFP", "TieneIPSA", "UltimaFecha", "PrimeraFecha"
    ]

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True) + pd.offsets.MonthEnd(0)
    df["Nemo"] = df["Nemo"].astype(str).str.upper().str.strip()
    df["PesoAFP"] = pd.to_numeric(df["PesoAFP"], errors="coerce")
    df["PesoIPSA"] = pd.to_numeric(df["PesoIPSA"], errors="coerce")
    df["Gap"] = pd.to_numeric(df["Gap"], errors="coerce")

    df["TieneAFP"] = df["TieneAFP"].astype(str).str.strip().str.lower()
    df["TieneIPSA"] = df["TieneIPSA"].astype(str).str.strip().str.lower()

    def norm_flag(x):
        if x in ["tiene", "sí", "si", "yes", "y", "1", "true", "x"]:
            return "Tiene"
        return "No tiene"

    df["TieneAFP"] = df["TieneAFP"].apply(norm_flag)
    df["TieneIPSA"] = df["TieneIPSA"].apply(norm_flag)

    primera_fecha_series = pd.to_datetime(df["PrimeraFecha"], errors="coerce", dayfirst=True).dropna()
    primera_fecha = (primera_fecha_series.iloc[0] + pd.offsets.MonthEnd(0)) if len(primera_fecha_series) else None

    df = df.dropna(subset=["Fecha", "Nemo", "Gap"]).copy()

    if df.empty:
        raise ValueError("La hoja 'Hola Valores' no tiene filas válidas después de leer Fecha, Nemo y Gap.")

    if primera_fecha is not None:
        df = df[df["Fecha"] >= primera_fecha].copy()

    if last_date_override is not None:
        df = df[df["Fecha"] <= last_date_override].copy()

    if df.empty:
        raise ValueError(
            "Después de aplicar PrimeraFecha / ÚltimaFecha (I2), la base quedó vacía. "
            "Revisa Hola Valores!I2, J y la columna Fecha."
        )

    df = df.sort_values(["Nemo", "Fecha"]).reset_index(drop=True)

    meta = {"sheet_hola": sh, "sheets": sheets}
    return df, last_date_override, primera_fecha, meta


def load_ipsa_series(xls_source, meta: dict) -> Optional[pd.DataFrame]:
    sheets = meta["sheets"]
    sh_ipsa = _pick_sheet(sheets, SHEET_IPSA, aliases=["ipsa", "index", "indice", "ipsa hist", "ipsa_hist"])
    if sh_ipsa is None:
        return None

    ips = _read_excel(xls_source, sheet_name=sh_ipsa)
    if ips.empty:
        return None

    date_col = None
    for cand in ["Fecha", "fecha", "Date", "date"]:
        if cand in ips.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = ips.columns[0]

    value_col = None
    for cand in ["IPSA", "ipsa", "Close", "close", "Valor", "valor", "Index", "index", "Price", "price"]:
        if cand in ips.columns and cand != date_col:
            value_col = cand
            break
    if value_col is None and len(ips.columns) >= 2:
        value_col = [c for c in ips.columns if c != date_col][0]
    if value_col is None:
        return None

    out = ips[[date_col, value_col]].copy()
    out.columns = ["Fecha", "IPSA"]
    out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce", dayfirst=True) + pd.offsets.MonthEnd(0)
    out["IPSA"] = pd.to_numeric(out["IPSA"], errors="coerce")
    out = out.dropna(subset=["Fecha", "IPSA"]).sort_values("Fecha").reset_index(drop=True)
    return out


# =========================================================
# Features
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Nemo", group_keys=False)

    df["Delta_Gap"] = g["Gap"].diff()
    df["MA_3"] = g["Gap"].apply(lambda s: s.rolling(3, min_periods=3).mean())
    df["MA_6"] = g["Gap"].apply(lambda s: s.rolling(6, min_periods=6).mean())
    df["STD_6"] = g["Gap"].apply(lambda s: s.rolling(6, min_periods=6).std())
    df["Z_6"] = (df["Gap"] - df["MA_6"]) / (df["STD_6"].replace(0, np.nan))

    df["Gap_Pctl"] = g["Gap"].apply(lambda s: s.rank(pct=True))
    df["Delta_Pctl"] = g["Delta_Gap"].apply(lambda s: s.rank(pct=True))

    for lag in [1, 2, 3]:
        df[f"Gap_lag{lag}"] = g["Gap"].shift(lag)
        df[f"Delta_lag{lag}"] = g["Delta_Gap"].shift(lag)

    df["Rank_Gap_mes"] = df.groupby("Fecha")["Gap"].rank(ascending=False, method="dense")

    df["Delta_next"] = g["Delta_Gap"].shift(-1)
    df["Up_next"] = (df["Delta_next"] > 0).astype(int)

    return df


# =========================================================
# Señales
# =========================================================
def _class_movimiento(delta_gap: float, umbral: float = 0.0) -> str:
    if pd.isna(delta_gap):
        return "Sin dato"
    if delta_gap > umbral:
        return "Comprando"
    if delta_gap < -umbral:
        return "Vendiendo"
    return "Manteniendo"


def _class_posicion(gap: float, umbral: float = 0.0) -> str:
    if pd.isna(gap):
        return "Neutral"
    if gap > umbral:
        return "Largo"
    if gap < -umbral:
        return "Corto"
    return "Neutral"


def add_rules_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Nemo", group_keys=False)

    delta_ma3 = g["Delta_Gap"].transform(lambda s: s.rolling(3, min_periods=2).mean())

    df["Score_Flujo"] = (
        1.5 * (df["Gap"] > 0).astype(int) +
        2.0 * (df["Delta_Gap"] > 0).astype(int) +
        1.0 * (df["Rank_Gap_mes"] <= 10).astype(int) +
        1.0 * (df["Gap"] > df["MA_3"]).astype(int) +
        1.0 * (df["Delta_Gap"] > delta_ma3.fillna(0)).astype(int)
    )

    df["Score_RiesgoSalida"] = (
        2.0 * (df["Gap_Pctl"] >= 0.85).astype(int) +
        1.5 * (df["Z_6"] >= 1.0).astype(int) +
        2.0 * (df["Delta_Gap"] < 0).astype(int) +
        1.5 * (df["Gap"] < df["MA_3"]).astype(int)
    )

    def normalize_0_100(s):
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series([50.0] * len(s), index=s.index)
        return 100 * (s - mn) / (mx - mn)

    df["FlowScore_0_100"] = df.groupby("Fecha")["Score_Flujo"].transform(normalize_0_100)

    df["Movimiento"] = df["Delta_Gap"].apply(_class_movimiento)
    df["Posicion"] = df["Gap"].apply(_class_posicion)

    df["Compra_Fuerte"] = (
        (df["Delta_Pctl"] >= 0.85) &
        (df["Delta_Gap"] > 0) &
        (df["Gap"] > df["MA_3"])
    )

    df["Venta_Fuerte"] = (
        (df["Delta_Pctl"] <= 0.15) &
        (df["Delta_Gap"] < 0) &
        (df["Gap"] < df["MA_3"])
    )

    df["CompraVenta_Fuerte"] = "Neutral"
    df.loc[df["Compra_Fuerte"], "CompraVenta_Fuerte"] = "Compra fuerte"
    df.loc[df["Venta_Fuerte"], "CompraVenta_Fuerte"] = "Venta fuerte"

    delta_prev = g["Delta_Gap"].shift(1)

    df["Flujo_AFP"] = "Manteniendo"
    df.loc[(df["Delta_Gap"] > 0) & (df["Delta_Gap"] >= delta_prev.fillna(-999)), "Flujo_AFP"] = "Entrada activas"
    df.loc[
        (df["Delta_Gap"] > 0) & (df["Delta_Gap"] < delta_prev.fillna(999)) & (g["Delta_Pctl"].shift(1) >= 0.75),
        "Flujo_AFP"
    ] = "Entrada seguidoras"
    df.loc[(df["Delta_Gap"] < 0) & (df["Delta_Gap"] <= delta_prev.fillna(999)), "Flujo_AFP"] = "Salida activas"
    df.loc[
        (df["Delta_Gap"] < 0) & (df["Delta_Gap"] > delta_prev.fillna(-999)) & (delta_prev < 0),
        "Flujo_AFP"
    ] = "Salida seguidoras"

    df["Senal"] = "Manteniendo"
    df.loc[df["CompraVenta_Fuerte"] == "Compra fuerte", "Senal"] = "Compra fuerte"
    df.loc[df["CompraVenta_Fuerte"] == "Venta fuerte", "Senal"] = "Venta fuerte"
    df.loc[(df["Senal"] == "Manteniendo") & (df["Flujo_AFP"] == "Entrada activas"), "Senal"] = "Entrada activas"
    df.loc[(df["Senal"] == "Manteniendo") & (df["Flujo_AFP"] == "Entrada seguidoras"), "Senal"] = "Entrada seguidoras"
    df.loc[(df["Senal"] == "Manteniendo") & (df["Flujo_AFP"] == "Salida activas"), "Senal"] = "Salida activas"
    df.loc[(df["Senal"] == "Manteniendo") & (df["Flujo_AFP"] == "Salida seguidoras"), "Senal"] = "Salida seguidoras"
    df.loc[(df["Senal"] == "Manteniendo") & (df["Movimiento"] == "Comprando"), "Senal"] = "Comprando"
    df.loc[(df["Senal"] == "Manteniendo") & (df["Movimiento"] == "Vendiendo"), "Senal"] = "Vendiendo"

    def semaforo(signal: str) -> str:
        if signal in ["Compra fuerte", "Entrada activas"]:
            return "🟢"
        if signal in ["Entrada seguidoras", "Comprando", "Manteniendo"]:
            return "🟡"
        if signal in ["Salida seguidoras", "Salida activas", "Venta fuerte", "Vendiendo"]:
            return "🔴"
        return "🟡"

    df["Semaforo"] = df["Senal"].apply(semaforo)
    return df


# =========================================================
# Modelos
# =========================================================
def train_predict_global(df_feat: pd.DataFrame):
    dfm = df_feat.copy()

    feature_cols_num = [
        "Gap", "Delta_Gap",
        "MA_3", "MA_6", "STD_6", "Z_6",
        "Gap_Pctl", "Rank_Gap_mes",
        "Gap_lag1", "Gap_lag2", "Gap_lag3",
        "Delta_lag1", "Delta_lag2", "Delta_lag3"
    ]
    feature_cols_num = [c for c in feature_cols_num if c in dfm.columns]

    dfm = dfm.dropna(subset=feature_cols_num + ["Up_next", "Delta_next"])
    if dfm.empty:
        raise ValueError("No quedaron filas suficientes para entrenar el modelo después de construir features.")

    dfm = dfm.sort_values(["Fecha", "Nemo"]).reset_index(drop=True)

    X = dfm[["Nemo"] + feature_cols_num]
    y_cls = dfm["Up_next"].astype(int)
    y_reg = dfm["Delta_next"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Nemo"]),
            ("num", "passthrough", feature_cols_num),
        ],
        remainder="drop"
    )

    clf = Pipeline(steps=[("pre", pre), ("model", LogisticRegression(max_iter=900))])
    reg = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])

    tss = TimeSeriesSplit(n_splits=5)
    aucs, accs = [], []

    for tr_idx, te_idx in tss.split(X, y_cls):
        clf.fit(X.iloc[tr_idx], y_cls.iloc[tr_idx])
        proba = clf.predict_proba(X.iloc[te_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            aucs.append(roc_auc_score(y_cls.iloc[te_idx], proba))
        except Exception:
            pass
        accs.append(accuracy_score(y_cls.iloc[te_idx], pred))

    clf.fit(X, y_cls)
    reg.fit(X, y_reg)

    dfm["P_Up_next"] = clf.predict_proba(X)[:, 1]
    dfm["Delta_next_hat"] = reg.predict(X)

    metrics = {
        "AUC_mean": float(np.nanmean(aucs)) if len(aucs) else np.nan,
        "ACC_mean": float(np.mean(accs)) if len(accs) else np.nan,
        "rows": int(len(dfm))
    }
    return dfm, metrics


# =========================================================
# Acciones + rankings
# =========================================================
def add_actions(df_model: pd.DataFrame) -> pd.DataFrame:
    dfm = df_model.copy()

    def tactical(r):
        p = r.get("P_Up_next", np.nan)
        fs = r.get("FlowScore_0_100", np.nan)
        senal = r.get("Senal", "Manteniendo")

        if pd.notna(p) and pd.notna(fs):
            if (p >= 0.60) and (fs >= 70) and (senal in ["Compra fuerte", "Entrada activas"]):
                return "Comprar"
            if (p >= 0.60) and (senal in ["Compra fuerte", "Entrada activas", "Entrada seguidoras", "Comprando"]):
                return "Comprar suave"
            if (p <= 0.40) and (senal in ["Venta fuerte", "Salida activas", "Salida seguidoras", "Vendiendo"]):
                return "Vender / Reducir"
            if (p <= 0.40):
                return "Reducir"
            return "Mantener"

        if senal in ["Compra fuerte", "Entrada activas"]:
            return "Comprar"
        if senal in ["Venta fuerte", "Salida activas"]:
            return "Reducir"
        return "Mantener"

    def relative(r):
        fs = r.get("FlowScore_0_100", np.nan)
        p = r.get("P_Up_next", np.nan)

        if pd.isna(fs):
            fs = 50
        if pd.isna(p):
            p = 0.50

        tilt = 2.0 * (fs - 50) + 200 * (p - 0.50)
        tilt = float(np.clip(tilt, -200, 200))

        if tilt >= 60:
            return "Sobreponderar", tilt
        if tilt <= -60:
            return "Subponderar", tilt
        return "Neutral", tilt

    def timing_trade(r):
        flujo = r.get("Flujo_AFP", "Manteniendo")
        if flujo == "Entrada seguidoras":
            return "Comprar en T"
        if flujo == "Entrada activas":
            return "Comprar / Mantener"
        if flujo == "Salida activas":
            return "Vender / Reducir"
        if flujo == "Salida seguidoras":
            return "Reducir suave"
        return "Mantener"

    dfm["Accion_Tactica"] = dfm.apply(tactical, axis=1)
    out_r = dfm.apply(relative, axis=1, result_type="expand")
    dfm["Accion_Relativa"] = out_r[0]
    dfm["Tilt_bps"] = out_r[1].astype(float)
    dfm["Recomendacion_Timing"] = dfm.apply(timing_trade, axis=1)

    dfm["Prob_Compra_AFP_ProxMes"] = (dfm["P_Up_next"] * 100).clip(0, 100)

    def normalize_rank(df, col):
        return df.groupby("Fecha")[col].transform(lambda x: x.rank(pct=True))

    dfm["Rank_ProbCompra"] = normalize_rank(dfm, "Prob_Compra_AFP_ProxMes")
    dfm["Rank_Delta"] = normalize_rank(dfm, "Delta_Gap")
    dfm["Rank_Flow"] = normalize_rank(dfm, "FlowScore_0_100")

    dfm["AFP_Flow_Rank"] = (
        0.45 * dfm["Rank_ProbCompra"] +
        0.30 * dfm["Rank_Delta"] +
        0.25 * dfm["Rank_Flow"]
    ) * 100

    dfm["Prob_Entrada_AFP"] = (
        0.60 * (dfm["Delta_Gap"] > 0).astype(int) +
        0.25 * (dfm["Gap"] > dfm["MA_3"]).astype(int) +
        0.15 * (dfm["FlowScore_0_100"] >= 70).astype(int)
    ) * 100

    dfm["Prob_Salida_AFP"] = (
        0.60 * (dfm["Delta_Gap"] < 0).astype(int) +
        0.25 * (dfm["Gap"] < dfm["MA_3"]).astype(int) +
        0.15 * (dfm["Score_RiesgoSalida"] >= 4).astype(int)
    ) * 100

    return dfm


# =========================================================
# Eventos
# =========================================================
def build_events(df_model: pd.DataFrame) -> pd.DataFrame:
    events = []

    for paper in df_model["Nemo"].unique():
        d = df_model[df_model["Nemo"] == paper].sort_values("Fecha").copy()
        prev = None

        for _, r in d.iterrows():
            sig = r.get("Senal", None)
            if pd.isna(sig):
                continue

            if prev is None or sig != prev:
                events.append({
                    "Nemo": paper,
                    "Fecha": r["Fecha"],
                    "Semaforo": r.get("Semaforo", ""),
                    "Senal": sig,
                    "Nota": f"Cambio de señal: {prev} → {sig}" if prev else f"Inicio: {sig}",
                    "Gap": r.get("Gap", np.nan),
                    "Delta_Gap": r.get("Delta_Gap", np.nan),
                    "Flujo_AFP": r.get("Flujo_AFP", ""),
                    "CompraVenta_Fuerte": r.get("CompraVenta_Fuerte", "Neutral"),
                    "Prob_Compra_AFP_ProxMes": r.get("Prob_Compra_AFP_ProxMes", np.nan),
                })
                prev = sig

    return pd.DataFrame(events).sort_values(["Nemo", "Fecha"]).reset_index(drop=True)


# =========================================================
# Apéndice
# =========================================================
def build_appendix_tables() -> dict:
    reglas = pd.DataFrame([
        ["Compra fuerte", "Delta Gap muy positivo dentro de la historia del papel y Gap sobre su media corta.", "Percentil Delta >= 85% y Gap > media 3 meses."],
        ["Venta fuerte", "Delta Gap muy negativo dentro de la historia del papel y Gap bajo su media corta.", "Percentil Delta <= 15% y Gap < media 3 meses."],
        ["Entrada activas", "El Gap sube y el flujo actual es igual o más fuerte que el mes previo.", "Delta Gap > 0 y Delta actual >= Delta previo."],
        ["Entrada seguidoras", "El Gap sube, pero con menor fuerza que el mes anterior, siguiendo una entrada previa.", "Delta Gap > 0, Delta actual < Delta previo y percentil Delta previo >= 75%."],
        ["Salida activas", "El Gap cae y la salida actual es igual o más fuerte que el mes previo.", "Delta Gap < 0 y Delta actual <= Delta previo."],
        ["Salida seguidoras", "El Gap cae, pero la caída actual es menor que la del mes anterior, siguiendo una salida previa.", "Delta Gap < 0, Delta actual > Delta previo y Delta previo < 0."],
        ["Comprando", "Cambio mensual positivo, sin señal fuerte.", "Delta Gap > 0."],
        ["Vendiendo", "Cambio mensual negativo, sin señal fuerte.", "Delta Gap < 0."],
        ["Manteniendo", "Sin cambio relevante o sin señal dominante.", "Caso residual."],
    ], columns=["Señal", "Qué significa", "Cómo se calcula"])

    prob = pd.DataFrame([
        ["Probabilidad de compra AFP próximo mes", "Modelo logístico que estima la probabilidad de que el Delta Gap del próximo mes sea positivo.", "Gap actual, Delta Gap actual, medias móviles, desvío, percentiles y rezagos por papel."],
        ["Rango 0% - 100%", "0% indica muy baja probabilidad de aumento del Gap próximo mes; 100% indica muy alta probabilidad.", "Se muestra como P_Up_next x 100."],
        ["Probabilidad de entrada AFP", "Score simple orientado a detectar compras probables.", "60% si Delta Gap > 0 + 25% si Gap > media 3M + 15% si FlowScore >= 70."],
        ["Probabilidad de salida AFP", "Score simple orientado a detectar ventas probables.", "60% si Delta Gap < 0 + 25% si Gap < media 3M + 15% si Score_RiesgoSalida >= 4."],
    ], columns=["Concepto", "Explicación", "Cálculo"])

    acciones = pd.DataFrame([
        ["Acción táctica", "Recomendación operativa de corto plazo.", "Se basa en señal actual, FlowScore y probabilidad de compra."],
        ["Comprar", "Setup fuerte de compra.", "Probabilidad >= 60%, FlowScore >= 70 y señal verde fuerte."],
        ["Comprar suave", "Señal favorable, pero menos contundente.", "Probabilidad >= 60% con señal favorable."],
        ["Mantener", "Sin ventaja táctica clara.", "Caso intermedio."],
        ["Reducir", "Deterioro moderado.", "Probabilidad <= 40% sin señal roja extrema."],
        ["Vender / Reducir", "Deterioro claro.", "Probabilidad <= 40% con señal de venta."],
        ["Acción relativa", "Posicionamiento relativo frente al universo.", "Combina FlowScore y probabilidad de compra."],
        ["Sobreponderar", "Preferir este papel versus otros.", "Tilt >= 60."],
        ["Neutral", "Sin sesgo relativo fuerte.", "-60 < Tilt < 60."],
        ["Subponderar", "Menor preferencia relativa.", "Tilt <= -60."],
    ], columns=["Concepto", "Qué significa", "Cómo se calcula"])

    metricas = pd.DataFrame([
        ["Gap", "Peso AFP - Peso IPSA", "Se muestra en porcentaje."],
        ["Delta Gap", "Cambio mensual del Gap", "Gap(t) - Gap(t-1), se muestra en porcentaje."],
        ["Posición", "Ubicación relativa frente al IPSA", "Largo si Gap > 0, Corto si Gap < 0, Neutral si Gap = 0."],
        ["Movimiento", "Dirección del cambio mensual", "Comprando si Delta Gap > 0, Vendiendo si Delta Gap < 0, Manteniendo si Delta Gap = 0."],
        ["FlowScore", "Score de flujo institucional dentro del mes", "Normalización 0 a 100 del Score_Flujo por fecha."],
        ["AFP Flow Rank", "Ranking institucional compuesto dentro del mes", "45% rank probabilidad compra + 30% rank Delta Gap + 25% rank FlowScore."],
    ], columns=["Métrica", "Definición", "Cómo leerla"])

    rangos = pd.DataFrame([
        ["Gap %", "Porcentaje", "Ejemplo: 2.5% significa que AFP pesa 2.5 puntos porcentuales más que IPSA."],
        ["Delta Gap %", "Porcentaje", "Ejemplo: 0.4% significa que el gap subió 0.4 puntos porcentuales en el mes."],
        ["Probabilidades", "0% a 100%", "Más alto = mayor chance esperada."],
        ["FlowScore / AFP Flow Rank", "0 a 100", "Más alto = mejor posición relativa en el mes."],
    ], columns=["Variable", "Rango", "Lectura"])

    return {
        "reglas_senales": reglas,
        "probabilidades": prob,
        "acciones": acciones,
        "metricas": metricas,
        "rangos": rangos,
    }


# =========================================================
# Outputs finales
# =========================================================
def build_outputs(xls_source):
    df, last_date_override, primera_fecha, meta = load_hola_valores(xls_source)

    df = add_features(df)
    df = add_rules_signals(df)

    dfm, metrics = train_predict_global(df)
    dfm = add_actions(dfm)

    df_ipsa = load_ipsa_series(xls_source, meta)

    last_date = dfm["Fecha"].max()
    if last_date_override is not None:
        if (dfm["Fecha"] == last_date_override).any():
            last_date = last_date_override
        else:
            ym = (
                (dfm["Fecha"].dt.year == last_date_override.year) &
                (dfm["Fecha"].dt.month == last_date_override.month)
            )
            if ym.any():
                last_date = dfm.loc[ym, "Fecha"].max()

    snap_last = dfm[dfm["Fecha"] == last_date].copy()
    if snap_last.empty:
        raise ValueError("No se pudo construir el snapshot de la última fecha. Revisa I2 y la columna Fecha.")

    snap_last["Movimiento"] = snap_last["Delta_Gap"].apply(_class_movimiento)
    snap_last["Posicion"] = snap_last["Gap"].apply(_class_posicion)

    ranking_entrada = snap_last.sort_values(
        ["Prob_Entrada_AFP", "Prob_Compra_AFP_ProxMes", "AFP_Flow_Rank", "Delta_Gap"],
        ascending=False
    ).copy()

    ranking_salida = snap_last.sort_values(
        ["Prob_Salida_AFP", "Delta_Gap", "Gap"],
        ascending=[False, True, True]
    ).copy()

    top_compras = snap_last.sort_values(
        ["Prob_Compra_AFP_ProxMes", "AFP_Flow_Rank", "Delta_Gap"],
        ascending=False
    ).head(20).copy()

    top_ventas = snap_last.sort_values(
        ["Prob_Salida_AFP", "Delta_Gap", "Gap"],
        ascending=[False, True, True]
    ).head(20).copy()

    events = build_events(dfm)
    appendix = build_appendix_tables()

    return (
        df,
        dfm,
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
    )
