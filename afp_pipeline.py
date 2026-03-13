import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any
from pathlib import Path

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

    low_map = {str(s).strip().lower(): s for s in available}
    if want.lower() in low_map:
        return low_map[want.lower()]

    for a in aliases:
        if a in available:
            return a
        if a.lower() in low_map:
            return low_map[a.lower()]

    for s in available:
        sl = str(s).strip().lower()
        if want.lower() in sl:
            return s
        for a in aliases:
            if a.lower() in sl:
                return s
    return None


def _read_excel(xls_source, sheet_name: str, header: Any = 0) -> pd.DataFrame:
    return pd.read_excel(xls_source, sheet_name=sheet_name, header=header, engine="openpyxl")


def _month_end(series_or_value):
    out = pd.to_datetime(series_or_value, errors="coerce", dayfirst=True)
    return out + pd.offsets.MonthEnd(0)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# =========================================================
# Fecha override desde Hola Valores!I2
# =========================================================
def _find_last_date_override_from_I2(xls_source, sheet_name: str) -> Optional[pd.Timestamp]:
    try:
        raw = _read_excel(xls_source, sheet_name=sheet_name, header=None)
        v = raw.iat[1, 8]  # I2
        dt = pd.to_datetime(v, errors="coerce", dayfirst=True)
        if pd.notna(dt):
            return pd.Timestamp(dt) + pd.offsets.MonthEnd(0)
        return None
    except Exception:
        return None


# =========================================================
# Carga hoja Hola Valores
# A: codigo
# B: fecha
# C: nemo
# D: peso cartera AFP
# E: peso IPSA
# F: gap
# G: si AFP tiene o no
# H: si IPSA tiene o no
# I: ultima fecha
# J: primera fecha
# =========================================================
def load_hola_valores(xls_source) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp], dict]:
    sheets = _excel_sheet_names(xls_source)

    sh = _pick_sheet(
        sheets,
        SHEET_HOLA,
        aliases=["hola valores", "hola_valores", "hola-valores", "hola"]
    )
    if sh is None:
        raise ValueError(f"No encuentro la hoja '{SHEET_HOLA}'. Hojas disponibles: {sheets}")

    last_date_override = _find_last_date_override_from_I2(xls_source, sh)

    raw = _read_excel(xls_source, sh)
    if raw.shape[1] < 10:
        raise ValueError(
            f"La hoja '{sh}' no tiene la estructura esperada de al menos 10 columnas. "
            f"Columnas detectadas: {list(raw.columns)}"
        )

    df = raw.iloc[:, :10].copy()
    df.columns = [
        "Codigo", "Fecha", "Nemo", "PesoAFP", "PesoIPSA", "Gap",
        "TieneAFP", "TieneIPSA", "UltimaFecha", "PrimeraFecha"
    ]

    df["Fecha"] = _month_end(df["Fecha"])
    df["Nemo"] = df["Nemo"].astype(str).str.upper().str.strip()
    df["Codigo"] = df["Codigo"].astype(str).str.strip()

    for c in ["PesoAFP", "PesoIPSA", "Gap"]:
        df[c] = _safe_numeric(df[c])

    def norm_flag(x):
        x = str(x).strip().lower()
        if x in ["tiene", "sí", "si", "yes", "y", "1", "true", "x"]:
            return "Tiene"
        return "No tiene"

    df["TieneAFP"] = df["TieneAFP"].apply(norm_flag)
    df["TieneIPSA"] = df["TieneIPSA"].apply(norm_flag)

    primera_fecha_series = _month_end(df["PrimeraFecha"]).dropna()
    primera_fecha = primera_fecha_series.iloc[0] if len(primera_fecha_series) else None

    df = df.dropna(subset=["Fecha", "Nemo", "Gap"]).copy()
    df = df[df["Nemo"].ne("") & df["Nemo"].ne("NAN")].copy()

    if primera_fecha is not None:
        df = df[df["Fecha"] >= primera_fecha].copy()

    if last_date_override is not None:
        df = df[df["Fecha"] <= last_date_override].copy()

    # Dedupe por papel/fecha: conserva la última fila no nula
    df = (
        df.sort_values(["Nemo", "Fecha"])
        .drop_duplicates(subset=["Nemo", "Fecha"], keep="last")
        .reset_index(drop=True)
    )

    meta = {"sheet_hola": sh, "sheets": sheets}
    return df, last_date_override, primera_fecha, meta



def _resolve_local_ipsa_file() -> Optional[str]:
    candidates = [
        Path("IPSA Hist.xlsx"),
        Path("IPSA_Hist.xlsx"),
        Path("IPSA.xlsx"),
        Path("/mnt/data/IPSA Hist.xlsx"),
        Path("/mnt/data/IPSA_Hist.xlsx"),
        Path("/mnt/data/IPSA.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def load_ipsa_series(xls_source, meta: dict) -> Optional[pd.DataFrame]:
    sheets = meta["sheets"]
    sh_ipsa = _pick_sheet(
        sheets,
        SHEET_IPSA,
        aliases=["ipsa", "index", "indice", "ipsa hist", "ipsa_hist", "ipsa histórica", "ipsa historica"]
    )

    ips_source = xls_source
    if sh_ipsa is None:
        local_ipsa = _resolve_local_ipsa_file()
        if local_ipsa is None:
            return None
        try:
            local_sheets = _excel_sheet_names(local_ipsa)
            sh_ipsa = _pick_sheet(
                local_sheets,
                SHEET_IPSA,
                aliases=["ipsa", "index", "indice", "ipsa hist", "ipsa_hist", "ipsa histórica", "ipsa historica"]
            )
            if sh_ipsa is None and len(local_sheets) == 1:
                sh_ipsa = local_sheets[0]
            ips_source = local_ipsa
        except Exception:
            return None

    ips = _read_excel(ips_source, sheet_name=sh_ipsa)
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
        other_cols = [c for c in ips.columns if c != date_col]
        value_col = other_cols[0] if other_cols else None
    if value_col is None:
        return None

    out = ips[[date_col, value_col]].copy()
    out.columns = ["Fecha", "IPSA"]
    out["Fecha"] = _month_end(out["Fecha"])
    out["IPSA"] = _safe_numeric(out["IPSA"])
    out = (
        out.dropna(subset=["Fecha", "IPSA"])
        .sort_values("Fecha")
        .drop_duplicates(subset=["Fecha"], keep="last")
        .reset_index(drop=True)
    )

    out["Ret_IPSA_1M"] = out["IPSA"].pct_change()
    out["MA_IPSA_3"] = out["IPSA"].rolling(3, min_periods=3).mean()
    return out


# =========================================================
# Features
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Nemo", "Fecha"]).reset_index(drop=True)
    g = df.groupby("Nemo", group_keys=False)

    df["Delta_Gap"] = g["Gap"].diff()
    df["MA_3"] = g["Gap"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    df["MA_6"] = g["Gap"].transform(lambda s: s.rolling(6, min_periods=6).mean())
    df["STD_6"] = g["Gap"].transform(lambda s: s.rolling(6, min_periods=6).std())
    df["Z_6"] = (df["Gap"] - df["MA_6"]) / (df["STD_6"].replace(0, np.nan))

    # Percentiles por papel
    df["Gap_Pctl"] = g["Gap"].rank(pct=True)
    df["Delta_Pctl"] = g["Delta_Gap"].rank(pct=True)

    # Percentiles cross-sectional del mes
    df["Gap_Pctl_mes"] = df.groupby("Fecha")["Gap"].rank(pct=True)
    df["Delta_Pctl_mes"] = df.groupby("Fecha")["Delta_Gap"].rank(pct=True)

    for lag in [1, 2, 3]:
        df[f"Gap_lag{lag}"] = g["Gap"].shift(lag)
        df[f"Delta_lag{lag}"] = g["Delta_Gap"].shift(lag)

    df["Rank_Gap_mes"] = df.groupby("Fecha")["Gap"].rank(ascending=False, method="dense")
    df["N_Papeles_mes"] = df.groupby("Fecha")["Nemo"].transform("count")

    # Persistencia / régimen
    df["Gap_Pos_3m"] = g["Gap"].transform(lambda s: (s > 0).rolling(3, min_periods=1).mean())
    df["Delta_Pos_3m"] = g["Delta_Gap"].transform(lambda s: (s > 0).rolling(3, min_periods=1).mean())
    df["Aceleracion_Gap"] = g["Delta_Gap"].diff()

    # Objetivo siguiente mes
    df["Delta_next"] = g["Delta_Gap"].shift(-1)
    df["Up_next"] = (df["Delta_next"] > 0).astype(int)

    return df


# =========================================================
# Señales amigables
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
    df = df.copy().sort_values(["Nemo", "Fecha"]).reset_index(drop=True)
    g = df.groupby("Nemo", group_keys=False)

    delta_ma3 = g["Delta_Gap"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    gap_abs_q75_mes = df.groupby("Fecha")["Gap"].transform(lambda s: s.abs().quantile(0.75))
    delta_abs_q75_mes = df.groupby("Fecha")["Delta_Gap"].transform(lambda s: s.abs().quantile(0.75))

    df["Score_Flujo"] = (
        1.50 * (df["Gap"] > 0).astype(int) +
        2.00 * (df["Delta_Gap"] > 0).astype(int) +
        1.00 * (df["Rank_Gap_mes"] <= np.maximum(10, np.ceil(df["N_Papeles_mes"] * 0.20))).astype(int) +
        1.00 * (df["Gap"] > df["MA_3"]).astype(int) +
        1.00 * (df["Delta_Gap"] > delta_ma3.fillna(0)).astype(int) +
        0.75 * (df["Gap_Pos_3m"] >= 2/3).astype(int) +
        0.75 * (df["Delta_Pos_3m"] >= 2/3).astype(int)
    )

    df["Score_RiesgoSalida"] = (
        2.00 * (df["Gap_Pctl"] >= 0.85).astype(int) +
        1.50 * (df["Z_6"] >= 1.0).astype(int) +
        2.00 * (df["Delta_Gap"] < 0).astype(int) +
        1.50 * (df["Gap"] < df["MA_3"]).astype(int) +
        1.00 * (df["Aceleracion_Gap"] < 0).astype(int)
    )

    def normalize_0_100(s: pd.Series) -> pd.Series:
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
        (df["Gap"] > df["MA_3"]) &
        (df["Delta_Gap"].abs() >= delta_abs_q75_mes.fillna(np.inf))
    )

    df["Venta_Fuerte"] = (
        (df["Delta_Pctl"] <= 0.15) &
        (df["Delta_Gap"] < 0) &
        (df["Gap"] < df["MA_3"]) &
        (df["Delta_Gap"].abs() >= delta_abs_q75_mes.fillna(np.inf))
    )

    df["CompraVenta_Fuerte"] = "Neutral"
    df.loc[df["Compra_Fuerte"], "CompraVenta_Fuerte"] = "Compra fuerte"
    df.loc[df["Venta_Fuerte"], "CompraVenta_Fuerte"] = "Venta fuerte"

    delta_prev = g["Delta_Gap"].shift(1)
    prev_fuerte_buy = g["Compra_Fuerte"].shift(1).fillna(False)
    prev_fuerte_sell = g["Venta_Fuerte"].shift(1).fillna(False)

    df["Flujo_AFP"] = "Manteniendo"

    # Liderazgo activo: compra/venta fuerte o aceleración clara
    df.loc[
        (df["Delta_Gap"] > 0) &
        ((df["Delta_Gap"] >= delta_prev.fillna(-np.inf)) | df["Compra_Fuerte"]),
        "Flujo_AFP"
    ] = "Entrada activas"

    df.loc[
        (df["Delta_Gap"] < 0) &
        ((df["Delta_Gap"] <= delta_prev.fillna(np.inf)) | df["Venta_Fuerte"]),
        "Flujo_AFP"
    ] = "Salida activas"

    # Seguidoras: movimiento en la misma dirección luego de una señal fuerte / activa previa
    df.loc[
        (df["Delta_Gap"] > 0) &
        (df["Delta_Gap"] < delta_prev.fillna(np.inf)) &
        ((g["Delta_Pctl"].shift(1) >= 0.75) | prev_fuerte_buy),
        "Flujo_AFP"
    ] = "Entrada seguidoras"

    df.loc[
        (df["Delta_Gap"] < 0) &
        (df["Delta_Gap"] > delta_prev.fillna(-np.inf)) &
        ((delta_prev < 0) | prev_fuerte_sell),
        "Flujo_AFP"
    ] = "Salida seguidoras"

    # Indicador explícito líder-seguidor T -> T+1
    df["Lider_Compra_T"] = (
        (g["Compra_Fuerte"].shift(1).fillna(False)) |
        (g["Flujo_AFP"].shift(1).fillna("").eq("Entrada activas"))
    )
    df["Seguidor_Compra_T1"] = df["Lider_Compra_T"] & (df["Flujo_AFP"] == "Entrada seguidoras")

    df["Lider_Venta_T"] = (
        (g["Venta_Fuerte"].shift(1).fillna(False)) |
        (g["Flujo_AFP"].shift(1).fillna("").eq("Salida activas"))
    )
    df["Seguidor_Venta_T1"] = df["Lider_Venta_T"] & (df["Flujo_AFP"] == "Salida seguidoras")

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
def _fallback_predictions(dfm: pd.DataFrame):
    """Fallback robusto cuando hay poca historia o una sola clase."""
    out = dfm.copy()

    # Heurística suave y estable, escalada a [0.05, 0.95]
    z = (
        0.45 * out["Delta_Gap"].fillna(0) +
        0.30 * (out["Gap"] - out["MA_3"]).fillna(0) +
        0.15 * out["Aceleracion_Gap"].fillna(0) +
        0.10 * (out["FlowScore_0_100"].fillna(50) - 50) / 100
    )
    if z.notna().any():
        std = z.std()
        z = z / std if pd.notna(std) and std not in [0, np.nan] else z
    p = 1 / (1 + np.exp(-z.clip(-5, 5)))
    out["P_Up_next"] = (0.90 * p + 0.05).clip(0.05, 0.95)
    out["Delta_next_hat"] = (
        0.60 * out["Delta_Gap"].fillna(0) +
        0.25 * (out["Gap"] - out["MA_3"]).fillna(0) +
        0.15 * out["Aceleracion_Gap"].fillna(0)
    )

    base_rate = float(out["Up_next"].mean()) if len(out) else np.nan
    metrics = {
        "AUC_mean": np.nan,
        "ACC_mean": np.nan,
        "rows": int(len(out)),
        "model_type": "fallback_heuristic",
        "base_rate_up": base_rate,
    }
    return out, metrics


def train_predict_global(df_feat: pd.DataFrame):
    dfm = df_feat.copy()

    feature_cols_num = [
        "Gap", "Delta_Gap",
        "MA_3", "MA_6", "STD_6", "Z_6",
        "Gap_Pctl", "Gap_Pctl_mes", "Delta_Pctl_mes",
        "Rank_Gap_mes",
        "Gap_Pos_3m", "Delta_Pos_3m", "Aceleracion_Gap",
        "Gap_lag1", "Gap_lag2", "Gap_lag3",
        "Delta_lag1", "Delta_lag2", "Delta_lag3",
        "FlowScore_0_100", "Score_RiesgoSalida"
    ]
    feature_cols_num = [c for c in feature_cols_num if c in dfm.columns]

    dfm = dfm.dropna(subset=feature_cols_num + ["Up_next", "Delta_next"])
    dfm = dfm.sort_values(["Fecha", "Nemo"]).reset_index(drop=True)

    if len(dfm) < 30 or dfm["Up_next"].nunique() < 2:
        return _fallback_predictions(dfm)

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

    clf = Pipeline(steps=[
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1200, class_weight="balanced"))
    ])
    reg = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])

    # Ajusta número de splits a la historia disponible
    n_splits = min(5, max(2, dfm["Fecha"].nunique() - 1))
    tss = TimeSeriesSplit(n_splits=n_splits)
    aucs, accs = [], []

    for tr_idx, te_idx in tss.split(X, y_cls):
        y_tr = y_cls.iloc[tr_idx]
        y_te = y_cls.iloc[te_idx]
        if y_tr.nunique() < 2:
            continue

        clf.fit(X.iloc[tr_idx], y_tr)
        proba = clf.predict_proba(X.iloc[te_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)

        if y_te.nunique() >= 2:
            try:
                aucs.append(roc_auc_score(y_te, proba))
            except Exception:
                pass
        accs.append(accuracy_score(y_te, pred))

    clf.fit(X, y_cls)
    reg.fit(X, y_reg)

    dfm["P_Up_next"] = clf.predict_proba(X)[:, 1]
    dfm["Delta_next_hat"] = reg.predict(X)

    metrics = {
        "AUC_mean": float(np.nanmean(aucs)) if len(aucs) else np.nan,
        "ACC_mean": float(np.mean(accs)) if len(accs) else np.nan,
        "rows": int(len(dfm)),
        "model_type": "logistic_ridge",
        "base_rate_up": float(y_cls.mean()),
        "n_splits_cv": int(n_splits),
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
        0.50 * (dfm["Delta_Gap"] > 0).astype(int) +
        0.20 * (dfm["Gap"] > dfm["MA_3"]).astype(int) +
        0.15 * (dfm["FlowScore_0_100"] >= 70).astype(int) +
        0.15 * (dfm["Flujo_AFP"].isin(["Entrada activas", "Entrada seguidoras"])).astype(int)
    ) * 100

    dfm["Prob_Salida_AFP"] = (
        0.50 * (dfm["Delta_Gap"] < 0).astype(int) +
        0.20 * (dfm["Gap"] < dfm["MA_3"]).astype(int) +
        0.15 * (dfm["Score_RiesgoSalida"] >= 4).astype(int) +
        0.15 * (dfm["Flujo_AFP"].isin(["Salida activas", "Salida seguidoras"])).astype(int)
    ) * 100

    # Métrica explícita líder-seguidor T -> T+1
    dfm["Prob_Seguimiento_T1"] = (
        0.50 * dfm["Lider_Compra_T"].astype(int) +
        0.25 * (dfm["Delta_Gap"] > 0).astype(int) +
        0.15 * (dfm["P_Up_next"] >= 0.55).astype(int) +
        0.10 * (dfm["FlowScore_0_100"] >= 60).astype(int)
    ) * 100

    return dfm


# =========================================================
# Eventos
# =========================================================
def build_events(df_model: pd.DataFrame) -> pd.DataFrame:
    events = []

    for paper in df_model["Nemo"].dropna().unique():
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

    if not events:
        return pd.DataFrame(columns=[
            "Nemo", "Fecha", "Semaforo", "Senal", "Nota",
            "Gap", "Delta_Gap", "Flujo_AFP", "CompraVenta_Fuerte",
            "Prob_Compra_AFP_ProxMes"
        ])

    return pd.DataFrame(events).sort_values(["Nemo", "Fecha"]).reset_index(drop=True)


# =========================================================
# Apéndice
# =========================================================
def build_appendix_tables() -> dict:
    reglas = pd.DataFrame([
        ["Compra fuerte", "Delta Gap muy positivo dentro de la historia del papel y Gap sobre su media corta.", "Percentil Delta >= 85% y Gap > media 3 meses."],
        ["Venta fuerte", "Delta Gap muy negativo dentro de la historia del papel y Gap bajo su media corta.", "Percentil Delta <= 15% y Gap < media 3 meses."],
        ["Entrada activas", "Las AFP activas lideran compras del mes.", "Delta Gap > 0 y fuerza actual >= mes previo, o compra fuerte."],
        ["Entrada seguidoras", "Las seguidoras acompañan con rezago T+1 luego de una entrada líder.", "Delta Gap > 0, menor fuerza marginal y antecedente fuerte previo."],
        ["Salida activas", "Las AFP activas lideran ventas del mes.", "Delta Gap < 0 y fuerza actual <= mes previo, o venta fuerte."],
        ["Salida seguidoras", "Las seguidoras acompañan con rezago la salida del mes previo.", "Delta Gap < 0, caída menos intensa y antecedente vendedor previo."],
        ["Comprando", "Cambio mensual positivo, sin señal fuerte.", "Delta Gap > 0."],
        ["Vendiendo", "Cambio mensual negativo, sin señal fuerte.", "Delta Gap < 0."],
        ["Manteniendo", "Sin cambio relevante o sin señal dominante.", "Caso residual."],
    ], columns=["Señal", "Qué significa", "Cómo se calcula"])

    prob = pd.DataFrame([
        ["Probabilidad de compra AFP próximo mes", "Modelo logístico que estima la probabilidad de que el Delta Gap del próximo mes sea positivo.", "Gap actual, Delta Gap actual, medias móviles, desvío, percentiles, persistencia y rezagos por papel."],
        ["Rango 0% - 100%", "0% indica muy baja probabilidad de aumento del Gap próximo mes; 100% indica muy alta probabilidad.", "Se muestra como P_Up_next x 100."],
        ["Probabilidad de entrada AFP", "Score simple orientado a detectar compras probables.", "Delta Gap positivo + Gap sobre media 3M + FlowScore alto + señal de entrada."],
        ["Probabilidad de salida AFP", "Score simple orientado a detectar ventas probables.", "Delta Gap negativo + Gap bajo media 3M + riesgo de salida + señal de salida."],
        ["Probabilidad de seguimiento T+1", "Mide si un papel con liderazgo activo en T tiene condiciones para recibir seguimiento al mes siguiente.", "Liderazgo previo + Delta positivo + probabilidad modelo + FlowScore."],
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
        ["Prob. seguimiento T+1", "Chance de que las seguidoras acompañen el movimiento.", "Más alto = mejor setup líder-seguidor."],
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
    snap_last["Movimiento"] = snap_last["Delta_Gap"].apply(_class_movimiento)
    snap_last["Posicion"] = snap_last["Gap"].apply(_class_posicion)

    ranking_entrada = snap_last.sort_values(
        ["Prob_Entrada_AFP", "Prob_Compra_AFP_ProxMes", "Prob_Seguimiento_T1", "AFP_Flow_Rank", "Delta_Gap"],
        ascending=False
    ).copy()

    ranking_salida = snap_last.sort_values(
        ["Prob_Salida_AFP", "Delta_Gap", "Gap"],
        ascending=[False, True, True]
    ).copy()

    top_compras = snap_last.sort_values(
        ["Prob_Compra_AFP_ProxMes", "Prob_Seguimiento_T1", "AFP_Flow_Rank", "Delta_Gap"],
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
