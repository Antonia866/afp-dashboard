import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

SHEET_LISTA = "Hola Valores"
SHEET_DATA  = "valores para graficos"
SHEET_IPSA  = "IPSA"


# ----------------------------
# Excel utils
# ----------------------------
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


def _read_excel(xls_source, sheet_name: str, header=0) -> pd.DataFrame:
    return pd.read_excel(xls_source, sheet_name=sheet_name, engine="openpyxl", header=header)


# ----------------------------
# Última fecha: Hola Valores!I2
# ----------------------------
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


# ----------------------------
# Loaders
# ----------------------------
def load_universe_and_override_date(xls_source) -> Tuple[np.ndarray, Optional[pd.Timestamp], dict]:
    sheets = _excel_sheet_names(xls_source)

    sh_lista = _pick_sheet(
        sheets,
        SHEET_LISTA,
        aliases=["hola valores", "hola_valores", "universo", "universe"]
    )
    if sh_lista is None:
        raise ValueError(f"No encuentro la hoja '{SHEET_LISTA}'. Hojas disponibles: {sheets}")

    last_date_override = _find_last_date_override_from_I2(xls_source, sh_lista)

    hv = _read_excel(xls_source, sheet_name=sh_lista)

    if "Nemo" not in hv.columns:
        for cand in ["Ticker", "ticker", "TICKER"]:
            if cand in hv.columns:
                hv = hv.rename(columns={cand: "Nemo"})
                break

    for req in ["Nemo", "AFP", "IPSA"]:
        if req not in hv.columns:
            raise ValueError(f"En '{sh_lista}' falta columna '{req}'. Columnas: {list(hv.columns)}")

    hv["Nemo"] = hv["Nemo"].astype(str).str.upper().str.strip()
    hv["AFP"]  = hv["AFP"].astype(str).str.lower().str.strip()
    hv["IPSA"] = hv["IPSA"].astype(str).str.lower().str.strip()

    universo = hv.loc[(hv["AFP"] == "tiene") & (hv["IPSA"] == "tiene"), "Nemo"].dropna().unique()
    meta = {"sheet_lista": sh_lista, "sheets": sheets}
    return universo, last_date_override, meta


def load_data(xls_source, universo: np.ndarray, meta: dict) -> pd.DataFrame:
    sheets = meta["sheets"]
    sh_data = _pick_sheet(
        sheets,
        SHEET_DATA,
        aliases=["valores para graficos", "valores_para_graficos", "data", "datos", "valores para gráficos"]
    )
    if sh_data is None:
        raise ValueError(f"No encuentro la hoja '{SHEET_DATA}'. Hojas disponibles: {sheets}")

    df = _read_excel(xls_source, sheet_name=sh_data)

    if "Fecha" not in df.columns:
        for cand in ["fecha", "Date", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Fecha"})
                break

    if "Nemo" not in df.columns:
        for cand in ["Ticker", "ticker", "TICKER"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Nemo"})
                break

    if "GAP" not in df.columns:
        for c in df.columns:
            if "gap" in str(c).lower():
                df = df.rename(columns={c: "GAP"})
                break

    for req in ["Fecha", "Nemo", "GAP"]:
        if req not in df.columns:
            raise ValueError(f"En '{sh_data}' falta columna '{req}'. Columnas: {list(df.columns)}")

    # ✅ dayfirst + fin de mes
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True) + pd.offsets.MonthEnd(0)
    df["Nemo"]  = df["Nemo"].astype(str).str.upper().str.strip()
    df["GAP"]   = pd.to_numeric(df["GAP"], errors="coerce")

    df = df.dropna(subset=["Fecha", "Nemo", "GAP"])
    df = df[df["Nemo"].isin(universo)].sort_values(["Nemo", "Fecha"]).reset_index(drop=True)
    return df


# ----------------------------
# IPSA loader (ARREGLADO/ROBUSTO)
# ----------------------------
def _infer_date_col(df: pd.DataFrame) -> Optional[str]:
    # candidato por nombre
    for cand in ["Fecha", "fecha", "Date", "date", "FECHA"]:
        if cand in df.columns:
            return cand

    # inferencia: columna con mejor parseo a datetime
    best_col, best_ratio = None, 0.0
    for c in df.columns[:10]:  # no hace falta revisar 200 columnas
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        ratio = parsed.notna().mean()
        if ratio > best_ratio and ratio >= 0.5:
            best_col, best_ratio = c, ratio
    return best_col


def _infer_value_col(df: pd.DataFrame, date_col: str) -> Optional[str]:
    # candidato por nombre
    for cand in ["IPSA", "ipsa", "Close", "close", "Valor", "valor", "Index", "index", "Price", "price", "Nivel", "nivel"]:
        if cand in df.columns and cand != date_col:
            return cand

    # inferencia: columna numérica con más datos
    best_col, best_cnt = None, -1
    for c in df.columns:
        if c == date_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        cnt = int(s.notna().sum())
        if cnt > best_cnt and cnt >= 10:
            best_col, best_cnt = c, cnt
    return best_col


def load_ipsa_series(xls_source, meta: dict) -> Optional[pd.DataFrame]:
    sheets = meta["sheets"]

    # 1) intenta por nombre esperado/alias
    sh_ipsa = _pick_sheet(
        sheets,
        SHEET_IPSA,
        aliases=[
            "IPSA Hist", "IPSA HIST", "IPSA_hist", "IPSA HISTÓRICO", "IPSA historico",
            "Indice IPSA", "Índice IPSA", "indice", "índice", "index"
        ]
    )

    # 2) si no encontró, busca cualquier hoja que contenga "ipsa"
    if sh_ipsa is None:
        for s in sheets:
            if "ipsa" in s.lower():
                sh_ipsa = s
                break

    # 3) si existe hoja IPSA, la parsea robusto
    if sh_ipsa is not None:
        try:
            ips = _read_excel(xls_source, sheet_name=sh_ipsa)
            if ips is None or ips.empty:
                return None

            date_col = _infer_date_col(ips)
            if date_col is None:
                return None

            value_col = _infer_value_col(ips, date_col)
            if value_col is None:
                return None

            out = ips[[date_col, value_col]].copy()
            out.columns = ["Fecha", "IPSA"]
            out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce", dayfirst=True) + pd.offsets.MonthEnd(0)
            out["IPSA"]  = pd.to_numeric(out["IPSA"], errors="coerce")
            out = out.dropna(subset=["Fecha", "IPSA"]).sort_values("Fecha").reset_index(drop=True)
            if out.empty:
                return None
            return out
        except Exception:
            pass

    # 4) fallback: IPSA puede venir como columna en "valores para graficos"
    try:
        sh_data = _pick_sheet(
            sheets,
            SHEET_DATA,
            aliases=["valores para graficos", "valores_para_graficos", "data", "datos", "valores para gráficos"]
        )
        if sh_data is None:
            return None

        d = _read_excel(xls_source, sheet_name=sh_data)
        if "Fecha" not in d.columns:
            for cand in ["fecha", "Date", "date"]:
                if cand in d.columns:
                    d = d.rename(columns={cand: "Fecha"})
                    break
        if "Fecha" not in d.columns:
            return None

        # busca columna IPSA dentro de esta hoja
        val_col = None
        for c in d.columns:
            cl = str(c).lower()
            if "ipsa" == cl or "ipsa" in cl:
                val_col = c
                break
        if val_col is None:
            return None

        out = d[["Fecha", val_col]].copy()
        out.columns = ["Fecha", "IPSA"]
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce", dayfirst=True) + pd.offsets.MonthEnd(0)
        out["IPSA"]  = pd.to_numeric(out["IPSA"], errors="coerce")
        out = out.dropna(subset=["Fecha", "IPSA"]).groupby("Fecha", as_index=False)["IPSA"].mean()
        out = out.sort_values("Fecha").reset_index(drop=True)
        if out.empty:
            return None
        return out
    except Exception:
        return None


# ----------------------------
# Features / signals
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Nemo", group_keys=False)

    df["Delta_GAP"] = g["GAP"].diff()
    df["Aceleracion"] = g["Delta_GAP"].diff()

    df["MA_3"] = g["GAP"].apply(lambda s: s.rolling(3, min_periods=3).mean())
    df["MA_6"] = g["GAP"].apply(lambda s: s.rolling(6, min_periods=6).mean())
    df["STD_6"] = g["GAP"].apply(lambda s: s.rolling(6, min_periods=6).std())
    df["Z_6"] = (df["GAP"] - df["MA_6"]) / (df["STD_6"].replace(0, np.nan))

    df["GAP_Pctl"] = g["GAP"].apply(lambda s: s.rank(pct=True))

    for lag in [1, 2, 3]:
        df[f"GAP_lag{lag}"] = g["GAP"].shift(lag)
        df[f"Delta_lag{lag}"] = g["Delta_GAP"].shift(lag)
        df[f"Acc_lag{lag}"] = g["Aceleracion"].shift(lag)

    df["Delta_MA3"] = g["Delta_GAP"].apply(lambda s: s.rolling(3, min_periods=3).mean())
    df["Impulso"] = df["Delta_GAP"] - df["Delta_MA3"]

    df["Rank_GAP_mes"] = df.groupby("Fecha")["GAP"].rank(ascending=False, method="dense")
    df["Delta_Pctl"] = g["Delta_GAP"].apply(lambda s: s.rank(pct=True))

    df["Delta_next"] = g["Delta_GAP"].shift(-1)
    df["Up_next"] = (df["Delta_next"] > 0).astype(int)

    return df


def add_rules_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Nemo", group_keys=False)

    df["Score_AFPFlow"] = (
        1.5*(df["GAP"] > 0).astype(int) +
        2.0*(df["Delta_GAP"] > 0).astype(int) +
        1.5*(df["Aceleracion"] > 0).astype(int) +
        1.0*(df["Rank_GAP_mes"] <= 10).astype(int) +
        1.0*(df["Impulso"] > 0).astype(int)
    )

    def normalize_0_100(s):
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series([50]*len(s), index=s.index)
        return 100*(s - mn)/(mx - mn)

    df["FlowScore_0_100"] = df.groupby("Fecha")["Score_AFPFlow"].transform(normalize_0_100)

    def fase(row):
        gap, d, a = row["GAP"], row["Delta_GAP"], row["Aceleracion"]
        if pd.isna(gap) or pd.isna(d) or pd.isna(a):
            return "HOLD"
        if gap > 0 and d > 0 and a > 0:
            return "BUY (fuerte)"
        if gap > 0 and d > 0 and a < 0:
            return "BUY (débil)"
        if d < 0:
            return "SELL"
        return "HOLD"

    df["Fase"] = df.apply(fase, axis=1)

    def semaforo(row):
        if row["Fase"] in ["BUY (fuerte)"]:
            return "🟢"
        if row["Fase"] in ["SELL"]:
            return "🔴"
        return "🟡"

    df["Semaforo"] = df.apply(semaforo, axis=1)

    # Compra/Venta fuerte (institucional) usando percentiles de Delta
    df["Compra_Fuerte"] = (df["Delta_Pctl"] >= 0.85) & (df["Delta_GAP"] > 0) & (df["Aceleracion"] > 0) & (df["Impulso"] > 0)
    df["Venta_Fuerte"]  = (df["Delta_Pctl"] <= 0.15) & (df["Delta_GAP"] < 0) & (df["Aceleracion"] < 0) & (df["Impulso"] < 0)

    df["CompraVenta_Fuerte"] = "Neutral"
    df.loc[df["Compra_Fuerte"], "CompraVenta_Fuerte"] = "Compra fuerte"
    df.loc[df["Venta_Fuerte"],  "CompraVenta_Fuerte"] = "Venta fuerte"

    # Flujo AFP activas/seguidoras (heurística)
    df["Flujo_AFP"] = "Sin señal clara"
    df.loc[(df["Delta_GAP"] > 0) & (df["Aceleracion"] > 0), "Flujo_AFP"] = "Entrada activas"
    df.loc[(df["Delta_GAP"] > 0) & (df["Aceleracion"] < 0) & (g["Delta_Pctl"].shift(1) >= 0.75), "Flujo_AFP"] = "Entrada seguidoras"
    df.loc[(df["Delta_GAP"] < 0) & (df["Aceleracion"] < 0), "Flujo_AFP"] = "Salida activas"
    df.loc[(df["Delta_GAP"] < 0) & (df["Aceleracion"] > 0) & (g["Delta_GAP"].shift(1) < 0), "Flujo_AFP"] = "Salida seguidoras"

    return df


# ----------------------------
# Models
# ----------------------------
def train_predict_global(df_feat: pd.DataFrame):
    dfm = df_feat.copy()

    feature_cols_num = [
        "GAP", "Delta_GAP", "Aceleracion", "Impulso",
        "MA_3", "MA_6", "STD_6", "Z_6",
        "GAP_Pctl", "Rank_GAP_mes",
        "GAP_lag1", "GAP_lag2", "GAP_lag3",
        "Delta_lag1", "Delta_lag2", "Delta_lag3",
        "Acc_lag1", "Acc_lag2", "Acc_lag3"
    ]
    feature_cols_num = [c for c in feature_cols_num if c in dfm.columns]

    dfm = dfm.dropna(subset=feature_cols_num + ["Up_next", "Delta_next"])
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


def add_actions(df_model: pd.DataFrame) -> pd.DataFrame:
    dfm = df_model.copy()

    def tactical(r):
        p = r.get("P_Up_next", np.nan)
        fs = r.get("FlowScore_0_100", np.nan)
        sem = r.get("Semaforo", "🟡")

        if pd.notna(p) and pd.notna(fs):
            if (p >= 0.60) and (fs >= 70) and (sem == "🟢"):
                return "BUY"
            if (p >= 0.60) and (sem == "🟢"):
                return "BUY (light)"
            if (p <= 0.40) and (sem == "🔴"):
                return "SELL/REDUCE"
            if (p <= 0.40):
                return "REDUCE"
            return "HOLD"

        if sem == "🟢":
            return "BUY"
        if sem == "🔴":
            return "REDUCE"
        return "HOLD"

    def relative(r):
        fs = r.get("FlowScore_0_100", np.nan)
        p = r.get("P_Up_next", np.nan)
        sem = r.get("Semaforo", "🟡")

        if pd.isna(fs): fs = 50
        if pd.isna(p):  p = 0.50

        tilt = 2.0*(fs - 50) + 200*(p - 0.50)
        tilt = float(np.clip(tilt, -200, 200))

        if tilt >= 60 and sem == "🟢":
            return "OVERWEIGHT", tilt
        if tilt <= -60 and sem == "🔴":
            return "UNDERWEIGHT", tilt
        return "NEUTRAL", tilt

    def timing_trade(r):
        flujo = r.get("Flujo_AFP", "Sin señal clara")
        if flujo == "Entrada seguidoras":
            return "COMPRAR en T"
        if flujo == "Entrada activas":
            return "COMPRAR / MANTENER"
        if flujo == "Salida activas":
            return "VENDER / REDUCIR"
        if flujo == "Salida seguidoras":
            return "REDUCIR (light)"
        return "MANTENER"

    dfm["Accion_Tactica"] = dfm.apply(tactical, axis=1)
    out_r = dfm.apply(relative, axis=1, result_type="expand")
    dfm["Accion_Relativa"] = out_r[0]
    dfm["Tilt_bps"] = out_r[1].astype(float)
    dfm["Recomendacion_Timing"] = dfm.apply(timing_trade, axis=1)

    # ✅ para evitar KeyError siempre
    dfm["Prob_Compra_AFP_ProxMes"] = dfm["P_Up_next"]

    return dfm


def build_events(df_model: pd.DataFrame) -> pd.DataFrame:
    events = []
    for paper in df_model["Nemo"].unique():
        d = df_model[df_model["Nemo"] == paper].sort_values("Fecha").copy()
        prev = None
        for _, r in d.iterrows():
            fase = r.get("Fase", None)
            if pd.isna(fase):
                continue
            if prev is None or fase != prev:
                events.append({
                    "Nemo": paper,
                    "Fecha": r["Fecha"],
                    "Semaforo": r.get("Semaforo", ""),
                    "Fase": fase,
                    "Nota": f"Cambio de fase: {prev} → {fase}" if prev else f"Inicio: {fase}",
                    "GAP": r.get("GAP", np.nan),
                    "Delta_GAP": r.get("Delta_GAP", np.nan),
                    "Aceleracion": r.get("Aceleracion", np.nan),
                    "Impulso": r.get("Impulso", np.nan),
                    "CompraVenta_Fuerte": r.get("CompraVenta_Fuerte", "Neutral"),
                    "Prob_Compra_AFP_ProxMes": r.get("Prob_Compra_AFP_ProxMes", np.nan),
                })
                prev = fase

    return pd.DataFrame(events).sort_values(["Nemo", "Fecha"]).reset_index(drop=True)


def build_outputs(xls_source):
    universo, last_date_override, meta = load_universe_and_override_date(xls_source)

    df = load_data(xls_source, universo, meta)
    df = add_features(df)
    df = add_rules_signals(df)

    dfm, metrics = train_predict_global(df)
    dfm = add_actions(dfm)

    # ✅ IPSA robusto + fallback
    df_ipsa = load_ipsa_series(xls_source, meta)

    last_date = dfm["Fecha"].max()
    if last_date_override is not None:
        if (dfm["Fecha"] == last_date_override).any():
            last_date = last_date_override
        else:
            ym = (dfm["Fecha"].dt.year == last_date_override.year) & (dfm["Fecha"].dt.month == last_date_override.month)
            if ym.any():
                last_date = dfm.loc[ym, "Fecha"].max()

    snap_last = dfm[dfm["Fecha"] == last_date].copy()
    events = build_events(dfm)

    return df, dfm, snap_last, metrics, events, last_date, df_ipsa


  
