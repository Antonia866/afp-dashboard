import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

SHEET_HOLA = "Hola Valores"


# =========================================================
# LECTURA BASE: SIEMPRE DESDE "Hola Valores"
# =========================================================
def _read_excel(xls_source, sheet_name: str, header=0) -> pd.DataFrame:
    return pd.read_excel(xls_source, sheet_name=sheet_name, engine="openpyxl", header=header)


def load_hola_valores(xls_source) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Usa siempre la hoja Hola Valores.
    Estructura esperada:
    A codigo
    B fecha
    C nemo
    D peso cartera AFP
    E peso IPSA
    F gap
    G AFP tiene/no
    H IPSA tiene/no
    I ultima fecha
    J primera fecha
    """
    hv = _read_excel(xls_source, SHEET_HOLA)

    # Leer última fecha estrictamente desde I2
    raw = _read_excel(xls_source, SHEET_HOLA, header=None)
    v_last = raw.iat[1, 8]   # I2
    last_date = pd.to_datetime(v_last, errors="coerce", dayfirst=True)
    if pd.isna(last_date):
        raise ValueError("No pude leer la última fecha en Hola Valores!I2")
    last_date = pd.Timestamp(last_date) + pd.offsets.MonthEnd(0)

    # Renombrar por posición para no depender del header exacto
    cols = list(hv.columns)
    if len(cols) >= 10:
        hv = hv.rename(columns={
            cols[0]: "Codigo",
            cols[1]: "Fecha",
            cols[2]: "Nemo",
            cols[3]: "Peso_Cartera_AFP",
            cols[4]: "Peso_IPSA",
            cols[5]: "GAP",
            cols[6]: "AFP_Tiene",
            cols[7]: "IPSA_Tiene",
            cols[8]: "Ultima_Fecha",
            cols[9]: "Primera_Fecha",
        })

    required = ["Fecha", "Nemo", "Peso_Cartera_AFP", "Peso_IPSA", "GAP", "AFP_Tiene", "IPSA_Tiene"]
    for c in required:
        if c not in hv.columns:
            raise ValueError(f"Falta la columna '{c}' en Hola Valores. Columnas detectadas: {list(hv.columns)}")

    hv["Fecha"] = pd.to_datetime(hv["Fecha"], errors="coerce", dayfirst=True) + pd.offsets.MonthEnd(0)
    hv["Nemo"] = hv["Nemo"].astype(str).str.upper().str.strip()
    hv["Peso_Cartera_AFP"] = pd.to_numeric(hv["Peso_Cartera_AFP"], errors="coerce")
    hv["Peso_IPSA"] = pd.to_numeric(hv["Peso_IPSA"], errors="coerce")
    hv["GAP"] = pd.to_numeric(hv["GAP"], errors="coerce")
    hv["AFP_Tiene"] = hv["AFP_Tiene"].astype(str).str.lower().str.strip()
    hv["IPSA_Tiene"] = hv["IPSA_Tiene"].astype(str).str.lower().str.strip()

    hv = hv.dropna(subset=["Fecha", "Nemo", "GAP"])

    # Universo: solo papeles que están en ambos
    hv = hv[(hv["AFP_Tiene"] == "tiene") & (hv["IPSA_Tiene"] == "tiene")].copy()

    return hv, last_date


# =========================================================
# FEATURES
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Nemo", "Fecha"])
    g = df.groupby("Nemo", group_keys=False)

    # Flujo mensual
    df["Delta_GAP"] = g["GAP"].diff()

    # Cambio del flujo (interno, no visible)
    df["Cambio_Flujo"] = g["Delta_GAP"].diff()

    # Fuerza del flujo (interno, no visible)
    df["Delta_GAP_MA3"] = g["Delta_GAP"].apply(lambda s: s.rolling(3, min_periods=3).mean())
    df["Fuerza_Flujo"] = df["Delta_GAP"] - df["Delta_GAP_MA3"]

    # Percentiles por papel
    df["GAP_Pctl"] = g["GAP"].apply(lambda s: s.rank(pct=True))
    df["Delta_Pctl"] = g["Delta_GAP"].apply(lambda s: s.rank(pct=True))

    # Medias / dispersión para contexto
    df["GAP_MA3"] = g["GAP"].apply(lambda s: s.rolling(3, min_periods=3).mean())
    df["GAP_MA6"] = g["GAP"].apply(lambda s: s.rolling(6, min_periods=6).mean())
    df["GAP_STD6"] = g["GAP"].apply(lambda s: s.rolling(6, min_periods=6).std())
    df["GAP_Z6"] = (df["GAP"] - df["GAP_MA6"]) / df["GAP_STD6"].replace(0, np.nan)

    # Lags para modelo
    for lag in [1, 2, 3]:
        df[f"GAP_lag{lag}"] = g["GAP"].shift(lag)
        df[f"Delta_lag{lag}"] = g["Delta_GAP"].shift(lag)
        df[f"CambioFlujo_lag{lag}"] = g["Cambio_Flujo"].shift(lag)
        df[f"FuerzaFlujo_lag{lag}"] = g["Fuerza_Flujo"].shift(lag)

    # Target siguiente mes
    df["Delta_next"] = g["Delta_GAP"].shift(-1)
    df["Up_next"] = (df["Delta_next"] > 0).astype(int)

    return df


# =========================================================
# ETIQUETAS INTUITIVAS
# =========================================================
def classify_fase(row) -> str:
    gap = row.get("GAP", np.nan)
    delta = row.get("Delta_GAP", np.nan)

    if pd.isna(gap) or pd.isna(delta):
        return "Manteniendo"

    if gap > 0 and delta > 0:
        return "Largo comprando"
    if gap > 0 and delta < 0:
        return "Largo vendiendo"
    if gap < 0 and delta < 0:
        return "Corto aumentando"
    if gap < 0 and delta > 0:
        return "Corto cubriendo"
    return "Manteniendo"


def classify_flujo(row) -> str:
    delta = row.get("Delta_GAP", np.nan)
    cambio = row.get("Cambio_Flujo", np.nan)
    delta_pctl = row.get("Delta_Pctl", np.nan)

    if pd.isna(delta) or pd.isna(cambio):
        return "Sin señal clara"

    if delta > 0 and cambio > 0:
        return "Entrada activas"
    if delta > 0 and cambio < 0 and pd.notna(delta_pctl) and delta_pctl >= 0.50:
        return "Entrada seguidoras"
    if delta < 0 and cambio < 0:
        return "Salida activas"
    if delta < 0 and cambio > 0:
        return "Salida seguidoras"
    return "Sin señal clara"


def classify_compra_venta_fuerte(row) -> str:
    delta = row.get("Delta_GAP", np.nan)
    cambio = row.get("Cambio_Flujo", np.nan)
    fuerza = row.get("Fuerza_Flujo", np.nan)
    delta_pctl = row.get("Delta_Pctl", np.nan)

    if pd.isna(delta) or pd.isna(cambio) or pd.isna(fuerza):
        return "Neutral"

    if delta > 0 and cambio > 0 and fuerza > 0 and pd.notna(delta_pctl) and delta_pctl >= 0.85:
        return "Compra fuerte"
    if delta < 0 and cambio < 0 and fuerza < 0 and pd.notna(delta_pctl) and delta_pctl <= 0.15:
        return "Venta fuerte"
    return "Neutral"


def add_intuitive_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Fase"] = df.apply(classify_fase, axis=1)
    df["Flujo_AFP"] = df.apply(classify_flujo, axis=1)
    df["CompraVenta_Fuerte"] = df.apply(classify_compra_venta_fuerte, axis=1)

    def semaforo(row):
        fuerte = row["CompraVenta_Fuerte"]
        fase = row["Fase"]

        if fuerte == "Compra fuerte":
            return "🟢"
        if fuerte == "Venta fuerte":
            return "🔴"
        if fase in ["Largo comprando", "Corto cubriendo"]:
            return "🟢"
        if fase in ["Largo vendiendo", "Corto aumentando"]:
            return "🔴"
        return "🟡"

    df["Semaforo"] = df.apply(semaforo, axis=1)

    return df


# =========================================================
# MODELO
# =========================================================
def train_predict_global(df_feat: pd.DataFrame):
    dfm = df_feat.copy()

    feature_cols_num = [
        "GAP", "Delta_GAP", "Cambio_Flujo", "Fuerza_Flujo",
        "GAP_MA3", "GAP_MA6", "GAP_STD6", "GAP_Z6",
        "GAP_Pctl", "Delta_Pctl",
        "GAP_lag1", "GAP_lag2", "GAP_lag3",
        "Delta_lag1", "Delta_lag2", "Delta_lag3",
        "CambioFlujo_lag1", "CambioFlujo_lag2", "CambioFlujo_lag3",
        "FuerzaFlujo_lag1", "FuerzaFlujo_lag2", "FuerzaFlujo_lag3"
    ]
    feature_cols_num = [c for c in feature_cols_num if c in dfm.columns]

    # -----------------------------------------------------
    # FIX:
    # 1) df_train: solo filas con target disponible para entrenar
    # 2) df_pred: filas con features suficientes para predecir
    # De esta forma la última fecha no se pierde aunque no tenga Delta_next
    # -----------------------------------------------------
    df_train = dfm.dropna(subset=feature_cols_num + ["Up_next", "Delta_next"]).copy()
    df_train = df_train.sort_values(["Fecha", "Nemo"]).reset_index(drop=True)

    df_pred = dfm.dropna(subset=feature_cols_num).copy()
    df_pred = df_pred.sort_values(["Fecha", "Nemo"]).reset_index(drop=True)

    if df_train.empty:
        metrics = {
            "AUC_mean": np.nan,
            "ACC_mean": np.nan,
            "rows": 0
        }
        dfm["P_Up_next"] = np.nan
        dfm["Delta_next_hat"] = np.nan
        return dfm, metrics

    X_train = df_train[["Nemo"] + feature_cols_num]
    y_cls = df_train["Up_next"].astype(int)
    y_reg = df_train["Delta_next"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Nemo"]),
            ("num", "passthrough", feature_cols_num),
        ],
        remainder="drop"
    )

    clf = Pipeline(steps=[("pre", pre), ("model", LogisticRegression(max_iter=1000))])
    reg = Pipeline(steps=[("pre", pre), ("model", Ridge(alpha=1.0))])

    # Validación temporal solo sobre dataset de entrenamiento
    aucs, accs = [], []
    n_splits = min(5, len(df_train) - 1)

    if n_splits >= 2:
        tss = TimeSeriesSplit(n_splits=n_splits)
        for tr_idx, te_idx in tss.split(X_train, y_cls):
            y_te = y_cls.iloc[te_idx]
            if y_te.nunique() < 2:
                continue

            clf.fit(X_train.iloc[tr_idx], y_cls.iloc[tr_idx])
            proba = clf.predict_proba(X_train.iloc[te_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)

            try:
                aucs.append(roc_auc_score(y_te, proba))
            except Exception:
                pass

            try:
                accs.append(accuracy_score(y_te, pred))
            except Exception:
                pass

    # Fit final con todo el train
    clf.fit(X_train, y_cls)
    reg.fit(X_train, y_reg)

    # Predicción sobre TODAS las filas predecibles, incluida última fecha
    if not df_pred.empty:
        X_pred = df_pred[["Nemo"] + feature_cols_num]
        df_pred["P_Up_next"] = clf.predict_proba(X_pred)[:, 1]
        df_pred["Delta_next_hat"] = reg.predict(X_pred)
    else:
        df_pred["P_Up_next"] = np.nan
        df_pred["Delta_next_hat"] = np.nan

    # Merge de predicciones al dataset completo
    dfm = dfm.merge(
        df_pred[["Nemo", "Fecha", "P_Up_next", "Delta_next_hat"]],
        on=["Nemo", "Fecha"],
        how="left"
    )

    metrics = {
        "AUC_mean": float(np.nanmean(aucs)) if len(aucs) else np.nan,
        "ACC_mean": float(np.mean(accs)) if len(accs) else np.nan,
        "rows": int(len(df_train))
    }
    return dfm, metrics


# =========================================================
# ACCIONES
# =========================================================
def add_actions(df_model: pd.DataFrame) -> pd.DataFrame:
    dfm = df_model.copy()

    raw_score = (
        35 * (dfm["Delta_GAP"] > 0).astype(int) +
        25 * (dfm["Cambio_Flujo"] > 0).astype(int) +
        20 * (dfm["Fuerza_Flujo"] > 0).astype(int) +
        20 * (dfm["GAP"] > 0).astype(int)
    )

    if raw_score.max() == raw_score.min():
        dfm["FlowScore_0_100"] = 50
    else:
        dfm["FlowScore_0_100"] = 100 * (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())

    dfm["Prob_Compra_AFP_ProxMes"] = dfm["P_Up_next"]

    def accion_tactica(row):
        p = row.get("Prob_Compra_AFP_ProxMes", np.nan)
        sem = row.get("Semaforo", "🟡")
        fuerte = row.get("CompraVenta_Fuerte", "Neutral")

        if fuerte == "Compra fuerte" and pd.notna(p) and p >= 0.60:
            return "BUY"
        if fuerte == "Venta fuerte" and pd.notna(p) and p <= 0.40:
            return "SELL/REDUCE"
        if sem == "🟢" and pd.notna(p) and p >= 0.60:
            return "BUY (light)"
        if sem == "🔴" and pd.notna(p) and p <= 0.40:
            return "REDUCE"
        return "HOLD"

    def accion_relativa(row):
        p = row.get("Prob_Compra_AFP_ProxMes", np.nan)
        fs = row.get("FlowScore_0_100", np.nan)

        if pd.isna(p):
            p = 0.50
        if pd.isna(fs):
            fs = 50

        tilt = 2.0 * (fs - 50) + 200 * (p - 0.50)
        tilt = float(np.clip(tilt, -200, 200))

        if tilt >= 60:
            return "OVERWEIGHT"
        if tilt <= -60:
            return "UNDERWEIGHT"
        return "NEUTRAL"

    def timing(row):
        flujo = row.get("Flujo_AFP", "Sin señal clara")
        if flujo == "Entrada seguidoras":
            return "COMPRAR en T"
        if flujo == "Entrada activas":
            return "COMPRAR / MANTENER"
        if flujo == "Salida activas":
            return "VENDER / REDUCIR"
        if flujo == "Salida seguidoras":
            return "REDUCIR (light)"
        return "MANTENER"

    dfm["Accion_Tactica"] = dfm.apply(accion_tactica, axis=1)
    dfm["Accion_Relativa"] = dfm.apply(accion_relativa, axis=1)
    dfm["Recomendacion_Timing"] = dfm.apply(timing, axis=1)

    return dfm


# =========================================================
# EVENTOS / TIMELINE
# =========================================================
def build_events(df_model: pd.DataFrame) -> pd.DataFrame:
    events = []

    for paper in sorted(df_model["Nemo"].unique()):
        d = df_model[df_model["Nemo"] == paper].sort_values("Fecha").copy()
        prev_sem = None
        prev_fase = None

        for _, r in d.iterrows():
            sem = r.get("Semaforo", "🟡")
            fase = r.get("Fase", "Manteniendo")

            if prev_sem is None or sem != prev_sem or fase != prev_fase:
                events.append({
                    "Nemo": paper,
                    "Fecha": r["Fecha"],
                    "Semaforo": sem,
                    "Fase": fase,
                    "Flujo_AFP": r.get("Flujo_AFP", ""),
                    "CompraVenta_Fuerte": r.get("CompraVenta_Fuerte", ""),
                    "GAP": r.get("GAP", np.nan),
                    "Delta_GAP": r.get("Delta_GAP", np.nan),
                    "Prob_Compra_AFP_ProxMes": r.get("Prob_Compra_AFP_ProxMes", np.nan),
                    "Nota": f"{fase} | {r.get('Flujo_AFP', '')} | {r.get('CompraVenta_Fuerte', '')}"
                })
                prev_sem = sem
                prev_fase = fase

    return pd.DataFrame(events).sort_values(["Nemo", "Fecha"]).reset_index(drop=True)


# =========================================================
# BUILD OUTPUTS
# =========================================================
def build_outputs(xls_source):
    hv, last_date = load_hola_valores(xls_source)

    df = add_features(hv)
    df = add_intuitive_labels(df)

    dfm, metrics = train_predict_global(df)
    dfm = add_intuitive_labels(dfm)
    dfm = add_actions(dfm)

    # Última fecha: SIEMPRE desde I2, pero ahora buscando en dfm completo
    # no solo en filas de entrenamiento
    if (dfm["Fecha"] == last_date).any():
        use_last_date = last_date
    else:
        ym = (dfm["Fecha"].dt.year == last_date.year) & (dfm["Fecha"].dt.month == last_date.month)
        if ym.any():
            use_last_date = dfm.loc[ym, "Fecha"].max()
        else:
            use_last_date = dfm["Fecha"].max()

    snap_last = dfm[dfm["Fecha"] == use_last_date].copy()
    events = build_events(dfm)

    return df, dfm, snap_last, metrics, events, use_last_date
