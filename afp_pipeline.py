import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

SHEET_LISTA = "Hola Valores"
SHEET_DATA  = "valores para graficos"


def load_universe(file_path: str) -> np.ndarray:
    hv = pd.read_excel(file_path, sheet_name=SHEET_LISTA)

    hv["Nemo"] = hv["Nemo"].astype(str).str.upper().str.strip()
    hv["AFP"]  = hv["AFP"].astype(str).str.lower().str.strip()
    hv["IPSA"] = hv["IPSA"].astype(str).str.lower().str.strip()

    universo = hv.loc[(hv["AFP"] == "tiene") & (hv["IPSA"] == "tiene"), "Nemo"].dropna().unique()
    return universo


def load_data(file_path: str, universo: np.ndarray) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=SHEET_DATA)

    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Nemo"]  = df["Nemo"].astype(str).str.upper().str.strip()
    df["GAP"]   = pd.to_numeric(df["GAP"], errors="coerce")

    df = df.dropna(subset=["Fecha", "Nemo", "GAP"])
    df = df[df["Nemo"].isin(universo)].sort_values(["Nemo", "Fecha"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Nemo", group_keys=False)

    # Core dynamics
    df["Delta_GAP"] = g["GAP"].diff()
    df["Aceleracion"] = g["Delta_GAP"].diff()

    # Rolling stats
    df["MA_3"] = g["GAP"].apply(lambda s: s.rolling(3, min_periods=3).mean())
    df["MA_6"] = g["GAP"].apply(lambda s: s.rolling(6, min_periods=6).mean())
    df["STD_6"] = g["GAP"].apply(lambda s: s.rolling(6, min_periods=6).std())
    df["Z_6"] = (df["GAP"] - df["MA_6"]) / (df["STD_6"].replace(0, np.nan))

    # Historical percentile within paper
    df["GAP_Pctl"] = g["GAP"].apply(lambda s: s.rank(pct=True))

    # Lags
    for lag in [1, 2, 3]:
        df[f"GAP_lag{lag}"] = g["GAP"].shift(lag)
        df[f"Delta_lag{lag}"] = g["Delta_GAP"].shift(lag)
        df[f"Acc_lag{lag}"] = g["Aceleracion"].shift(lag)

    # Momentum / impulse
    df["Delta_MA3"] = g["Delta_GAP"].apply(lambda s: s.rolling(3, min_periods=3).mean())
    df["Impulso"] = df["Delta_GAP"] - df["Delta_MA3"]

    # Cross-sectional ranks by month
    df["Rank_GAP_mes"] = df.groupby("Fecha")["GAP"].rank(ascending=False, method="dense")

    # "Strong expansion" within paper (used for the old T+1 idea; kept as feature)
    df["Delta_Pctl"] = g["Delta_GAP"].apply(lambda s: s.rank(pct=True))
    df["Fuerte_Expansion"] = df["Delta_Pctl"] >= 0.75

    # Targets (next month)
    df["Delta_next"] = g["Delta_GAP"].shift(-1)
    df["Up_next"] = (df["Delta_next"] > 0).astype(int)

    return df


def add_rules_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("Nemo", group_keys=False)

    # Rule-based signals (still useful as flags)
    df["Acumulacion"] = (df["GAP"] > 0) & (df["Delta_GAP"] > 0) & (df["Aceleracion"] > 0)
    df["Arrastre_T1"] = (g["Fuerte_Expansion"].shift(1) == True) & (df["Delta_GAP"] > 0) & (df["Aceleracion"] < 0)
    df["Saturacion"] = (df["GAP_Pctl"] >= 0.85) & (df["Delta_GAP"] < 0) & (df["Aceleracion"] < 0)

    # Scores (raw)
    df["Score_AFPFlow"] = (
        1.5*(df["GAP"] > 0).astype(int) +
        2.0*(df["Delta_GAP"] > 0).astype(int) +
        1.5*(df["Aceleracion"] > 0).astype(int) +
        1.0*(df["Rank_GAP_mes"] <= 10).astype(int) +
        1.0*(df["Impulso"] > 0).astype(int)
    )

    df["Score_SatRisk"] = (
        2.0*(df["GAP_Pctl"] >= 0.85).astype(int) +
        1.5*(df["Z_6"] >= 1.0).astype(int) +
        2.0*(df["Delta_GAP"] < 0).astype(int) +
        1.5*(df["Aceleracion"] < 0).astype(int)
    )

    # =========================
    # Profesional: Estado largos/cortos (reemplaza "fase")
    # =========================
    def classify(row):
        gap = row.get("GAP", np.nan)
        dgap = row.get("Delta_GAP", np.nan)
        acc = row.get("Aceleracion", np.nan)

        # Si faltan datos, neutral
        if pd.isna(gap) or pd.isna(dgap) or pd.isna(acc):
            return "NEUTRAL"

        # Largos (sobreponderación)
        if gap > 0:
            if dgap > 0 and acc > 0:
                return "LARGOS ACELERANDO"
            elif dgap > 0 and acc < 0:
                return "LARGOS MODERANDO"
            elif dgap < 0:
                return "DESCARGANDO LARGOS"
            else:
                return "LARGOS ESTABLE"

        # Cortos (infraponderación)
        if gap < 0:
            if dgap < 0 and acc < 0:
                return "CORTOS ACELERANDO"
            elif dgap > 0:
                return "CUBRIENDO CORTOS"
            else:
                return "CORTOS ESTABLE"

        return "NEUTRAL"

    df["Fase"] = df.apply(classify, axis=1)

    # =========================
    # Score 0–100 (intensidad) por mes
    # =========================
    def normalize_0_100(s):
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series([50]*len(s), index=s.index)
        return 100*(s - mn)/(mx - mn)

    df["FlowScore_0_100"] = df.groupby("Fecha")["Score_AFPFlow"].transform(normalize_0_100)

    # =========================
    # Semáforo
    # =========================
    def traffic_light(estado):
        if estado in ["LARGOS ACELERANDO", "CUBRIENDO CORTOS"]:
            return "🟢"
        if estado in ["DESCARGANDO LARGOS", "CORTOS ACELERANDO"]:
            return "🔴"
        return "🟡"

    df["Semaforo"] = df["Fase"].apply(traffic_light)

    return df


def train_predict_global(df_feat: pd.DataFrame):
    """
    Global model:
      - Clasificación: P(ΔGAP próximo mes > 0)
      - Regresión: ΔGAP esperado próximo mes
    """
    dfm = df_feat.copy()

    feature_cols_num = [
        "GAP", "Delta_GAP", "Aceleracion", "MA_3", "MA_6", "STD_6", "Z_6",
        "GAP_Pctl", "Impulso", "Rank_GAP_mes",
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

    clf = Pipeline(steps=[("pre", pre), ("model", LogisticRegression(max_iter=700))])
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

    # In-sample probabilities/estimates (useful for charts; backtest tab uses Up_next directly)
    dfm["P_Up_next"] = clf.predict_proba(X)[:, 1]
    dfm["Delta_next_hat"] = reg.predict(X)

    metrics = {
        "AUC_mean": float(np.nanmean(aucs)) if len(aucs) else np.nan,
        "ACC_mean": float(np.mean(accs)) if len(accs) else np.nan,
        "rows": int(len(dfm))
    }

    return dfm, metrics


def build_events(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Eventos cuando cambia el estado (Fase) + nota ejecutiva.
    """
    events = []

    for paper in df_model["Nemo"].unique():
        d = df_model[df_model["Nemo"] == paper].sort_values("Fecha").copy()

        prev = None
        for _, r in d.iterrows():
            fase = r.get("Fase", None)
            if pd.isna(fase):
                continue

            if prev is None or fase != prev:
                sem = r.get("Semaforo", "")
                gap = r.get("GAP", np.nan)
                dgap = r.get("Delta_GAP", np.nan)
                acc = r.get("Aceleracion", np.nan)
                p_up = r.get("P_Up_next", np.nan)
                score = r.get("FlowScore_0_100", np.nan)

                if fase == "LARGOS ACELERANDO":
                    nota = "Entrada fuerte AFP: sobreponderación aumenta y acelera (flujo comprador)."
                elif fase == "LARGOS MODERANDO":
                    nota = "Sigue expansión, pero con menor impulso (posible arrastre/seguidores)."
                elif fase == "DESCARGANDO LARGOS":
                    nota = "Reducción de sobreponderación (rotación / toma de utilidad)."
                elif fase == "CORTOS ACELERANDO":
                    nota = "Aumenta infraponderación (flujo negativo estructural)."
                elif fase == "CUBRIENDO CORTOS":
                    nota = "Cubre infraponderación (potencial reversión positiva)."
                elif fase == "LARGOS ESTABLE":
                    nota = "Posición larga estable (sin aceleración clara)."
                elif fase == "CORTOS ESTABLE":
                    nota = "Posición corta estable (sin cambios relevantes)."
                else:
                    nota = "Fase neutral / transición."

                events.append({
                    "Nemo": paper,
                    "Fecha": r["Fecha"],
                    "Semaforo": sem,
                    "Fase": fase,
                    "Nota": nota,
                    "GAP": gap,
                    "Delta_GAP": dgap,
                    "Aceleracion": acc,
                    "FlowScore_0_100": score,
                    "P_Up_next": p_up
                })

                prev = fase

    ev = pd.DataFrame(events).sort_values(["Nemo", "Fecha"]).reset_index(drop=True)
    return ev


def build_outputs(file_path: str):
    universo = load_universe(file_path)
    df = load_data(file_path, universo)
    df = add_features(df)
    df = add_rules_signals(df)
    dfm, metrics = train_predict_global(df)

    last_date = dfm["Fecha"].max()
    snap = dfm[dfm["Fecha"] == last_date].copy()

    snap["Rank_Oportunidad"] = snap["Score_AFPFlow"].rank(ascending=False, method="dense")
    snap["Rank_RiesgoSat"] = snap["Score_SatRisk"].rank(ascending=False, method="dense")

    events = build_events(dfm)
    return df, dfm, snap, metrics, events