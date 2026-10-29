"""Convierte los Reportes Mensuales de la CMF (datos_monitores/cmf/*.xlsx) en las series
mensuales del Monitor de Bancos (métricas con sufijo _m), con las mismas definiciones
que la base histórica (validadas contra el archivo Sistema_bancario_Chile.xlsx).

Salida: datos_monitores/series_cmf.json  {"generado":..., "series": {ENT: {metric: {AAAA-MM: valor}}}, "errores": [...]}
Solo procesa desde 2022-01 (plan de cuentas vigente).
"""
import glob, json, os, re, unicodedata
from datetime import datetime, timezone
import openpyxl

BASE = os.path.dirname(os.path.abspath(__file__))
BANCOS = {  # entidad del monitor -> patrones del nombre en la CMF
    "CHILE": [r"^banco de chile"], "SAN": [r"^banco santander"], "BCI": [r"^banco de cr.dito e inversi"],
    "ITAU": [r"ita."], "BICE": [r"^banco bice"], "SCOTIA": [r"^scotiabank"], "BFALA": [r"^banco falabella"],
    "SIS": [r"^sistema bancario"],
}
def norm(s):
    s = unicodedata.normalize("NFD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()

def sheet(wb, *keys):
    for ws in wb.worksheets:
        t = norm(ws.title)
        if all(k in t for k in keys): return ws
    raise KeyError("hoja " + " ".join(keys))

def table(ws):
    rows = list(ws.iter_rows(values_only=True))
    ncol = max(len(r) for r in rows)
    hdr = [" | ".join(norm(rows[i][j]) for i in range(4, 12) if i < len(rows) and j < len(rows[i]) and rows[i][j] not in (None, "")) for j in range(ncol)]
    ents = {}
    for r in rows:
        if r and len(r) > 1 and isinstance(r[1], str):
            n = norm(r[1])
            for e, pats in BANCOS.items():
                if e not in ents and any(re.search(p, n) for p in pats): ents[e] = r
    return hdr, ents

def col(hdr, *keys, start=2):
    for j in range(start, len(hdr)):
        if all(k in hdr[j] for k in keys): return j
    raise KeyError("columna " + " / ".join(keys))

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def leer(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    def put(e, m, v):
        if v is not None: out.setdefault(e, {})[m] = v
    # Activos
    h, E = table(sheet(wb, "activos bancos 2"))
    jc, jcons, jviv, jprov = col(h, "comerciales", "total"), col(h, "consumo", "total"), col(h, "vivienda", "total"), col(h, "provisiones", "total")
    jprovb = col(h, "adeudado por bancos", start=jprov + 1)
    for e, r in E.items():
        com, cons, viv = num(r[jc]), num(r[jcons]), num(r[jviv])
        put(e, "coloc_com_m", com); put(e, "coloc_cons_m", cons); put(e, "coloc_viv_m", viv)
        if None not in (com, cons, viv): put(e, "coloc_m", com + cons + viv)
        p, pb = num(r[jprov]), num(r[jprovb]) or 0.0
        if p is not None: out.setdefault(e, {})["_prov_coloc"] = abs(p) - abs(pb)
    h, E = table(sheet(wb, "activos bancos 1"))
    ja, jdeu, jpac = col(h, "total activos"), col(h, "instrumentos financieros de deuda"), col(h, "pactos de retroventa")
    for e, r in E.items():
        put(e, "activos_m", num(r[ja]))
        d = out.get(e, {})
        if "coloc_m" in d and num(r[jdeu]) is not None:
            put(e, "aie_m", d["coloc_m"] + num(r[jdeu]) + (num(r[jpac]) or 0.0))
    h, E = table(sheet(wb, "pasivos bancos 2"))
    jad = col(h, "provisiones adicionales")
    for e, r in E.items():
        d = out.get(e, {})
        if "_prov_coloc" in d: put(e, "prov_m", d.pop("_prov_coloc") + abs(num(r[jad]) or 0.0))
    h, E = table(sheet(wb, "pasivos bancos 1"))
    jd, jv, jp = col(h, "depositos totales", "total"), col(h, "a la vista", "total"), col(h, "a plazo")
    for e, r in E.items():
        put(e, "depositos_m", num(r[jd])); put(e, "dep_vista_m", num(r[jv])); put(e, "dep_plazo_m", num(r[jp]))
    h, E = table(sheet(wb, "patrimonio"))
    jpat = col(h, "propietarios")
    for e, r in E.items(): put(e, "patrimonio_m", num(r[jpat]))
    # Resultados acumulados del año
    h, E = table(sheet(wb, "est. del resultado bancos"))
    jmir, jcom, jgp, jut = col(h, "margen de intereses"), col(h, "comisiones"), col(h, "gasto por perdidas crediticias"), col(h, "resultado del ejercicio")
    for e, r in E.items():
        put(e, "_mir_acum", num(r[jmir])); put(e, "_com_acum", num(r[jcom])); put(e, "_gp_acum", num(r[jgp])); put(e, "utilidad_acum_m", num(r[jut]))
    # Indicadores
    h, E = table(sheet(wb, "rentab"))
    jroa, jroae, jef = col(h, "(roa)"), col(h, "(roae)"), col(h, "eficiencia", "ingresos operacionales")
    for e, r in E.items():
        put(e, "roa_m", num(r[jroa])); put(e, "roae_m", num(r[jroae])); put(e, "efic_m", num(r[jef]))
    h, E = table(sheet(wb, "mora y deteriorada"))
    jn = col(h, "colocaciones a costo amortiz", "total"); jnc = col(h, "comerciales"); jncons = col(h, "consumo"); jnv = col(h, "vivienda")
    for e, r in E.items():
        put(e, "npl_m", num(r[jn])); put(e, "npl_com_m", num(r[jnc])); put(e, "npl_cons_m", num(r[jncons])); put(e, "npl_viv_m", num(r[jnv]))
    return out

def main():
    files = sorted(glob.glob(os.path.join(BASE, "cmf", "reporte_mensual_*.xlsx")))
    meses, errores = {}, []
    for f in files:
        per = re.search(r"(\d{4}-\d{2})", f).group(1)
        if per < "2022-01": continue
        try: meses[per] = leer(f)
        except Exception as ex: errores.append(f"{per}: {ex}")
    series = {}
    pers = sorted(meses)
    def S(e, m, p, v):
        if v is not None: series.setdefault(e, {}).setdefault(m, {})[p] = round(v, 6)
    for i, p in enumerate(pers):
        prev = pers[i - 1] if i and int(p[5:]) > 1 else None
        for e, d in meses[p].items():
            for m, v in d.items():
                if not m.startswith("_"): S(e, m, p, v)
            # flujos del mes = acumulado del año - acumulado del mes anterior (enero = acumulado)
            for acc, m in (("_mir_acum", "ing_int_m"), ("_com_acum", "comisiones_m"), ("_gp_acum", "gasto_prov_m"), ("utilidad_acum_m", "utilidad_m")):
                if acc not in d: continue
                if int(p[5:]) == 1: v = d[acc]
                elif prev and acc in meses[prev].get(e, {}): v = d[acc] - meses[prev][e][acc]
                else: continue
                S(e, m, p, -v if m == "gasto_prov_m" else v)  # gasto: positivo = gasto neto; negativo = liberación
            cm = d.get("coloc_m"); aie = d.get("aie_m")
            ii = series.get(e, {}).get("ing_int_m", {}).get(p); gp = series.get(e, {}).get("gasto_prov_m", {}).get(p)
            if ii is not None and aie: S(e, "nim_m", p, ii * 12 / aie * 100)
            if gp is not None and cm: S(e, "cor_m", p, gp * 12 / cm * 100)
            if d.get("prov_m") and d.get("npl_m") and cm: S(e, "cobertura_m", p, d["prov_m"] / (d["npl_m"] / 100 * cm) * 100)
        # Nota: las participaciones de mercado (ms_*) y las series *_cl_m del monitor usan saldos solo en Chile,
        # que este reporte consolidado no trae; no se generan aquí.
    # últimos 12 meses
    for e, ms in series.items():
        for flow, stock, m in (("ing_int_m", "aie_m", "nim_ltm"), ("gasto_prov_m", "coloc_m", "cor_ltm")):
            f, s = ms.get(flow, {}), ms.get(stock, {})
            for i, p in enumerate(pers):
                win = pers[max(0, i - 11): i + 1]
                if len(win) == 12 and all(q in f and q in s for q in win):
                    S(e, m, p, sum(f[q] for q in win) / (sum(s[q] for q in win) / 12) * 100)
    out = {"generado": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fuente": "CMF, Reporte Mensual de Información Financiera del Sistema Bancario",
           "meses": pers, "series": series, "errores": errores}
    with open(os.path.join(BASE, "series_cmf.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(pers)} meses, {sum(len(v) for v in series.values())} series, errores: {errores[:5]}")

if __name__ == "__main__":
    main()
