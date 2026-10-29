"""Extrae de los XBRL de la CMF (datos_monitores/eeff/<ENT>/*.zip) las cifras consolidadas trimestrales
de cada emisor (MM$) y aplica el mapeo validado (eeff_mapeo.json) a las métricas del Monitor de Retail.

Salida: datos_monitores/series_eeff.json
  {"generado", "fuente", "raw": {ENT: {tag: {AAAA-Qn: valor}}}, "series": {ENT: {metric: {AAAA-Qn: valor}}}, "nombres": {ENT: razón social}}
Trimestre: contexto TrimestreActual / TrimestreMovilActual; si falta (p. ej. cierre anual), acumulado del año menos acumulado del trimestre anterior.
"""
import glob, json, os, re, zipfile
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
FLUJOS = ["ifrs-full:Revenue", "ifrs-full:CostOfSales", "ifrs-full:GrossProfit", "ifrs-full:ProfitLoss",
          "ifrs-full:ProfitLossAttributableToOwnersOfParent", "ifrs-full:ProfitLossFromOperatingActivities",
          "ifrs-full:DepreciationAndAmortisationExpense", "ifrs-full:DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss",
          "ifrs-full:DistributionCosts", "ifrs-full:AdministrativeExpense", "ifrs-full:FinanceCosts", "ifrs-full:FinanceIncome",
          "ifrs-full:ProfitLossBeforeTax", "ifrs-full:IncomeTaxExpenseContinuingOperations"]
SALDOS = ["ifrs-full:Assets", "ifrs-full:Equity", "ifrs-full:EquityAttributableToOwnersOfParent", "ifrs-full:CashAndCashEquivalents",
          "ifrs-full:OtherCurrentFinancialLiabilities", "ifrs-full:OtherNoncurrentFinancialLiabilities",
          "ifrs-full:CurrentLeaseLiabilities", "ifrs-full:NoncurrentLeaseLiabilities", "ifrs-full:OtherCurrentFinancialAssets",
          "ifrs-full:InvestmentProperty", "ifrs-full:Inventories"]

def leer(path):
    z = zipfile.ZipFile(path)
    x = z.read([n for n in z.namelist() if n.endswith(".xbrl")][0]).decode("utf-8", "replace")
    nombre = (re.findall(r"NameOfReportingEntity[^>]*>([^<]+)<", x) or [""])[0]
    out = {"q": {}, "ytd": {}, "saldo": {}}
    for tag, ctx, v in re.findall(r'<([\w-]+:\w+) [^>]*contextRef="(TrimestreActual|TrimestreMovilActual|TrimestreAcumuladoActual|CierreTrimestreActual)"[^>]*>([^<]+)<', x):
        try: val = float(v) / 1e6
        except ValueError: continue
        k = {"TrimestreActual": "q", "TrimestreMovilActual": "q", "TrimestreAcumuladoActual": "ytd", "CierreTrimestreActual": "saldo"}[ctx]
        if tag in (FLUJOS if k != "saldo" else SALDOS): out[k].setdefault(tag, val)
    return nombre, out

def main():
    raw, nombres = {}, {}
    for d in sorted(glob.glob(os.path.join(BASE, "eeff", "*"))):
        ent = os.path.basename(d); per = {}
        for f in sorted(glob.glob(os.path.join(d, "*_C.zip"))):
            m = re.search(r"_(\d{4})(\d{2})_C\.zip$", f)
            try: nombre, data = leer(f)
            except Exception: continue
            nombres[ent] = nombre or nombres.get(ent, "")
            per[f"{m.group(1)}-Q{int(m.group(2)) // 3}"] = data
        r = raw.setdefault(ent, {})
        for p in sorted(per):
            y, q = int(p[:4]), int(p[-1])
            prevp = f"{y}-Q{q-1}" if q > 1 else None
            for tag in FLUJOS:
                v = per[p]["q"].get(tag)
                if v is None and tag in per[p]["ytd"]:
                    if q == 1: v = per[p]["ytd"][tag]
                    elif prevp in per and tag in per[prevp]["ytd"]: v = per[p]["ytd"][tag] - per[prevp]["ytd"][tag]
                if v is not None: r.setdefault(tag, {})[p] = round(v, 3)
                if tag in per[p]["ytd"]: r.setdefault(tag + "#acum", {})[p] = round(per[p]["ytd"][tag], 3)
            for tag in SALDOS:
                if tag in per[p]["saldo"]: r.setdefault(tag, {})[p] = round(per[p]["saldo"][tag], 3)
    mapeo = json.load(open(os.path.join(BASE, "eeff_mapeo.json"), encoding="utf-8")) if os.path.exists(os.path.join(BASE, "eeff_mapeo.json")) else {}
    series = {}
    for ent, mets in mapeo.get("mapeo", {}).items():
        for met, formula in mets.items():
            partes = re.findall(r"([+-]?)\s*([\w-]+:\w+(?:#acum)?)", formula)
            pers = set.intersection(*[set(raw.get(ent, {}).get(t, {})) for _, t in partes]) if partes else set()
            for p in pers:
                v = sum((-1 if s == "-" else 1) * raw[ent][t][p] for s, t in partes)
                series.setdefault(ent, {}).setdefault(met, {})[p] = round(v, 3)
            if met == "ganancia_bruta" and "ingresos" in mets:  # margen bruto derivado
                pass
        sm = series.get(ent, {})
        if "ganancia_bruta" in sm and "ingresos" in sm:
            sm["mg_bruto"] = {p: round(sm["ganancia_bruta"][p] / sm["ingresos"][p] * 100, 3) for p in sm["ganancia_bruta"] if p in sm["ingresos"] and sm["ingresos"][p]}
    out = {"generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fuente": "CMF, estados financieros IFRS consolidados (XBRL) de cada emisor", "nombres": nombres, "raw": raw, "series": series}
    with open(os.path.join(BASE, "series_eeff.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(nombres); print({e: {m: max(v) for m, v in s.items()} for e, s in series.items()})

if __name__ == "__main__":
    main()
