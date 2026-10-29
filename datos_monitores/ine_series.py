"""Convierte los cuadros del INE (Índice de Actividad del Comercio, base 2018=100) en las series
de consumo del Monitor de Retail: variación a/a del IACM total y por línea de productos.

Salida: datos_monitores/series_ine.json {"generado","fuente","series":{ENT:{metric:{AAAA-MM:valor}}}}
"""
import glob, json, os, re
from datetime import datetime, timezone
import xlrd

BASE = os.path.dirname(os.path.abspath(__file__))
MES = {"ene":1,"feb":2,"mar":3,"abr":4,"may":5,"jun":6,"jul":7,"ago":8,"sep":9,"oct":10,"nov":11,"dic":12}
LINEAS = {  # columna del cuadro 5 -> métrica del monitor
    "Línea 4A": ("CONSUMO", "crec_cat@Alimentos y bebidas"),
    "Línea 6": ("CONSUMO", "crec_cat@Vestuario y calzado"),
    "Línea 7": ("CONSUMO", "crec_cat@Electrónica y hogar"),
    "Línea 8": ("CONSUMO", "crec_cat@Mejoramiento del hogar"),
    "Línea 9": ("CONSUMO", "crec_cat@Bienes de consumo diversos"),
    "Línea 2": ("CONSUMO", "crec_cat@Repuestos y accesorios de vehículos"),
}

def main():
    f = glob.glob(os.path.join(BASE, "ine", "Base promedio año 2018=100", "Series mensuales desde enero de 2018*"))[0]
    sh = xlrd.open_workbook(f).sheet_by_name("5")
    hdr = [str(c).strip() for c in sh.row_values(5)]
    idx = {}
    for r in range(6, sh.nrows):
        m = str(sh.cell_value(r, 1)).strip().lower()
        mm = re.match(r"([a-z]{3})-(\d{2})", m)
        if mm and mm.group(1) in MES:
            per = f"20{mm.group(2)}-{MES[mm.group(1)]:02d}"
            idx[per] = sh.row_values(r)
    cols = {"__total__": next(j for j, h in enumerate(hdr) if h.lower().startswith("índice de actividad del comercio"))}
    for j, h in enumerate(hdr):
        if h in LINEAS: cols[h] = j
    series = {}
    for per, row in idx.items():
        prev = f"{int(per[:4]) - 1}-{per[5:]}"
        if prev not in idx: continue
        for k, j in cols.items():
            try: v = (float(row[j]) / float(idx[prev][j]) - 1) * 100
            except (TypeError, ValueError, ZeroDivisionError): continue
            ent, met = ("CL", "ventas_minoristas_yoy") if k == "__total__" else LINEAS[k]
            series.setdefault(ent, {}).setdefault(met, {})[per] = round(v, 4)
    out = {"generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fuente": "INE, Índice de Actividad del Comercio al por Menor (IACM), cuadro 5, base 2018=100, precios constantes",
           "archivo": os.path.relpath(f, BASE), "series": series}
    with open(os.path.join(BASE, "series_ine.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print({e: {m: max(v) for m, v in ms.items()} for e, ms in series.items()})

if __name__ == "__main__":
    main()
