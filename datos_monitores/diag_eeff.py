"""Diagnóstico: cómo descargar estados financieros (XBRL) de emisores desde la CMF."""
import os, re, urllib.request, urllib.parse
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostico")
os.makedirs(OUT, exist_ok=True)
def req(url, data=None):
    r = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode() if data else None, headers={"User-Agent": UA})
    with urllib.request.urlopen(r, timeout=60) as f: return f.read()
log = []
base = "https://www.cmfchile.cl/institucional/mercados/entidad.php"
q = {"mercado": "V", "rut": "90749000", "grupo": "", "tipoentidad": "RVEMI", "vig": "VI", "control": "svs", "pestania": "3"}
for nombre, url, data in [
    ("get", base + "?" + urllib.parse.urlencode(q), None),
    ("post", base + "?" + urllib.parse.urlencode(q), {"aa": "2026", "mm": "06", "tipo": "C", "tipo_norma": "IFRS", "enviado": "1", "orden": "", "mm_ifrs": "06", "aa_ifrs": "2026"}),
    ("xbrl", "https://www.cmfchile.cl/institucional/inc/inf_financiera/ifrs/safec_ifrs_verarchivo.php?auth=&send=&rut=90749000&mm=06&aa=2026&archivo=90749000_202606_C.zip&desc_archivo=Estados%20financieros%20(XBRL)&tipo_archivo=XBRL", None),
]:
    try:
        b = req(url, data); open(os.path.join(OUT, f"eeff_{nombre}.bin"), "wb").write(b); log.append(f"{nombre}: {len(b)} bytes, inicio {b[:60]!r}")
    except Exception as e: log.append(f"{nombre}: {e}")
open(os.path.join(OUT, "eeff_log.txt"), "w").write("\n".join(log))
print("\n".join(log))
