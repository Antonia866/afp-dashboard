"""Descarga automática de fuentes públicas para los monitores de Bancos y Retail.

- CMF: Reporte Mensual de Información Financiera del Sistema Bancario (un Excel por mes).
- INE: cuadros estadísticos (Excel) de Actividad Mensual del Comercio (IAC, IACM, ISUP).

Guarda los archivos en datos_monitores/cmf y datos_monitores/ine y deja un registro en
datos_monitores/manifest.json. No modifica archivos que no cambiaron.
"""
import hashlib, html, json, os, re, sys, time, unicodedata, urllib.request, urllib.parse
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
MESES = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,"julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12}

CMF_LIST = "https://www.cmfchile.cl/portal/estadisticas/626/w4-propertyvalue-28911.html"
CMF_LISTAS = [  # (listado, carpeta, prefijo, extensiones)
    (CMF_LIST, "cmf", "reporte_mensual", ("xlsx",)),
    ("https://www.cmfchile.cl/portal/estadisticas/626/w4-propertyvalue-28917.html", "cmf_balances", "balance", ("zip", "txt", "xlsx")),
]
INE_PAGES = [
    "https://www.ine.gob.cl/estadisticas-por-tema/comercio-y-servicios/actividad-mensual-del-comercio",
]

def get(url, binary=False, tries=3):
    last = None
    for i in range(tries):
        try:
            url = urllib.parse.quote(url, safe=":/?=&%#()+,;~@!$'*[]")
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "es-CL,es;q=0.9"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except Exception as e:  # noqa
            last = e; time.sleep(3 * (i + 1))
    raise last

def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == hashlib.sha256(data).hexdigest():
                return False
    with open(path, "wb") as f:
        f.write(data)
    return True

def cmf(log):
    return sum(cmf_lista(log, *x) for x in CMF_LISTAS)

def cmf_lista(log, CMF_LIST, carpeta, prefijo, exts):
    page = get(CMF_LIST)
    # artículos del listado: w4-article-NNNNN.html con su título "... - Mes AAAA"
    arts = {}
    for m in re.finditer(r'href="([^"]*w4-article-(\d+)\.html)"[^>]*>(.*?)</a>', page, re.S):
        title = html.unescape(re.sub("<[^>]+>", " ", m.group(3))).strip()
        arts[m.group(2)] = {"url": urllib.parse.urljoin(CMF_LIST, m.group(1)), "title": title}
    files = {}
    for m in re.finditer(r'href="([^"]*articles-(\d+)_recurso_\d+\.(?:%s)[^"]*)"' % "|".join(exts), page):
        files.setdefault(m.group(2), urllib.parse.urljoin(CMF_LIST, html.unescape(m.group(1))))
    n_new = 0
    for aid in sorted(set(arts) | set(files)):
        title = arts.get(aid, {}).get("title", "")
        url = files.get(aid)
        if not url:
            try:
                art = get(arts[aid]["url"])
                mm = re.search(r'href="([^"]*articles-%s_recurso_\d+\.(?:%s)[^"]*)"' % (aid, "|".join(exts)), art)
                url = urllib.parse.urljoin(arts[aid]["url"], html.unescape(mm.group(1))) if mm else None
                if not title:
                    tt = re.search(r"<title>(.*?)</title>", art, re.S); title = tt.group(1) if tt else ""
            except Exception as e:  # noqa
                log.append(f"CMF artículo {aid}: {e}"); continue
        if not url:
            continue
        mt = re.search(r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+(\d{4})", title.lower())
        mt2 = re.search(r"(\d{4})/(\d{2})", title)
        if mt:
            per = f"{mt.group(2)}-{MESES[mt.group(1)]:02d}"
        elif mt2:
            per = f"{mt2.group(1)}-{mt2.group(2)}"
        else:
            per = f"articulo-{aid}"
        try:
            data = get(url, binary=True)
        except Exception as e:  # noqa
            log.append(f"CMF {per}: no se pudo descargar ({e})"); continue
        ext = re.search(r"\.(\w+)(?:\?|$)", url).group(1).lower()
        if save(os.path.join(BASE, carpeta, f"{prefijo}_{per}.{ext}"), data):
            n_new += 1; log.append(f"CMF {prefijo} {per}: actualizado ({len(data)//1024} KB) desde {url}")
    return n_new

def post_json(url, data):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers={"User-Agent": UA, "X-Requested-With": "XMLHttpRequest",
                                 "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))

def limpio(t):
    t = unicodedata.normalize("NFKC", str(t or "")).strip()
    return re.sub(r'[\\/:*?"<>|]+', "_", t)[:120] or "sin_nombre"

def ine(log):
    """Recorre la carpeta 'CUADROS ESTADÍSTICOS' de cada página del INE (misma API que usa su sitio)."""
    n_new = 0
    for p in INE_PAGES:
        page = get(p)
        base = p.rstrip("/") + "/"
        m = re.search(r'data-folder="([0-9a-f-]{36})"[^>]*>(?:<[^>]+>)*\s*CUADROS ESTAD', page, re.I)
        if not m:
            log.append(f"INE: no se encontró la carpeta de cuadros en {p}"); continue
        pila, vistos = [(m.group(1), [])], 0
        while pila and vistos < 120:
            fid, ruta = pila.pop(); vistos += 1
            try: hijos = post_json(base + "hijosCarpeta/", {"idFolder": fid}).get("folder") or []
            except Exception as e: log.append(f"INE carpeta {'/'.join(ruta) or 'raíz'}: {e}"); continue
            if hijos:
                for h in hijos: pila.append((h.get("Id"), ruta + [limpio(h.get("Titulo"))]))
                continue
            try: docs = post_json(base + "getArchivos/", {"idFolder": fid}).get("documento") or []
            except Exception as e: log.append(f"INE archivos {'/'.join(ruta)}: {e}"); continue
            for d in docs:
                tipo = (d.get("Tipo") or "").lower()
                if tipo not in (".xlsx", ".xls", ".csv"): continue
                url = (d.get("Url") or "").replace("http:", "https:")
                destino = os.path.join(BASE, "ine", *ruta, limpio(d.get("Titulo")) + tipo)
                try: data = get(url, binary=True)
                except Exception as e: log.append(f"INE {d.get('Titulo')}: no se pudo descargar ({e})"); continue
                if save(destino, data):
                    n_new += 1; log.append(f"INE {'/'.join(ruta)}/{d.get('Titulo')}: actualizado ({len(data)//1024} KB)")
    return n_new

EMISORES = {  # entidad del monitor -> RUT (sin DV) en la CMF; el nombre se verifica en cada XBRL
    "FALAB": "90749000", "CENCOSUD": "93834000", "RIPLEY": "99579730", "SMU": "76012676",
    "MALLPLAZA": "76017019", "PARAUCO": "94627000", "CENCOMALLS": "76433310",
    "HITES": "96947020", "LAPOLAR": "96874030", "TRICOT": "76266594",
}
XBRL_URL = ("https://www.cmfchile.cl/institucional/inc/inf_financiera/ifrs/safec_ifrs_verarchivo.php?auth=&send=&rut={rut}"
            "&mm={mm}&aa={aa}&archivo={rut}_{aa}{mm}_C.zip&desc_archivo=Estados%20financieros%20(XBRL)&tipo_archivo=XBRL")

def eeff(log):
    """Estados financieros IFRS consolidados (XBRL) de los emisores, últimos 12 trimestres."""
    n_new = 0
    hoy = datetime.now(timezone.utc)
    trimestres = []
    y, q = hoy.year, (hoy.month - 1) // 3 + 1
    for _ in range(13):
        q -= 1
        if q == 0: y, q = y - 1, 4
        trimestres.append((str(y), f"{q*3:02d}"))
    for ent, rut in EMISORES.items():
        for aa, mm in trimestres:
            destino = os.path.join(BASE, "eeff", ent, f"{rut}_{aa}{mm}_C.zip")
            if os.path.exists(destino) and (aa, mm) != trimestres[0] and (aa, mm) != trimestres[1]:
                continue  # trimestres antiguos ya descargados no se vuelven a pedir
            try:
                data = get(XBRL_URL.format(rut=rut, aa=aa, mm=mm), binary=True)
            except Exception as e:
                continue
            if data[:2] != b"PK":
                continue  # aún no publicado
            if save(destino, data):
                n_new += 1; log.append(f"EEFF {ent} {aa}-{mm}: actualizado ({len(data)//1024} KB)")
    return n_new

def main():
    log, res = [], {}
    for nombre, fn in (("cmf", cmf), ("ine", ine), ("eeff", eeff)):
        try:
            res[nombre] = fn(log)
        except Exception as e:  # noqa
            res[nombre] = None; log.append(f"{nombre.upper()}: error general ({e})")
    man_path = os.path.join(BASE, "manifest.json")
    man = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
    hist = man.get("historial", [])
    hist.insert(0, {"fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"), "nuevos": res, "log": log})
    archivos = {d: sorted(os.listdir(os.path.join(BASE, d))) for d in ("cmf", "cmf_balances", "ine", "eeff") if os.path.isdir(os.path.join(BASE, d))}
    man = {"actualizado": hist[0]["fecha"], "archivos": archivos, "historial": hist[:60],
           "fuentes": {"cmf": CMF_LIST, "ine": INE_PAGES}}
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(json.dumps(hist[0], ensure_ascii=False, indent=1))
    if all(v is None for v in res.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
