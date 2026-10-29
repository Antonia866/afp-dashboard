"""Descarga automática de fuentes públicas para los monitores de Bancos y Retail.

- CMF: Reporte Mensual de Información Financiera del Sistema Bancario (un Excel por mes).
- INE: cuadros estadísticos (Excel) de Actividad Mensual del Comercio (IAC, IACM, ISUP).

Guarda los archivos en datos_monitores/cmf y datos_monitores/ine y deja un registro en
datos_monitores/manifest.json. No modifica archivos que no cambiaron.
"""
import hashlib, html, json, os, re, sys, time, urllib.request, urllib.parse
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
MESES = {"enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,"julio":7,"agosto":8,"septiembre":9,"setiembre":9,"octubre":10,"noviembre":11,"diciembre":12}

CMF_LIST = "https://www.cmfchile.cl/portal/estadisticas/626/w4-propertyvalue-28911.html"
INE_PAGES = [
    "https://www.ine.gob.cl/estadisticas-por-tema/comercio-y-servicios/actividad-mensual-del-comercio",
]

def get(url, binary=False, tries=3):
    last = None
    for i in range(tries):
        try:
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
    page = get(CMF_LIST)
    # artículos del listado: w4-article-NNNNN.html con su título "... - Mes AAAA"
    arts = {}
    for m in re.finditer(r'href="([^"]*w4-article-(\d+)\.html)"[^>]*>(.*?)</a>', page, re.S):
        title = html.unescape(re.sub("<[^>]+>", " ", m.group(3))).strip()
        arts[m.group(2)] = {"url": urllib.parse.urljoin(CMF_LIST, m.group(1)), "title": title}
    files = {}
    for m in re.finditer(r'href="([^"]*articles-(\d+)_recurso_\d+\.xlsx[^"]*)"', page):
        files.setdefault(m.group(2), urllib.parse.urljoin(CMF_LIST, html.unescape(m.group(1))))
    n_new = 0
    for aid in sorted(set(arts) | set(files)):
        title = arts.get(aid, {}).get("title", "")
        url = files.get(aid)
        if not url:
            try:
                art = get(arts[aid]["url"])
                mm = re.search(r'href="([^"]*articles-%s_recurso_\d+\.xlsx[^"]*)"' % aid, art)
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
        if save(os.path.join(BASE, "cmf", f"reporte_mensual_{per}.xlsx"), data):
            n_new += 1; log.append(f"CMF {per}: actualizado ({len(data)//1024} KB) desde {url}")
    return n_new

def ine(log):
    n_new = 0
    for p in INE_PAGES:
        page = get(p)
        links = set(urllib.parse.urljoin(p, html.unescape(h)) for h in re.findall(r'href="([^"]+\.xlsx?[^"]*)"', page, re.I))
        for url in sorted(links):
            name = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path))
            name = re.sub(r"[^A-Za-z0-9._()\-áéíóúñÁÉÍÓÚÑ ]", "_", name)
            try:
                data = get(url, binary=True)
            except Exception as e:  # noqa
                log.append(f"INE {name}: no se pudo descargar ({e})"); continue
            if save(os.path.join(BASE, "ine", name), data):
                n_new += 1; log.append(f"INE {name}: actualizado ({len(data)//1024} KB) desde {url}")
    return n_new

def main():
    log, res = [], {}
    for nombre, fn in (("cmf", cmf), ("ine", ine)):
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
    archivos = {d: sorted(os.listdir(os.path.join(BASE, d))) for d in ("cmf", "ine") if os.path.isdir(os.path.join(BASE, d))}
    man = {"actualizado": hist[0]["fecha"], "archivos": archivos, "historial": hist[:60],
           "fuentes": {"cmf": CMF_LIST, "ine": INE_PAGES}}
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(json.dumps(hist[0], ensure_ascii=False, indent=1))
    if all(v is None for v in res.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
