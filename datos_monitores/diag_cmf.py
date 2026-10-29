import os, re, urllib.request
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostico"); os.makedirs(OUT, exist_ok=True)
for pid in ("28917", "32901", "43452"):
    url = f"https://www.cmfchile.cl/portal/estadisticas/626/w4-propertyvalue-{pid}.html"
    try:
        b = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60).read()
        open(os.path.join(OUT, f"cmf_{pid}.html"), "wb").write(b)
        arts = sorted(set(re.findall(rb'href="([^"]*w4-article-\d+\.html)"', b)))[:3]
        for a in arts:
            u = a.decode()
            if u.startswith("/"): u = "https://www.cmfchile.cl" + u
            try:
                bb = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=60).read()
                open(os.path.join(OUT, f"cmf_{pid}_art_{re.search(r'article-(\d+)', u).group(1)}.html"), "wb").write(bb)
            except Exception as e: print(u, e)
    except Exception as e: print(url, e)
