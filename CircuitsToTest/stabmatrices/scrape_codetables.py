"""
scrape_codetables.py - pull best-known quantum stabilizer codes from codetables.de

website has NO API -> its a PHP form that returns one HTML page per query the single-code query URL is deterministic:

    https://www.codetables.de/QECC/QECC.php?q=<q>&n=<n>&k=<k>

when something looks wrong - dump one raw page and read it:
    python3 scrape_codetables.py --q 2 --n 5 --k 1 --dump
    python3 scrape_codetables.py --q 2 --n-range 5 10 --k-range 1 4 --out-dir OUTPUT_DIR
and also open that exact URL in a browser to compare

this is one academics server (Markus Grassl, codes@codetables.de)
keep the delay; results are cached!
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

BASE = "https://www.codetables.de/QECC/QECC.php"
CACHE = Path("codetables_cache")
USER_AGENT = "codetables-scraper/1.0 (research; ajla.sacic@nyu.edu)"
UNAVAILABLE_MARKERS = ("not available", "no code", "does not exist", "no such code", "sorry")

def build_url(q, n, k):
    return f"{BASE}?{urlencode({'q': q, 'n': n, 'k': k})}"
 
 
def fetch(q, n, k, delay=2.0, cache_dir=CACHE, use_cache=True):
    cache_dir.mkdir(exist_ok=True)
    cached = cache_dir / f"q{q}_n{n}_k{k}.html"
    url = build_url(q, n, k)
    if use_cache and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace"), url
 
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        final_url = resp.geturl()
    cached.write_text(html, encoding = "utf-8")
    # be nice to the server
    time.sleep(delay)  
    return html, final_url
 
 
def clean_pre(html):
    m = re.search(r"<pre>(.*?)</pre>", html, re.I | re.S)
    text = m.group(1) if m else html
    text = re.sub(r"<[^>]+>", "", text)
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"')):
        text = text.replace(a, b)
    return text.strip()
 

 
# A single stabilizer-matrix row on its own line, e.g. [1 0 1 0 1|0 0 1 0 1] <- NOT REAL
# matches one [...] with no nested brackets -> so it never matches the [[n,k,d]] parameter lines
_ROW_RE = re.compile(r"^\[[^\[\]]*\]$")
 
def _scrape_distance(pre, n, k, d_lower, d_upper):
    triples = re.findall(r"\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]", pre)
    for a, b, c in triples:
        if int(a) == n and int(b) == k:
            return int(c)
    if triples:
        return int(triples[0][2])
    if d_lower is not None and d_lower == d_upper:
        return d_lower
    return None
 
 
def _scrape_matrix_block(pre):
    lines = pre.splitlines()
    start = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "stabiliz" in low and "matrix" in low:
            start = i + 1
            break
    if start is None:
        return ""
 
    rows = []
    for ln in lines[start:]:
        s = ln.strip()
        if not s:
            # blank line after the matrix -> block is over
            if rows:          
                break
            # tolerate blank line(s) before the first row
            continue          
        if _ROW_RE.match(s):
            rows.append(s)
        # first non-row line after the matrix -> over
        elif rows:
            break             
        else:
             # unexpected line before any row -> give up
            break            
    return "\n".join(rows)
 
 
def parse(html, q, n, k):
    
    # parse one result page
    # record["status"]:
    #   "ok" -> a stabilizer matrix was found (exact d known)
    #   "no_matrix" -> page loaded, bounds maybe present, but no matrix stored
    #   "unavailable" -> site says the code/construction isnt available (used for error checking -> dimension mismatch)
    # record["matrix"]: matrix text (rows), or "" if "no_matrix"
    
    pre = clean_pre(html)
    low = pre.lower()
 
    def grab_int(label):
        mm = re.search(label + r"\s*[:=]?\s*(\d+)", pre, re.I)
        return int(mm.group(1)) if mm else None
 
    d_lower = grab_int(r"lower\s*bound")
    d_upper = grab_int(r"upper\s*bound")
    d = _scrape_distance(pre, n, k, d_lower, d_upper)
    matrix = _scrape_matrix_block(pre)
 
    if matrix:
        status = "ok"
    elif any(mk in low for mk in UNAVAILABLE_MARKERS):
        status = "unavailable"
    else:
        status = "no_matrix"
 
    return {"q": q, "n": n, "k": k, "d": d, "d_lower": d_lower, "d_upper": d_upper, "status": status, 
            "n_generators": len(matrix.splitlines()) if matrix else 0, "matrix": matrix}
 
def out_filename(q, n, k, d):
    return f"q_{q}_n_{n}_k_{k}_d{d}"
 
def frange(lo, hi):
    return range(int(lo), int(hi) + 1)
 
 
def main(argv=None):
    ap = argparse.ArgumentParser(description="Scrape quantum codes from codetables.de")
    ap.add_argument("--q", type = int, default = 1,  help = "local-dimension EXPONENT; actual local dimension = 2**q (1->2, 2->4, 3->8)")
    ap.add_argument("--n", type = int, help = "single length n")
    ap.add_argument("--k", type = int, help = "single dimension k")
    ap.add_argument("--n-range", nargs = 2, type = int, metavar = ("NMIN", "NMAX"))
    ap.add_argument("--k-range", nargs = 2, type = int, metavar = ("KMIN", "KMAX"))
    ap.add_argument("--delay", type = float, default = 2.0, help = "seconds between requests")
    ap.add_argument("--out-dir", type = str, default = ".", help = "directory for the matrix files (default: current dir)")
    ap.add_argument("--dump", action = "store_true", help = "print URL + full raw page for ONE code and exit (diagnostic)")
    ap.add_argument("--no-cache", action = "store_true", help = "ignore the on-disk cache")
    args = ap.parse_args(argv)
 
    q = 2 ** args.q
 
    if args.dump:
        if args.n is None or args.k is None:
            ap.error("--dump needs a single --n and --k")
        html, url = fetch(q, args.n, args.k, delay=0.0, use_cache=not args.no_cache)
        print("URL:", url)
        print("-" * 60, "RAW <PRE> (or page text):", "-" * 60, sep="\n")
        print(clean_pre(html))
        return
 
    if args.n is None and args.n_range is None:
        ap.error("give --n or --n-range")
    if args.k is None and args.k_range is None:
        ap.error("give --k or --k-range")
 
    ns = [args.n] if args.n is not None else list(frange(*args.n_range))
    ks = [args.k] if args.k is not None else list(frange(*args.k_range))
 
    out_dir = Path(args.out_dir+"/codetables_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
 
    written = 0
    for n in ns:
        for k in ks:
            if k > n:
                continue
            try:
                html, _ = fetch(q, n, k, delay=args.delay, use_cache=not args.no_cache)
            except Exception as e:
                print(f"[!] q{q} n{n} k{k}: fetch failed: {e}", file=sys.stderr)
                continue
 
            rec = parse(html, q, n, k)
            drange = f"{rec['d_lower']}..{rec['d_upper']}"
 
            if rec["status"] == "ok" and rec["d"] is not None:
                path = out_dir / out_filename(q, n, k, rec["d"])
                path.write_text(rec["matrix"] + "\n", encoding="utf-8")
                written += 1
                print(f"[[{n},{k},{rec['d']}]]_{q}  ok           "
                      f"rows={rec['n_generators']}  -> {path}")
            else:
                # No stored matrix -> nothing to save (bounds only). Just report.
                print(f"[[{n},{k}]]_{q}  {rec['status']:11s}  "
                      f"d={drange}  (no matrix stored, no file written)")
 
    print(f"\nWrote {written} matrix file(s) to {out_dir}/")
 
 
if __name__ == "__main__":
    main()