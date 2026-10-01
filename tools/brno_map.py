"""Build the hero map (assets/brno-map.svg) from the Jihlavská closure scenario
of the traffic-closures-sim pipeline and inline it into index.html.

    python3 tools/brno_map.py [path/to/Traffic-sim-backend]

Grey: the whole modelled network. Orange/blue: links that gain/lose more than
THRESHOLD vehicles per day when Jihlavská is closed. Black: the closed section.
"""
import json, math, re, sys
from pathlib import Path
from shapely.geometry import LineString

SIM = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "git/personal/Traffic-sim-backend")
SCENARIO = SIM / "simulation_for_article/updated_version_8/mapa_scenar_Jihlavská.html"
NETWORK = SIM / "outputs/brno/baseline/network/network_links.geojson"
SITE = Path(__file__).resolve().parent.parent

LON0, LON1, LAT0, LAT1 = 16.470, 16.690, 49.140, 49.236   # crop: Brno around the closure
W = 1600
THRESHOLD = 1000                                           # veh/day
KX = math.cos(math.radians((LAT0 + LAT1) / 2))
SCALE = W / ((LON1 - LON0) * KX)
H = round((LAT1 - LAT0) * SCALE)

MAJOR = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link"}


def project(coords):
    return [((lon - LON0) * KX * SCALE, (LAT1 - lat) * SCALE) for lon, lat in coords]


def inside(pts):
    return any(-20 <= x <= W + 20 and -20 <= y <= H + 20 for x, y in pts)


def path_d(lines, tol=0.6):
    out = []
    for pts in lines:
        if not inside(pts):
            continue
        if len(pts) > 2:
            pts = list(LineString(pts).simplify(tol).coords)
        out.append("M" + "L".join(f"{x:.0f} {y:.0f}" for x, y in pts))
    return "".join(out)


def load_scenario():
    text = SCENARIO.read_text(encoding="utf-8")
    start = text.index("const D = ") + len("const D = ")
    return json.loads(text[start:text.index("\n", start)].rstrip(";"))


def main():
    data = load_scenario()
    closed_ids = set(data.get("zavrete_ids") or [])
    up = {1: [], 2: [], 3: []}
    down = {1: [], 2: [], 3: []}
    closed = []
    for link in data["hrany"]:
        pts = project(link["c"])
        if link["id"] in closed_ids:
            closed.append(pts)
            continue
        delta = link["s"] - link["b"]
        if abs(delta) < THRESHOLD:
            continue
        size = 1 if abs(delta) < 3000 else 2 if abs(delta) < 8000 else 3
        (up if delta > 0 else down)[size].append(pts)

    minor, major = [], []
    for f in json.loads(NETWORK.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        if not g:
            continue
        parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        bucket = major if f["properties"].get("link_type") in MAJOR else minor
        bucket.extend(project(p) for p in parts)

    layers = [
        ("net-minor", path_d(minor, 1.6)),
        ("net-major", path_d(major)),
    ]
    for size in (1, 2, 3):
        layers.append((f"down s{size}", path_d(down[size])))
    for size in (1, 2, 3):
        layers.append((f"up s{size}", path_d(up[size])))
    layers.append(("closed", path_d(closed, 0.3)))

    # label the closed street just above the middle of its drawn extent
    xs = [x for pts in closed for x, _ in pts]
    ys = [y for pts in closed for _, y in pts]
    label = (f'<text class="label" x="{(min(xs) + max(xs)) / 2:.0f}" y="{min(ys) - 18:.0f}" '
             'text-anchor="middle">Jihlavská</text>') if xs else ""

    svg = (f'<svg class="citymap" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice" '
           'role="img" aria-labelledby="map-title">'
           '<title id="map-title">Brno street network with the modelled effect of closing Jihlavská street</title>'
           + "".join(f'<path class="{c}" d="{d}"/>' for c, d in layers if d)
           + label + "</svg>")
    (SITE / "assets/brno-map.svg").write_text(svg, encoding="utf-8")

    index = SITE / "index.html"
    html = index.read_text(encoding="utf-8")
    html, n = re.subn(r"(<!-- map:start -->).*?(<!-- map:end -->)",
                      lambda m: m.group(1) + svg + m.group(2), html, flags=re.S)
    if n:
        index.write_text(html, encoding="utf-8")
    print(f"{W}x{H}  links up={sum(map(len, up.values()))} down={sum(map(len, down.values()))} "
          f"closed={len(closed)}  svg={len(svg) / 1024:.0f} kB  inlined={bool(n)}")


if __name__ == "__main__":
    main()
