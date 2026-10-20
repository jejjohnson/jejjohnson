"""Profile README figures, drawn with GeoStack's own figure primitives.

Usage: python make_profile_figures.py <path to geostack/docs/images/overview> <outdir>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("mf", src / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
sys.modules["mf"] = mf
spec.loader.exec_module(mf)

Svg, INK, MUTED, LINE, PANEL, BORDER = mf.Svg, mf.INK, mf.MUTED, mf.LINE, mf.PANEL, mf.BORDER
OBS, STATE, ML, ENG = mf.OBS, mf.STATE, mf.ML, mf.ENG
NUM = ("#475569", "#f1f5f9")
DEC = ("#be123c", "#fff1f2")
OCEAN = mf.DOMAINS["Ocean"]
ATMOS = mf.DOMAINS["Atmosphere"]
ORANGE = mf.ORACLE


def chips(f, x, y, names, hue=(BORDER, "#ffffff"), size=11.5, h=23, gap=6, dashed=False):
    for n in names:
        x += f.chip(x, y, n, hue=hue, size=size, h=h, pad=9, dashed=dashed) + gap
    return x


def hero() -> Svg:
    f = Svg(
        1140,
        612,
        "The vision: observations from any sensor flow through GeoStack into"
        " differentiable physics (GeoModels) and inference (GeoML), built on"
        " GeoNumerics, and out as decisions with uncertainty; gradients flow"
        " back end to end. Two case studies instantiate it: methane, from"
        " radiance to emission rate, and sea surface height, from altimetry"
        " tracks to gap-free maps.",
    )
    f.text(30, 40, "From observations to decisions, differentiable end to end",
           size=19, weight=700, anchor="start")
    f.text(30, 62, "one stack of small JAX libraries · layers import downward only · every step has a test oracle",
           size=12.5, color=MUTED, anchor="start")

    # columns
    top, hgt = 92, 250
    # observe
    f.rect(30, top, 190, hgt, stroke=OBS[0], fill=OBS[1])
    f.text(48, top + 28, "Observe", size=15, weight=700, color=OBS[0], anchor="start")
    f.lines(48, top + 48, ["any sensor, any level", "L0 → L2"], size=11.5, color=MUTED, anchor="start")
    y = top + 92
    for row in (["hyperspectral"], ["multispectral"], ["altimetry"], ["in-situ"]):
        chips(f, 48, y, row, hue=(OBS[0], "#ffffff"))
        y += 33

    # GeoStack
    f.rect(258, top, 190, hgt, stroke=OBS[0], fill="#ffffff")
    f.text(276, top + 28, "GeoStack", size=15, weight=700, color=OBS[0], anchor="start")
    f.lines(276, top + 48, ["read · grid · co-register", "patch · compose"], size=11.5, color=MUTED, anchor="start")
    y = top + 92
    for n in ["geotoolz", "geoarray", "xrtoolz", "pipekit"]:
        f.chip(276, y, n, size=11.5, h=23, pad=9)
        y += 33

    # physics ⇄ inference
    px = 486
    f.rect(px, top, 400, 112, stroke=STATE[0], fill=STATE[1])
    f.text(px + 18, top + 28, "GeoModels · physics", size=15, weight=700, color=STATE[0], anchor="start")
    f.text(px + 18, top + 50, "forward models you can differentiate:  y = H(f(θ)) + ε",
           size=12, color=INK, anchor="start")
    chips(f, px + 18, top + 70, ["radtrax", "plumax", "somax"])
    f.rect(px, top + 138, 400, 112, stroke=ML[0], fill=ML[1])
    f.text(px + 18, top + 166, "GeoML · inference", size=15, weight=700, color=ML[0], anchor="start")
    f.text(px + 18, top + 188, "posteriors, emulators, amortized maps:  p(θ | y)",
           size=12, color=INK, anchor="start")
    chips(f, px + 18, top + 208, ["vardax", "filterax", "pyrox", "geonnax", "gauss_flows", "xtremax"], size=10, h=22, gap=4)
    f.arrow([(px + 160, top + 112), (px + 160, top + 136)], color=INK, both=True)
    f.text(px + 172, top + 129, "simulate  ·  invert", size=11, color=MUTED, anchor="start")

    # decide
    f.rect(924, top, 186, hgt, stroke=DEC[0], fill=DEC[1])
    f.text(942, top + 28, "Decide", size=15, weight=700, color=DEC[0], anchor="start")
    f.lines(942, top + 48, ["L3 → L5, with error bars"], size=11.5, color=MUTED, anchor="start")
    y = top + 92
    for n in ["emission rates", "gap-free maps ± σ", "return levels", "alerts"]:
        f.chip(942, y, n, hue=(DEC[0], "#ffffff"), size=11.5, h=23, pad=9)
        y += 33

    mid = top + hgt / 2
    f.arrow([(220, mid), (256, mid)], color=INK)
    f.arrow([(448, mid), (484, mid)], color=INK)
    f.arrow([(886, mid), (922, mid)], color=INK)

    # numerics foundation
    nb = top + hgt + 16
    f.rect(258, nb, 628, 40, stroke=NUM[0], fill=NUM[1])
    f.text(276, nb + 25, "GeoNumerics", size=13, weight=700, color=NUM[0], anchor="start")
    chips(f, 380, nb + 9, ["gaussx", "finitevolX", "spectraldiffx"], size=11, h=22)
    f.text(870, nb + 25, "structured linear algebra · PDE operators", size=11, color=MUTED, anchor="end")

    # gradient return path
    gy = nb + 66
    f.arrow([(1017, top + hgt), (1017, gy), (125, gy), (125, top + hgt + 2)],
            color=STATE[0], dashed=True)
    f.text(571, gy - 8, "jax.grad flows back through every step:  ∂ decision / ∂ observation, ∂ / ∂ θ",
           size=12, color=STATE[0], weight=600)

    # case studies
    cy = gy + 30
    f.text(30, cy + 6, "Case studies", size=13, weight=700, anchor="start")
    rows = [
        ("Methane", ATMOS, "retrieval + plumes",
         ["EMIT · S2 · Landsat radiance", "matched filter + OE", "plume vetting · IME", "emission rate (kg/h)"],
         "radtrax · plumax · vardax"),
        ("Sea surface height", OCEAN, "interpolation + DA",
         ["along-track altimetry", "QG model · neural fields · OI", "4D-Var · ensembles", "gap-free SSH map ± σ"],
         "somax · geonnax · vardax · OceanBench"),
    ]
    for i, (name, hue, sub, steps, pkgs) in enumerate(rows):
        y = cy + 20 + i * 60
        f.rect(30, y, 1080, 48, stroke=hue[0], fill=hue[1])
        f.text(46, y + 22, name, size=13.5, weight=700, color=hue[0], anchor="start")
        f.text(46, y + 38, sub, size=11, color=MUTED, anchor="start")
        x = 210
        for j, s in enumerate(steps):
            w = f.chip(x, y + 12, s, hue=(hue[0], "#ffffff"), size=11.5, h=24, pad=10)
            if j < len(steps) - 1:
                f.arrow([(x + w + 3, y + 24), (x + w + 19, y + 24)], color=hue[0], width=1.4)
            x += w + 24
        f.text(1096, y + 29, pkgs, size=11, color=MUTED, anchor="end", italic=True)
    return f


def lineage() -> Svg:
    f = Svg(
        1140,
        470,
        "Swimlane timeline of first commits, 2017 to 2026: kernels, manifold"
        " learning, Gaussianization, neural fields and extremes code became"
        " GeoML; the 2023 ocean solvers became GeoModels and GeoNumerics;"
        " OceanBench became GeoStack's benchmarks; methane work became radtrax"
        " and plumax.",
    )
    f.text(30, 40, "A decade of research code, consolidated into one stack",
           size=19, weight=700, anchor="start")
    f.text(30, 62, "first commit of each repo, by the GeoStack layer it grew into",
           size=12.5, color=MUTED, anchor="start")
    x0, x1 = 190, 800
    def X(y):
        return x0 + (y - 2017) * (x1 - x0) / 9
    top, lh = 92, 76
    lanes = [
        ("GeoML", ML, "learning + inference",
         [("manipy", 2017.1, 0), ("kernellib", 2017.2, 1), ("rbig", 2018.0, 0),
          ("ml4ssh · neural fields", 2022.4, 0), ("jbayesevt", 2024.0, 1)],
         ["kernellib · pyrox · gauss_flows", "geonnax · vardax · xtremax"]),
        ("GeoModels", STATE, "+ GeoNumerics",
         [("finitevolX · spectraldiffx · somax", 2023.8, 0)],
         ["somax · finitevolX", "spectraldiffx · gaussx"]),
        ("Benchmarks", OCEAN, "data + evaluation",
         [("OceanBench · NeurIPS 2023", 2023.05, 0)],
         ["geotoolz · xrtoolz", "pipekit · GeoBench"]),
        ("Methane", ATMOS, "retrieval + plumes",
         [("plumax · radtrax", 2026.3, 0)],
         ["radtrax · plumax", "matched filter → OE"]),
    ]
    for i, (name, hue, sub, items, dest) in enumerate(lanes):
        y = top + i * lh
        f.rect(30, y, 1080, lh - 10, stroke=hue[0], fill=hue[1], rx=10)
        f.text(48, y + 28, name, size=14, weight=700, color=hue[0], anchor="start")
        f.text(48, y + 46, sub, size=11, color=MUTED, anchor="start")
        cy = y + (lh - 10) / 2
        f.parts.append(f'<line x1="{x0-10}" y1="{cy}" x2="{X(2026)+0}" y2="{cy}" stroke="{hue[0]}" stroke-opacity="0.35" stroke-width="2"/>')
        for label, yr, row in items:
            x = X(yr)
            f.parts.append(f'<circle cx="{x:.1f}" cy="{cy}" r="4.5" fill="{hue[0]}"/>')
            cyy = cy - 30 if row == 0 else cy + 6
            w = mf._width(label, 11) + 16
            cx = x + 6 if x + 6 + w < 860 else x - w - 6
            f.chip(cx, cyy, label, hue=(hue[0], "#ffffff"), size=11, h=22, pad=8)
        dx = 870
        f.arrow([(X(2026) + 6, cy), (dx - 4, cy)], color=hue[0], width=1.8)
        f.rect(dx, y + 9, 226, lh - 28, stroke=hue[0], fill="#ffffff")
        f.lines(dx + 113, y + 32, dest, gap=16, size=11.5, color=INK)
    ay = top + 4 * lh + 6
    f.parts.append(f'<line x1="{x0}" y1="{ay}" x2="{X(2026)}" y2="{ay}" stroke="{LINE}"/>')
    for yr in range(2017, 2027):
        f.parts.append(f'<line x1="{X(yr):.1f}" y1="{ay-4}" x2="{X(yr):.1f}" y2="{ay+4}" stroke="{LINE}"/>')
        f.text(X(yr), ay + 20, str(yr), size=11.5, color=MUTED)
    f.text(983, ay + 20, "GeoStack, 2026", size=12, weight=700)
    return f


def geostack_hero() -> Svg:
    W = 1180
    f = Svg(
        W,
        800,
        "GeoStack along the data-maturity axis. Top, the science axis: L0"
        " unstructured observations to L5 derived products, observations up"
        " to L2 and model states from L3. Two case studies run along it:"
        " methane from radiance to emission rate, and sea surface height from"
        " altimeter tracks to reanalysis. Middle, the compute that makes each"
        " step, each following the same ladder from simple model to amortized"
        " inference. Bottom, the engineering axis: store, index, read, carry,"
        " patch, transform and compose, with the carrier switching from"
        " geoarray to xarray at L3.",
    )
    f.text(30, 42, "GeoStack: one stack along the data-maturity axis", size=20, weight=700, anchor="start")
    f.text(30, 64, "science says what each level means · engineering says how it is stored, carried and composed · compute is the step between levels",
           size=12.5, color=MUTED, anchor="start")
    gx, cw, bw = 170, 164, 136
    def cx(i):
        return gx + i * cw + bw / 2
    def bx(i):
        return gx + i * cw
    DER = ("#be123c", "#fff1f2")
    lv = [
        ("L0", "Unstructured", "raw instrument", OBS), ("L1", "Structured", "calibrated radiance", OBS),
        ("L2", "Retrieved", "per-pixel geophysics", OBS), ("L3", "Analysis", "gridded, gap-filled", STATE),
        ("L4", "Reanalysis", "dynamical model states", STATE), ("L5", "Derived", "extremes, indicators", DER),
    ]
    # header bands
    f.text((bx(0) + bx(2) + bw) / 2, 98, "observations", size=12, weight=700, color=OBS[0])
    f.text((bx(3) + bx(4) + bw) / 2, 98, "model states", size=12, weight=700, color=STATE[0])
    f.text(cx(5), 98, "decisions", size=12, weight=700, color=DER[0])
    # science band
    sy = 110
    f.rect(20, sy - 4, W - 40, 228, stroke=BORDER, fill=PANEL, rx=12)
    f.text(34, sy + 26, "SCIENCE", size=12, weight=800, color=INK, anchor="start")
    f.lines(34, sy + 44, ["what is measured", "and modelled"], size=11, color=MUTED, anchor="start")
    for i, (L, t, sub, hue) in enumerate(lv):
        dashed = L == "L5"
        f.rect(bx(i), sy + 8, bw, 64, stroke=hue[0], fill=hue[1], dashed=dashed)
        f.text(bx(i) + 12, sy + 28, L, size=12, weight=800, color=hue[0], anchor="start")
        f.text(bx(i) + 12, sy + 46, t, size=13.5, weight=700, anchor="start")
        f.text(bx(i) + 12, sy + 62, sub, size=10.5, color=MUTED, anchor="start")
        if i < 5:
            f.arrow([(bx(i) + bw + 2, sy + 40), (bx(i + 1) - 2, sy + 40)], color=INK, width=1.4)
    f.text(34, sy + 104, "domains", size=11, color=MUTED, anchor="start")
    chips(f, bx(0), sy + 90, ["ocean", "atmosphere", "land", "ice"], size=11, h=22)
    f.text(bx(2) + 40, sy + 105, "one radiance stream, retrieved into every geoscience", size=11, color=MUTED, anchor="start", italic=True)
    # case-study tracks
    tracks = [
        ("methane", ATMOS, 1, 5, ["radiance", "XCH₄ enhancement", "plume mask", "flux + assimilate", "kg/h · alerts"]),
        ("sea surface height", OCEAN, 2, 4, ["along-track SSH", "gap-free map ± σ", "QG 4D-Var state"]),
    ]
    for k, (name, hue, a, b, stops) in enumerate(tracks):
        y = sy + 138 + k * 40
        f.text(34, y + 15, name, size=12, weight=700, color=hue[0], anchor="start")
        f.parts.append(f'<line x1="{cx(a):.1f}" y1="{y+11}" x2="{cx(b):.1f}" y2="{y+11}" stroke="{hue[0]}" stroke-width="3" stroke-opacity="0.5"/>')
        for j, st in enumerate(stops):
            i = a + j
            w = mf._width(st, 11) + 18
            f.chip(cx(i) - w / 2, y, st, hue=(hue[0], hue[1]), size=11, h=22, pad=9)
    # compute band
    cy0 = sy + 240
    f.rect(20, cy0, W - 40, 150, stroke=ML[0], fill="#ffffff", rx=12)
    f.text(34, cy0 + 26, "COMPUTE", size=12, weight=800, color=ML[0], anchor="start")
    f.lines(34, cy0 + 44, ["the step between", "two levels"], size=11, color=MUTED, anchor="start")
    steps = [
        (1, "calibrate", "instrument models"), (2, "retrieve", "radtrax · matched filter, OE"),
        (3, "grid + gap-fill", "pyrox · geonnax · vardax OI"), (4, "model + assimilate", "somax · plumax · vardax · filterax"),
        (5, "derive", "xtremax"),
    ]
    for i, t, pk in steps:
        w = 150
        x = bx(i) - (cw - bw) / 2 - w / 2
        f.rect(x, cy0 + 12, w, 58, stroke=ML[0], fill=ML[1])
        f.text(x + w / 2, cy0 + 34, t, size=12.5, weight=700)
        words = pk.split(" · ")
        rows = [" · ".join(words[:2]), " · ".join(words[2:])] if len(words) > 2 else [pk]
        f.lines(x + w / 2, cy0 + 50, rows, gap=13, size=10, color=MUTED)
    f.rect(bx(1) - 22, cy0 + 82, bx(5) + bw - bx(1) + 22, 26, stroke=ORANGE[0], fill=ORANGE[1], rx=13)
    f.text((bx(1) - 22 + bx(5) + bw) / 2, cy0 + 99,
           "every step climbs the same ladder: simple model → trusted oracle (DA / inversion) → emulator → amortized inference",
           size=11.5, color=ORANGE[0], weight=600)
    f.rect(bx(1) - 22, cy0 + 114, bx(5) + bw - bx(1) + 22, 26, stroke=NUM[0], fill=NUM[1], rx=13)
    f.text((bx(1) - 22 + bx(5) + bw) / 2, cy0 + 131,
           "GeoNumerics underneath: gaussx · finitevolX · spectraldiffx  —  on JAX · Equinox · lineax · NumPyro",
           size=11.5, color=NUM[0], weight=600)
    # engineering band
    ey = cy0 + 166
    f.rect(20, ey, W - 40, 284, stroke=ENG[0], fill="#ffffff", rx=12)
    f.text(34, ey + 26, "ENGINEERING", size=12, weight=800, color=ENG[0], anchor="start")
    f.text(130, ey + 26, "how it is stored, indexed, read, carried, patched, transformed and composed", size=11, color=MUTED, anchor="start")
    split = (bx(2) + bw + bx(3)) / 2
    left0, right1 = bx(0), bx(5) + bw
    rows = [
        ("compose", [(left0, right1, "pipekit · operators, graphs, assimilation cycles, training", ENG)]),
        ("transform", [(bx(1), split - 8, "geoops", OBS), (split + 8, right1, "xrtoolz", STATE)]),
        ("patch", [(bx(1), bx(4) + bw, "geopatcher · carrier-neutral fields", ENG)]),
        ("carry", [(left0, split - 8, "geoarray · NumPy or JAX, differentiable grid", OBS), (split + 8, right1, "xarray · coordax", STATE)]),
        ("read + index", [(left0, split - 8, "geocatalog · readers", OBS), (split + 8, right1, "xrreader · geocatalog.states", STATE)]),
        ("store", [(left0, right1, "geocloud · COG for observations, Zarr / NetCDF for states", ENG)]),
    ]
    f.parts.append(f'<line x1="{split:.1f}" y1="{ey+36}" x2="{split:.1f}" y2="{ey+276}" stroke="{STATE[0]}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    for r, (lab, bars) in enumerate(rows):
        y = ey + 44 + r * 38
        f.text(bx(0) - 18, y + 19, lab, size=11.5, weight=600, anchor="end")
        for x0, x1, txt, hue in bars:
            f.rect(x0, y, x1 - x0, 26, stroke=hue[0], fill=hue[1], rx=13)
            f.text((x0 + x1) / 2, y + 17.5, txt, size=11.5)
    f.text(split, ey + 304, "carrier switch at L3: geoarray (observations) → xarray (states)", size=11.5, weight=700, color=STATE[0])
    f.h = int(ey + 318)
    return f


out.mkdir(parents=True, exist_ok=True)
for fig in (hero, lineage, geostack_hero):
    (out / f"{fig.__name__}.svg").write_text(fig().render())
print("ok")
