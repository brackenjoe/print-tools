#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joe Fernandez
"""Check one binary or ASCII STL mesh, or compare two, before printing.

  python3 check_mesh.py A.stl [B.stl] [options]
  python3 check_mesh.py --selftest

Needs numpy and matplotlib.

Two meshes (A and B): checks things like (a) an OpenSCAD rebuild matches an
original mesh, or (b) an unchanged part of a modified mesh (an attachment
base, a mounting face) is still the same as the original. Judged: the
horizontal sections at the chosen heights (XOR area) and the holes at one
height (pairing, centre offset, diameter). Printed for you to read, not
judged: triangles, bounds, volume, surface area and the open-edge count. A
source mesh downloaded from elsewhere is often not closed, and that is not
the comparison's question.

One mesh (A alone): prints A's stats, a table of horizontal sections (area
and loop count per height, reported, not judged), and the holes found at
--hole-z. Open edges (edges not shared by exactly two triangles) make the
result FAIL.

Holes: any closed section loop within 0.05 mm of a circle and smaller than
--hole-max-d (default 6 mm) is listed, so a round pin counts as well as a
bore. Larger bores are covered by the section XOR only.

Non-finite input is refused with exit 2: a mesh with any NaN or inf vertex
coordinate, and any NaN or inf option value (--res must also be above 0).

Heights: --z and --hole-z default to heights chosen from the z range of the
loaded meshes (8 evenly spaced section heights, and mid-height for holes);
a line says so when they are used. --shift-a / --shift-b translate a mesh
(no rotation) before checking; B's shift needs B.

Exit codes: 0 pass (or fail without --strict), 1 fail with --strict or an
unexpected error, 2 usage error or unreadable file.

Run with `-h` for the full option list.
"""
import argparse
import collections
import os
import struct
import sys

import numpy as np
from matplotlib.path import Path

ROUND_EDGE = 4   # decimals: manifold-edge vertex matching
ROUND_LOOP = 5   # decimals: section-loop endpoint matching


# ---------------------------------------------------------------- loading --

def load_stl(path):
    try:
        raw = open(path, 'rb').read()
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(2)
    is_ascii = True
    if len(raw) >= 84:
        n = struct.unpack('<I', raw[80:84])[0]
        if raw[:5] == b'solid' and len(raw) != 84 + 50 * n:
            is_ascii = True
        elif raw[:5] != b'solid' or len(raw) == 84 + 50 * n:
            is_ascii = False
    try:
        tris = _load_ascii(raw) if is_ascii else _load_binary(raw)
    except Exception as e:
        print(f"error: cannot parse {path}: {e}", file=sys.stderr)
        sys.exit(2)
    if tris.shape[0] == 0:
        print(f"error: no triangles read from {path}", file=sys.stderr)
        sys.exit(2)
    bad = ~np.isfinite(tris).all(axis=2)
    if bad.any():
        print(f"error: {path}: {int(bad.sum())} vertices are not finite "
              f"(NaN or inf)", file=sys.stderr)
        sys.exit(2)
    return tris


def _load_binary(raw):
    n = struct.unpack('<I', raw[80:84])[0]
    dt = np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')])
    a = np.frombuffer(raw[84:84 + n * 50], dtype=dt)
    return a['v'].reshape(-1, 3, 3).astype(float)


def _load_ascii(raw):
    text = raw.decode('ascii', errors='replace')
    tris = []
    cur = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('vertex'):
            parts = s.split()
            cur.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    return np.array(tris, dtype=float) if tris else np.zeros((0, 3, 3))


# ------------------------------------------------------------- mesh stats --

def volume(t):
    return np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6


def tri_areas(t):
    return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)


def bad_edge_count(t):
    cnt = collections.Counter()
    r = np.round(t, ROUND_EDGE)
    for tri in r:
        for i in range(3):
            a = tuple(tri[i])
            b = tuple(tri[(i + 1) % 3])
            cnt[tuple(sorted((a, b)))] += 1
    return sum(1 for c in cnt.values() if c != 2)


# ---------------------------------------------------------- plane section --

def section_segments(t, z):
    """Vectorised horizontal (z=const) plane section of a triangle soup.

    Returns an (n, 2, 2) array of 2D (x, y) segment endpoints.
    """
    d = t[:, :, 2] - z
    da = d
    db = np.roll(d, -1, axis=1)
    cross = (da < 0) != (db < 0)
    n_cross = cross.sum(axis=1)
    keep = np.where(n_cross == 2)[0]
    if keep.size == 0:
        return np.zeros((0, 2, 2))
    a = t[keep]
    b = np.roll(t[keep], -1, axis=1)
    daK = da[keep]
    dbK = db[keep]
    denom = daK - dbK
    s = np.divide(daK, denom, out=np.zeros_like(daK), where=denom != 0)
    pts = a + s[:, :, None] * (b - a)           # (k, 3, 3)
    m = cross[keep]                              # (k, 3) bool, exactly 2 True
    segs = np.array([pts[i][m[i]][:, :2] for i in range(keep.size)])
    return segs


def chain_loops(segs2d):
    """Chain a soup of 2D segments into closed loops (Counter + adjacency,
    walking shared endpoints). Returns (loops, warnings)."""
    adj = collections.defaultdict(list)
    for p, q in segs2d:
        kp = tuple(np.round(p, ROUND_LOOP))
        kq = tuple(np.round(q, ROUND_LOOP))
        if kp == kq:
            continue   # degenerate (zero-length) segment
        adj[kp].append(kq)
        adj[kq].append(kp)
    seen = set()
    loops = []
    warnings = []
    for start in list(adj.keys()):
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        prev = None
        cur = start
        while True:
            nxt_candidates = [p for p in adj[cur] if p != prev and p not in seen]
            if not nxt_candidates:
                break
            prev, cur = cur, nxt_candidates[0]
            chain.append(cur)
            seen.add(cur)
        if len(chain) < 3:
            continue
        closed = chain[0] in adj[chain[-1]]
        if not closed:
            warnings.append(
                f"warning: open loop chain ({len(chain)} pts), "
                f"start {chain[0]} end {chain[-1]}"
            )
        loops.append(np.array(chain, dtype=float))
    return loops, warnings


def loop_signed_area(loop):
    x = loop[:, 0]
    y = loop[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def section_area(loops):
    """Area enclosed by the loops under the even-odd rule: a loop's
    contribution is subtracted (it is a hole) if it is nested inside an
    odd number of the other loops.

    Nesting is tested with one of the loop's own boundary vertices (not an
    interior point such as a centroid): loop boundaries from a planar mesh
    section do not cross each other, so a vertex of loop L is either fully
    inside or fully outside another loop J's enclosed area. A centroid or
    vertex-mean point is not safe here -- for a loop that is a plain outer
    ring with a concentric hole loop inside it, the ring's own vertex mean
    can fall inside the hole's area and misclassify the ring as a hole too.
    """
    if not loops:
        return 0.0
    paths = [Path(l) for l in loops]
    total = 0.0
    for i, l in enumerate(loops):
        a = abs(loop_signed_area(l))
        pt = l[0:1]
        nest = sum(
            1 for j, p in enumerate(paths)
            if j != i and p.contains_points(pt)[0]
        )
        total += -a if nest % 2 == 1 else a
    return total


def rasterize_mask(loops, xs, ys):
    """Rasterise the XOR-across-loops fill of `loops` on the (xs, ys) grid.

    Each loop is tested only over its own local bounding sub-grid (padded
    by a cell), not the full grid: a small hole loop would otherwise force
    a full-grid Path.contains_points call just like the big outline loop,
    which dominates runtime for no reason when most loops are small.
    """
    H, W = len(ys), len(xs)
    mask = np.zeros((H, W), dtype=bool)
    if not loops or W == 0 or H == 0:
        return mask
    x0 = xs[0]
    dx = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
    y0 = ys[0]
    dy = (ys[1] - ys[0]) if len(ys) > 1 else 1.0
    for l in loops:
        if len(l) < 3:
            continue
        lxmin, lymin = l.min(axis=0)
        lxmax, lymax = l.max(axis=0)
        i0 = max(0, int(np.floor((lxmin - x0) / dx)) - 1)
        i1 = min(W, int(np.ceil((lxmax - x0) / dx)) + 2)
        j0 = max(0, int(np.floor((lymin - y0) / dy)) - 1)
        j1 = min(H, int(np.ceil((lymax - y0) / dy)) + 2)
        if i0 >= i1 or j0 >= j1:
            continue
        X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
        pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = Path(l).contains_points(pts).reshape(j1 - j0, i1 - i0)
        mask[j0:j1, i0:i1] ^= inside
    return mask


def xor_area(loopsA, loopsB, res):
    pts = [l for l in loopsA] + [l for l in loopsB]
    if not pts:
        return 0.0
    allp = np.vstack(pts)
    xmin, ymin = allp.min(axis=0) - 1.0
    xmax, ymax = allp.max(axis=0) + 1.0
    xs = np.arange(xmin + res / 2, xmax, res)
    ys = np.arange(ymin + res / 2, ymax, res)
    if xs.size == 0:
        xs = np.array([(xmin + xmax) / 2])
    if ys.size == 0:
        ys = np.array([(ymin + ymax) / 2])
    maskA = rasterize_mask(loopsA, xs, ys)
    maskB = rasterize_mask(loopsB, xs, ys)
    return float(np.logical_xor(maskA, maskB).sum()) * res * res


# --------------------------------------------------------------- circles --

def circle_fit(pts):
    """Algebraic (Kasa) least-squares circle fit. Returns cx, cy, r."""
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    rhs = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx = sol[0] / 2
    cy = sol[1] / 2
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    return float(cx), float(cy), r


def find_holes(loops, max_d=6.0):
    holes = []
    for l in loops:
        if len(l) < 3:
            continue
        cx, cy, r = circle_fit(l)
        radii = np.sqrt((l[:, 0] - cx) ** 2 + (l[:, 1] - cy) ** 2)
        max_resid = float(np.max(np.abs(radii - r)))
        if 2 * r < max_d and max_resid < 0.05:
            holes.append({
                'cx': cx, 'cy': cy,
                'fit_d': 2 * r,
                'vertex_d': float(2 * radii.mean()),
                'max_resid': max_resid,
            })
    holes.sort(key=lambda h: (round(h['cx'], 4), round(h['cy'], 4)))
    return holes


def pair_holes(holesA, holesB):
    remA = list(range(len(holesA)))
    remB = list(range(len(holesB)))
    pairs = []
    while remA and remB:
        best = None
        bestd = None
        for i in remA:
            for j in remB:
                d = ((holesA[i]['cx'] - holesB[j]['cx']) ** 2 +
                     (holesA[i]['cy'] - holesB[j]['cy']) ** 2) ** 0.5
                if bestd is None or d < bestd:
                    bestd = d
                    best = (i, j)
        i, j = best
        pairs.append((holesA[i], holesB[j], bestd))
        remA.remove(i)
        remB.remove(j)
    unmatchedA = [holesA[i] for i in remA]
    unmatchedB = [holesB[j] for j in remB]
    return pairs, unmatchedA, unmatchedB




# ------------------------------------------------------------------- cli --

def _finite(x):
    try:
        v = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {x!r}")
    if not np.isfinite(v):
        raise argparse.ArgumentTypeError(f"not finite: {x!r}")
    return v


def _positive(x):
    v = _finite(x)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0: {x!r}")
    return v


def parse_triplet(s):
    parts = [_finite(x) for x in s.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected dx,dy,dz, got {s!r}")
    return np.array(parts, dtype=float)


def parse_floats(s):
    return [_finite(x) for x in s.split(',') if x.strip() != '']


def _preprocess_argv(argv):
    """Merge '--shift-a VALUE' into '--shift-a=VALUE'.

    argparse treats a following token starting with '-' as a new option
    unless it matches a plain negative-number regex; "-219,-20.3,0" does
    not match that regex (it has commas and more than one number), so
    without this it is rejected as "expected one argument".
    """
    merge_opts = {'--shift-a', '--shift-b'}
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in merge_opts and i + 1 < len(argv):
            out.append(f"{a}={argv[i + 1]}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        epilog='Meshes with NaN or inf vertices, NaN or inf option values, '
               'and a --res that is not above 0 are refused with exit 2.')
    ap.add_argument('mesh_a', nargs='?', help='STL to check, or the reference when comparing')
    ap.add_argument('mesh_b', nargs='?', help='STL to compare against A (optional)')
    ap.add_argument('--shift-a', type=parse_triplet, default=np.zeros(3),
                    metavar='DX,DY,DZ', help='move A by this many mm before checking')
    ap.add_argument('--shift-b', type=parse_triplet, default=None,
                    metavar='DX,DY,DZ', help='move B by this many mm before checking')
    ap.add_argument('--z', type=parse_floats, default=None, metavar='Z1,Z2,...',
                    help='section heights, comma separated '
                         '(default: 8 evenly spaced over the z range)')
    ap.add_argument('--hole-z', type=_finite, default=None,
                    help='height at which round loops are fitted as holes '
                         '(default: middle of the z range)')
    ap.add_argument('--hole-max-d', type=_finite, default=6.0,
                    help='largest diameter in mm counted as a hole (default 6)')
    ap.add_argument('--res', type=_positive, default=0.02,
                    help='raster step in mm for section XOR areas (default 0.02)')
    ap.add_argument('--tol-area', type=_finite, default=0.5,
                    help='largest section XOR area in mm2 that passes (default 0.5)')
    ap.add_argument('--tol-hole', type=_finite, default=0.02,
                    help='largest hole centre offset and diameter change in mm that pass (default 0.02)')
    ap.add_argument('--strict', action='store_true',
                    help='exit 1 when the result is FAIL')
    ap.add_argument('--selftest', action='store_true',
                    help='run the built-in tests and exit')
    args = ap.parse_args(_preprocess_argv(sys.argv[1:]))
    if args.selftest:
        return selftest()
    if args.mesh_a is None:
        ap.error('mesh_a is required')
    if args.mesh_b is None and args.shift_b is not None:
        ap.error('--shift-b needs a second mesh')
    two = args.mesh_b is not None
    shift_b = args.shift_b if args.shift_b is not None else np.zeros(3)

    tA = load_stl(args.mesh_a) + args.shift_a
    tB = load_stl(args.mesh_b) + shift_b if two else None

    fails = 0
    warn_lines = []

    def mesh_block(label, path, t):
        vmin = t.reshape(-1, 3).min(axis=0)
        vmax = t.reshape(-1, 3).max(axis=0)
        size = vmax - vmin
        vol = volume(t)
        area = tri_areas(t).sum()
        bad = bad_edge_count(t)
        print(f"== {label}: {path} ==")
        print(f"triangles: {t.shape[0]}")
        print(f"bounds min: ({vmin[0]:.3f}, {vmin[1]:.3f}, {vmin[2]:.3f})")
        print(f"bounds max: ({vmax[0]:.3f}, {vmax[1]:.3f}, {vmax[2]:.3f})")
        print(f"size:       ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})")
        print(f"volume: {vol:.2f} mm3")
        print(f"surface area: {area:.2f} mm2")
        print(f"manifold: {bad} edge(s) not shared by exactly two triangles")
        return vol, bad

    volA, badA = mesh_block('A', args.mesh_a, tA)
    if two:
        print()
        volB, _ = mesh_block('B', args.mesh_b, tB)
        diff = volB - volA
        pct = (diff / volA * 100) if volA != 0 else float('nan')
        print(f"\nvolume diff B-A: {diff:.2f} mm3 ({pct:.2f} %)")

    # Heights not given are chosen from the z range of every loaded mesh.
    zs = tA[:, :, 2].ravel()
    if two:
        zs = np.concatenate([zs, tB[:, :, 2].ravel()])
    zmin = float(zs.min())
    H = float(zs.max()) - zmin
    auto = []
    if args.z is None:
        args.z = [zmin + (i + 0.5) * H / 8 for i in range(8)]
        auto.append("--z " + ",".join(f"{z:.3f}" for z in args.z))
    if args.hole_z is None:
        args.hole_z = zmin + H / 2
        auto.append(f"--hole-z {args.hole_z:.3f}")
    if auto:
        print(f"auto heights (z range {zmin:.3f} to {zmin + H:.3f}): "
              + "; ".join(auto))

    print("\n-- sections --")
    if two:
        print(f"{'z':>8} {'area A':>12} {'area B':>12} {'xor area':>12}  status")
    else:
        print(f"{'z':>8} {'area':>12} {'loops':>6}  status")
    for z in args.z:
        segsA = section_segments(tA, z)
        loopsA, wA = chain_loops(segsA)
        for w in wA:
            warn_lines.append(f"z={z}: A {w}")
        aA = section_area(loopsA)
        if not two:
            print(f"{z:8.3f} {aA:12.4f} {len(loopsA):6d}  ok")
            continue
        segsB = section_segments(tB, z)
        loopsB, wB = chain_loops(segsB)
        for w in wB:
            warn_lines.append(f"z={z}: B {w}")
        aB = section_area(loopsB)
        ax = xor_area(loopsA, loopsB, args.res)
        ok = ax <= args.tol_area
        if not ok:
            fails += 1
        print(f"{z:8.3f} {aA:12.4f} {aB:12.4f} {ax:12.4f}  {'ok' if ok else 'FAIL'}")

    print(f"\n-- holes at z={args.hole_z:.3f} --")
    segsA = section_segments(tA, args.hole_z)
    loopsA, wA = chain_loops(segsA)
    for w in wA:
        warn_lines.append(f"hole-z A: {w}")
    holesA = find_holes(loopsA, args.hole_max_d)
    holesB = []
    if two:
        segsB = section_segments(tB, args.hole_z)
        loopsB, wB = chain_loops(segsB)
        for w in wB:
            warn_lines.append(f"hole-z B: {w}")
        holesB = find_holes(loopsB, args.hole_max_d)

    def print_holes(label, holes):
        print(f"{label} holes ({len(holes)}):")
        for h in holes:
            print(f"  centre ({h['cx']:.3f}, {h['cy']:.3f})  "
                  f"vertex d {h['vertex_d']:.3f}  fit d {h['fit_d']:.3f}  "
                  f"(fit resid {h['max_resid']:.4f})")

    print_holes('A', holesA)
    if two:
        print_holes('B', holesB)

        if len(holesA) != len(holesB):
            print(f"hole count mismatch: A={len(holesA)} B={len(holesB)}  FAIL")
            fails += 1

        pairs, unA, unB = pair_holes(holesA, holesB)
        print("pairs:")
        for ha, hb, dist in pairs:
            ddelta = abs(ha['fit_d'] - hb['fit_d'])
            ok = dist <= args.tol_hole and ddelta <= args.tol_hole
            if not ok:
                fails += 1
            print(f"  A({ha['cx']:.3f},{ha['cy']:.3f}) <-> "
                  f"B({hb['cx']:.3f},{hb['cy']:.3f})  "
                  f"centre dist {dist:.4f}  diameter delta {ddelta:.4f}  "
                  f"{'ok' if ok else 'FAIL'}")
        for h in unA:
            print(f"  A({h['cx']:.3f},{h['cy']:.3f}) unmatched  FAIL")
            fails += 1
        for h in unB:
            print(f"  B({h['cx']:.3f},{h['cy']:.3f}) unmatched  FAIL")
            fails += 1
    elif badA > 0:
        print(f"\nmanifold: {badA} edge(s) not shared by exactly two "
              f"triangles  FAIL")
        fails += 1

    if warn_lines:
        print("\n-- warnings --")
        for w in warn_lines:
            print(w)

    if fails:
        print(f"\nRESULT: FAIL ({fails} items)")
    else:
        print("\nRESULT: PASS")

    if args.strict and fails:
        sys.exit(1)
    sys.exit(0)


# --------------------------------------------------------------- selftest --

def _quads_to_tris(quads):
    tris = []
    for q in quads:
        tris.append((q[0], q[1], q[2]))
        tris.append((q[0], q[2], q[3]))
    return np.array(tris, dtype=float)


def _make_box(x, y, z):
    """Closed box, 12 triangles, outward normals."""
    return _quads_to_tris([
        [(0, 0, 0), (0, y, 0), (x, y, 0), (x, 0, 0)],   # bottom
        [(0, 0, z), (x, 0, z), (x, y, z), (0, y, z)],   # top
        [(0, 0, 0), (x, 0, 0), (x, 0, z), (0, 0, z)],   # y = 0
        [(0, y, 0), (0, y, z), (x, y, z), (x, y, 0)],   # y = y
        [(0, 0, 0), (0, 0, z), (0, y, z), (0, y, 0)],   # x = 0
        [(x, 0, 0), (x, y, 0), (x, y, z), (x, 0, z)],   # x = x
    ])


def _make_holed_plate(size, height, hole_d, n=64):
    """size x size x height plate with a vertical through hole (n-gon) at
    the centre. The n circle angles are cast onto the square boundary, so
    the boundary vertices include the corners when n is a multiple of 8."""
    c = size / 2.0
    r = hole_d / 2.0
    tris = []
    ang = [2 * np.pi * i / n for i in range(n)]
    inner = []
    outer = []
    for a in ang:
        dx, dy = np.cos(a), np.sin(a)
        inner.append((c + r * dx, c + r * dy))
        t = c / max(abs(dx), abs(dy))
        ox = min(max(c + t * dx, 0.0), size)
        oy = min(max(c + t * dy, 0.0), size)
        outer.append((ox, oy))

    def v(p, z):
        return (p[0], p[1], z)

    for i in range(n):
        j = (i + 1) % n
        ci, cj, pi_, pj = inner[i], inner[j], outer[i], outer[j]
        # top face (z = height), ccw seen from above
        tris.append((v(ci, height), v(pi_, height), v(pj, height)))
        tris.append((v(ci, height), v(pj, height), v(cj, height)))
        # bottom face, reversed
        tris.append((v(ci, 0), v(pj, 0), v(pi_, 0)))
        tris.append((v(ci, 0), v(cj, 0), v(pj, 0)))
        # outer wall, outward normal
        tris.append((v(pi_, 0), v(pj, 0), v(pj, height)))
        tris.append((v(pi_, 0), v(pj, height), v(pi_, height)))
        # hole wall, normal into the hole
        tris.append((v(ci, 0), v(cj, height), v(cj, 0)))
        tris.append((v(ci, 0), v(ci, height), v(cj, height)))
    return np.array(tris, dtype=float)


def _write_binary_stl(path, tris):
    with open(path, 'wb') as f:
        f.write(b'binary stl written by check_mesh selftest'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        for t in tris:
            f.write(struct.pack('<3f', 0, 0, 0))
            for p in t:
                f.write(struct.pack('<3f', *p))
            f.write(struct.pack('<H', 0))


def _write_ascii_stl(path, tris):
    with open(path, 'w') as f:
        f.write('solid selftest\n')
        for t in tris:
            f.write('facet normal 0 0 0\n outer loop\n')
            for p in t:
                f.write('  vertex %r %r %r\n' % tuple(float(c) for c in p))
            f.write(' endloop\nendfacet\n')
        f.write('endsolid selftest\n')


def selftest():
    import re
    import subprocess
    import tempfile

    results = []

    def case(name, cond, detail=''):
        results.append((name, bool(cond)))
        print(('PASS  ' if cond else 'FAIL  ') + name
              + ('' if cond else '  ' + str(detail)))

    with tempfile.TemporaryDirectory() as d:
        def f(name):
            return os.path.join(d, name)

        box = _make_box(20, 20, 10)
        _write_binary_stl(f('box_bin.stl'), box)
        _write_ascii_stl(f('box_ascii.stl'), box)
        _write_binary_stl(f('box_moved.stl'), box + np.array([5.0, -3.0, 0.0]))
        plate = _make_holed_plate(20, 10, 5, 64)
        _write_binary_stl(f('plate.stl'), plate)
        _write_binary_stl(f('box_open.stl'), np.delete(box, 3, axis=0))

        def run(*argv):
            p = subprocess.run([sys.executable, os.path.abspath(__file__),
                                *argv], capture_output=True, text=True,
                               cwd=d)
            return p.returncode, p.stdout + p.stderr

        rc, out = run(f('box_bin.stl'), f('box_ascii.stl'), '--strict')
        case('identical box, binary vs ASCII, passes --strict',
             rc == 0 and 'RESULT: PASS' in out, f'rc={rc}\n{out}')

        rc, out = run(f('box_bin.stl'), f('box_moved.stl'),
                      '--shift-b', '-5,3,0', '--strict')
        case('translated box with matching --shift-b passes --strict',
             rc == 0 and 'RESULT: PASS' in out, f'rc={rc}\n{out}')

        rc, out = run(f('box_bin.stl'), f('plate.stl'), '--strict')
        case('box vs holed plate fails --strict with exit 1',
             rc == 1 and 'RESULT: FAIL' in out, f'rc={rc}\n{out}')

        rc, out = run(f('plate.stl'))
        m = re.search(r'A holes \((\d+)\):\n\s+centre \(([-\d.]+), ([-\d.]+)\)'
                      r'.*?fit d ([-\d.]+)', out)
        ok = (rc == 0 and m is not None and m.group(1) == '1'
              and abs(float(m.group(2)) - 10) < 0.05
              and abs(float(m.group(3)) - 10) < 0.05
              and abs(float(m.group(4)) - 5) < 0.1)
        case('holed plate, one mesh: one hole near (10, 10), fit d near 5',
             ok, f'rc={rc}\n{out}')

        rc, out = run(f('box_open.stl'), '--strict')
        m = re.search(r'manifold: (\d+) edge', out)
        case('open box, one mesh: manifold count > 0, RESULT FAIL, exit 1',
             rc == 1 and m is not None and int(m.group(1)) > 0
             and 'RESULT: FAIL' in out, f'rc={rc}\n{out}')

        rc, out = run(f('box_bin.stl'), '--shift-b', '1,0,0')
        case('--shift-b without a second mesh is a usage error, exit 2',
             rc == 2, f'rc={rc}\n{out}')

        big = _make_box(50, 50, 40)
        for tag, val in (('nan', np.nan), ('inf', np.inf)):
            badm = np.concatenate([big, np.full((1, 3, 3), val)])
            _write_binary_stl(f(f'big_{tag}.stl'), badm)
            rc, out = run(f('box_bin.stl'), f(f'big_{tag}.stl'), '--strict')
            case(f'compare, B has an all-{tag} triangle: exit 2, no PASS',
                 rc == 2 and 'RESULT: PASS' not in out, f'rc={rc}\n{out}')
            rc, out = run(f(f'big_{tag}.stl'), f('box_bin.stl'), '--strict')
            case(f'compare, A has an all-{tag} triangle: exit 2, no PASS',
                 rc == 2 and 'RESULT: PASS' not in out, f'rc={rc}\n{out}')

        _write_ascii_stl(f('nan_ascii.stl'),
                         np.concatenate([box, [[[np.nan, 0, 0], [1, 0, 0],
                                                [0, 1, 0]]]]))
        rc, out = run(f('nan_ascii.stl'))
        case('ASCII STL with a nan vertex: exit 2', rc == 2,
             f'rc={rc}\n{out}')

        rc, out = run(f('box_bin.stl'), '--z', 'nan')
        case('--z nan is a usage error, exit 2', rc == 2, f'rc={rc}\n{out}')

        rc, out = run(f('box_bin.stl'), '--res', '0')
        case('--res 0 is a usage error, exit 2', rc == 2, f'rc={rc}\n{out}')

        rc, out = run(f('box_bin.stl'))
        case('--z omitted prints an auto-height line',
             rc == 0 and 'auto heights' in out, f'rc={rc}\n{out}')

    bad = [name for name, ok in results if not ok]
    if bad:
        print(f"selftest FAILED: first failure: {bad[0]} "
              f"({len(bad)} of {len(results)} failed)")
        return 1
    print(f"selftest OK: {len(results)} cases")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
