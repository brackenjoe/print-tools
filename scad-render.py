#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joe Fernandez
"""Render STLs and PNG previews from one OpenSCAD model, and check every output.

  scad-render.py MODEL.scad MANIFEST [-D name=value]... [--openscad BIN]
                 [--imgsize W,H] [--colorscheme NAME] [--min-png BYTES]
                 [--no-xvfb]
  scad-render.py --selftest

MANIFEST is a file, or `-` for stdin. One output per line, fields separated by
whitespace; blank lines and `#` comments (to end of line) are ignored:

  stl  PART  OUT.stl  [TAG]
  png  PART  OUT.png  CAMERA  [p|o]  [fit|fixed]

PART is passed as `-D part="PART"`; PART `-` passes no part define. CAMERA is
OpenSCAD's --camera value (6 or 7 comma-separated numbers). After CAMERA come
zero to two optional tokens, in either order, at most one of each kind: p|o
(projection, default p, perspective) and fit|fixed (default fit, which adds
--viewall --autocenter; fixed adds neither). So `CAM`, `CAM o`, `CAM fixed`,
`CAM fixed o` and `CAM o fixed` are all valid. OpenSCAD's working directory
is the MODEL's directory, and OUT is a path relative to it. An OUT is refused
(exit 2, nothing deleted) if it is absolute, if its directory resolves (symlinks
followed) outside the model's directory, if it is the model file itself, if
it is an existing directory, or if it does not end in the job's extension
(.stl or .png, any case). Every OUT is deleted before the first render, so the
most a manifest can remove is .stl and .png files under the model's directory:
run a manifest you did not write only against a model in a directory whose
.stl and .png files you can lose.

Each -D on the command line goes to every render, after the part define,
verbatim. Its value is OpenSCAD syntax, so a string needs shell quoting:
-D 'part="body"'.

The whole manifest is checked before anything is rendered or deleted; a bad
line gives its line number and exit 2. Then every OUT is deleted and the
renders run in manifest order. From STL renders, each distinct `ECHO:` line
containing TAG is printed once per run; every distinct WARNING:/ERROR: line
from any render is printed once.

An output FAILS (and is deleted) if its render log has a line starting
`ERROR:` or containing `top level object is empty` (what an STL render logs
when the part selects nothing), the exit status is non-zero, the file is
missing, an STL is not a valid binary or ASCII STL, a PNG lacks the PNG
signature or is smaller than --min-png (default 0, off), or a PNG is blank:
every pixel one colour (`blank image`). The log check is needed: a failed
assert() in a PNG render exits 0 and writes a blank PNG, and so does a PNG
render of a part that draws nothing, with no log line at all (seen with the
OpenSCAD 2026.10.29 nightly). A blank PNG is the same size as a flat-colour
real one, so size cannot tell them apart; the check decodes the PNG with
zlib. Only 8-bit, non-interlaced PNGs are decoded; any other PNG skips the
blank check. PNG renders need a display, so they run under `xvfb-run -a`
unless --no-xvfb (without one the render fails, and some OpenSCAD builds
still exit 0). Software GL is forced with LIBGL_ALWAYS_SOFTWARE=1.

Log lines, reasons and OUT names are printed with control characters
(U+0000-U+001F, U+007F, U+0080-U+009F) shown as \\xNN text, so a model or
manifest cannot drive the terminal.

Exit 0 if every output is ok, 1 if any FAIL, 2 for a bad manifest, a missing
model, OpenSCAD or xvfb-run, or bad arguments.

Stdlib only.
"""
import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

PNG_SIG = b'\x89PNG\r\n\x1a\n'
LOG_FAIL = re.compile(r'^ERROR:|top level object is empty')


class ManifestError(Exception):
    pass


def inert(s):
    """s with C0, DEL and C1 control characters shown as \\xNN text."""
    return ''.join(f'\\x{ord(c):02x}' if ord(c) < 32 or 127 <= ord(c) < 160
                   else c for c in s)


def parse_manifest(text, base, model):
    """Return a list of job dicts; raise ManifestError naming the line."""
    jobs, seen = [], {}
    rbase = os.path.realpath(base)
    rmodel = os.path.realpath(model)
    for n, raw in enumerate(text.splitlines(), 1):
        f = raw.split('#', 1)[0].split()
        if not f:
            continue

        def bad(msg):
            raise ManifestError(f'manifest line {n}: {msg}')

        kind = f[0]
        if kind == 'stl':
            if len(f) not in (3, 4):
                bad(f'stl takes PART OUT [TAG], got {len(f) - 1} fields')
            job = {'kind': 'stl', 'part': f[1], 'out': f[2],
                   'tag': f[3] if len(f) == 4 else None}
        elif kind == 'png':
            if len(f) < 4:
                bad(f'png takes PART OUT CAMERA [p|o] [fit|fixed], '
                    f'got {len(f) - 1} fields')
            cam = f[3].split(',')
            try:
                [float(c) for c in cam]
            except ValueError:
                bad(f'camera {f[3]!r} is not comma-separated numbers')
            if len(cam) not in (6, 7):
                bad(f'camera {f[3]!r} has {len(cam)} numbers, need 6 or 7')
            proj, fit = 'p', 'fit'
            if len(f) > 6:
                bad(f'png takes PART OUT CAMERA [p|o] [fit|fixed], '
                    f'got {len(f) - 1} fields')
            seen_p = seen_f = False
            for extra in f[4:]:
                if extra in ('p', 'o'):
                    if seen_p:
                        bad(f'projection given twice ({extra!r}); '
                            f'at most one of p|o')
                    seen_p, proj = True, extra
                elif extra in ('fit', 'fixed'):
                    if seen_f:
                        bad(f'fit mode given twice ({extra!r}); '
                            f'at most one of fit|fixed')
                    seen_f, fit = True, extra
                else:
                    bad(f'{extra!r} is not p, o, fit or fixed')
            job = {'kind': 'png', 'part': f[1], 'out': f[2], 'camera': f[3],
                   'proj': proj, 'fit': fit}
        else:
            bad(f'unknown kind {kind!r} (expected stl or png)')
        out = job['out']
        if os.path.isabs(out):
            bad(f"OUT '{inert(out)}' is absolute; use a path relative to "
                f"the model's directory")
        path = os.path.normpath(os.path.join(base, out))
        rparent = os.path.realpath(os.path.dirname(path))
        try:
            inside = os.path.commonpath([rbase, rparent]) == rbase
        except ValueError:
            inside = False
        if not inside:
            bad(f"OUT '{inert(out)}' is outside the model's directory")
        if os.path.realpath(path) == rmodel:
            bad(f"OUT '{inert(out)}' is the model file")
        if os.path.isdir(path):
            bad(f"OUT '{inert(out)}' is an existing directory")
        if not out.lower().endswith('.' + kind):
            bad(f"OUT '{inert(out)}' does not end in .{kind}; a {kind} "
                f"manifest line writes a .{kind} file")
        if path in seen:
            bad(f'duplicate OUT {inert(out)} (first on line {seen[path]})')
        seen[path] = n
        job['path'] = path
        job['line'] = n
        jobs.append(job)
    if not jobs:
        raise ManifestError('manifest has no render lines')
    return jobs


def stl_problem(path):
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as fh:
            head = fh.read(84)
            if len(head) == 84:
                n = struct.unpack('<I', head[80:84])[0]
                if n > 0 and size == 84 + 50 * n:
                    return None
            fh.seek(0)
            data = fh.read()
    except OSError as e:
        return f'cannot read STL: {e}'
    if data.lstrip().startswith(b'solid') and b'facet' in data:
        return None
    return 'not a valid STL (no binary size match, no ASCII solid/facet)'


def png_blank(data):
    """True if every pixel is one colour, False if not, None if undecidable
    (not 8-bit non-interlaced, or damaged): the caller then skips the check."""
    try:
        if data[:8] != PNG_SIG:
            return None
        pos, idat, hdr = 8, [], None
        while pos + 8 <= len(data):
            ln, typ = struct.unpack('>I4s', data[pos:pos + 8])
            body = data[pos + 8:pos + 8 + ln]
            pos += 12 + ln
            if typ == b'IHDR':
                hdr = struct.unpack('>IIBBBBB', body)
            elif typ == b'IDAT':
                idat.append(body)
            elif typ == b'IEND':
                break
        if hdr is None or not idat:
            return None
        w, h, depth, ctype, _, _, interlace = hdr
        bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype)
        if depth != 8 or interlace != 0 or bpp is None or not w or not h:
            return None
        raw = zlib.decompress(b''.join(idat))
        stride = w * bpp
        if len(raw) != h * (stride + 1):
            return None
        first = None
        prev = bytearray(stride)
        for y in range(h):
            o = y * (stride + 1)
            ft = raw[o]
            row = bytearray(raw[o + 1:o + 1 + stride])
            if ft == 0:
                pass
            elif ft == 1:
                for i in range(bpp, stride):
                    row[i] = (row[i] + row[i - bpp]) & 255
            elif ft == 2:
                row = bytearray((x + u) & 255 for x, u in zip(row, prev))
            elif ft == 3:
                for i in range(stride):
                    left = row[i - bpp] if i >= bpp else 0
                    row[i] = (row[i] + ((left + prev[i]) >> 1)) & 255
            elif ft == 4:
                for i in range(stride):
                    if i >= bpp:
                        a, c = row[i - bpp], prev[i - bpp]
                    else:
                        a = c = 0
                    b = prev[i]
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    row[i] = (row[i] + pr) & 255
            else:
                return None
            if first is None:
                first = bytes(row[:bpp]) * w
            if bytes(row) != first:
                return False
            prev = row
        return True
    except (struct.error, zlib.error, IndexError, ValueError):
        return None


def png_problem(path, min_png):
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as fh:
            data = fh.read()
    except OSError as e:
        return f'cannot read PNG: {e}'
    if data[:8] != PNG_SIG:
        return 'not a PNG (bad signature)'
    if size < min_png:
        return f'PNG only {size} bytes, under --min-png {min_png}'
    if png_blank(data):
        return 'blank image: every pixel is one colour'
    return None


def find_openscad(arg):
    cand = arg or os.environ.get('OPENSCAD')
    if cand:
        return shutil.which(cand)
    for name in ('openscad', 'openscad-nightly'):
        p = shutil.which(name)
        if p:
            return p
    return None


def run_jobs(model, jobs, binp, extras, imgsize, scheme, min_png, use_xvfb, out):
    """Render and check. Returns the number of failed outputs."""
    base = os.path.dirname(os.path.abspath(model))
    modelname = os.path.basename(model)
    for j in jobs:
        try:
            os.remove(j['path'])
        except FileNotFoundError:
            pass
    shown, results = set(), []
    for j in jobs:
        pdef = [] if j['part'] == '-' else ['-D', f'part="{j["part"]}"']
        dflags = pdef + [x for e in extras for x in ('-D', e)]
        env = dict(os.environ)
        if j['kind'] == 'stl':
            cmd = [binp, '-o', j['path']] + dflags + [modelname]
        else:
            env['LIBGL_ALWAYS_SOFTWARE'] = '1'
            cmd = [binp, '-o', j['path'], f'--imgsize={imgsize}',
                   f'--camera={j["camera"]}']
            if j['fit'] == 'fit':
                cmd += ['--viewall', '--autocenter']
            cmd += [f'--projection={j["proj"]}', f'--colorscheme={scheme}',
                    '--render=true'] + dflags + [modelname]
            if use_xvfb:
                cmd = ['xvfb-run', '-a'] + cmd
        try:
            p = subprocess.run(cmd, cwd=base, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               errors='replace')
            log, rc = p.stdout, p.returncode
        except OSError as e:
            log, rc = '', None
            results.append((j, f'could not run: {inert(str(e))}'))
            continue
        lines = log.splitlines()
        for ln in lines:
            keep = bool(re.match(r'(WARNING|ERROR):', ln))
            if (j['kind'] == 'stl' and j['tag'] and ln.startswith('ECHO:')
                    and j['tag'] in ln):
                keep = True
            if keep and ln not in shown:
                shown.add(ln)
                print(inert(ln), file=out)
        reason = None
        bad = next((ln for ln in lines if LOG_FAIL.search(ln)), None)
        if bad:
            reason = f'render log: {inert(bad.strip())}'
        elif rc != 0:
            reason = f'exit status {rc}'
        elif not os.path.exists(j['path']):
            reason = 'output missing'
        else:
            reason = (stl_problem(j['path']) if j['kind'] == 'stl'
                      else png_problem(j['path'], min_png))
        results.append((j, reason))
    fails = 0
    for j, reason in results:
        if reason:
            fails += 1
            try:
                os.remove(j['path'])
            except FileNotFoundError:
                pass
            print(f'FAIL {inert(j["out"])} {reason}', file=out)
        else:
            print(f'ok   {inert(j["out"])} {os.path.getsize(j["path"])} bytes',
                  file=out)
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', nargs='?', help='the .scad file')
    ap.add_argument('manifest', nargs='?', help='the manifest file, or - for stdin')
    ap.add_argument('-D', dest='defs', action='append', default=[],
                    metavar='name=value', help='extra OpenSCAD define, every render')
    ap.add_argument('--openscad', metavar='BIN', help='OpenSCAD binary (default: $OPENSCAD, '
                    'else openscad, else openscad-nightly)')
    ap.add_argument('--imgsize', default='1400,1050', metavar='W,H',
                    help='PNG size in pixels (default 1400,1050)')
    ap.add_argument('--colorscheme', default='Tomorrow', metavar='NAME',
                    help='OpenSCAD colour scheme for PNGs (default Tomorrow)')
    ap.add_argument('--min-png', type=int, default=0, metavar='BYTES',
                    help='fail PNGs smaller than this (default 0, off)')
    ap.add_argument('--no-xvfb', action='store_true',
                    help='do not wrap PNG renders in xvfb-run')
    ap.add_argument('--selftest', action='store_true',
                    help='run the built-in tests (needs OpenSCAD and xvfb-run) and exit')
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.openscad)
    if not a.model or not a.manifest:
        ap.error('MODEL and MANIFEST are required')
    if not os.path.isfile(a.model):
        print(f'model not found: {a.model}', file=sys.stderr)
        return 2
    binp = find_openscad(a.openscad)
    if not binp:
        print('OpenSCAD not found (use --openscad, $OPENSCAD, or put openscad '
              'or openscad-nightly on PATH)', file=sys.stderr)
        return 2
    try:
        if a.manifest == '-':
            text = sys.stdin.read()
        else:
            with open(a.manifest) as fh:
                text = fh.read()
    except OSError as e:
        print(f'cannot read manifest: {e}', file=sys.stderr)
        return 2
    base = os.path.dirname(os.path.abspath(a.model))
    try:
        jobs = parse_manifest(text, base, a.model)
    except ManifestError as e:
        print(e, file=sys.stderr)
        return 2
    use_xvfb = not a.no_xvfb
    if use_xvfb and any(j['kind'] == 'png' for j in jobs) \
            and not shutil.which('xvfb-run'):
        print('xvfb-run not found; PNG renders need a display '
              '(install xvfb, or pass --no-xvfb if one is available)',
              file=sys.stderr)
        return 2
    fails = run_jobs(a.model, jobs, binp, a.defs, a.imgsize, a.colorscheme,
                     a.min_png, use_xvfb, sys.stdout)
    return 1 if fails else 0


SELFTEST_MODEL = '''part = "a";
assert(part != "bad", "boom");
echo("TAGX size 10");
if (part != "none") cube(10);
'''


def selftest(openscad_arg=None):
    if not find_openscad(openscad_arg) or not shutil.which('xvfb-run'):
        print('selftest needs openscad and xvfb-run on PATH', file=sys.stderr)
        return 2
    me = os.path.abspath(__file__)
    results = []

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'm.scad'), 'w') as fh:
            fh.write(SELFTEST_MODEL)
        model = os.path.join(d, 'm.scad')
        cam = '0,0,0,55,0,25,0'
        fcam = '5,5,5,55,0,25,60'  # fixed has no autofit: needs a distance

        def tool(manifest, *args, stdin=None, mdl=None):
            if stdin is None:
                mp = os.path.join(d, 'manifest')
                with open(mp, 'w') as fh:
                    fh.write(manifest)
                inp, mf = None, mp
            else:
                inp, mf = stdin, '-'
            p = subprocess.run([sys.executable, me, mdl or model, mf] + list(args)
                               + (['--openscad', openscad_arg] if openscad_arg else []),
                               input=inp, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
            return p.returncode, p.stdout + p.stderr

        def has(name):
            return os.path.exists(os.path.join(d, name))

        def case(name, cond, detail=''):
            results.append(cond)
            print(('PASS  ' if cond else 'FAIL  ') + name
                  + ('' if cond else '  ' + detail.strip().replace('\n', ' | ')))

        rc, o = tool(f'stl a a.stl TAGX\nstl a a2.stl TAGX\npng a a.png {cam}\n')
        case('stl+png ok, TAGX echo once',
             rc == 0 and o.count('ok   a.stl') == 1 and o.count('ok   a.png') == 1
             and o.count('TAGX size 10') == 1 and has('a.png'), f'rc={rc} {o}')

        rc, o = tool(f'png bad b.png {cam}\n')
        case('png assertion fails, file removed',
             rc == 1 and 'FAIL b.png' in o and 'boom' in o and not has('b.png'),
             f'rc={rc} {o}')

        rc, o = tool('stl bad b.stl\n')
        case('stl assertion fails',
             rc == 1 and 'FAIL b.stl' in o and not has('b.stl'), f'rc={rc} {o}')

        rc, o = tool('', stdin='# from a wrapper\n\nstl a s.stl\n')
        case('manifest from stdin',
             rc == 0 and 'ok   s.stl' in o and has('s.stl'), f'rc={rc} {o}')

        with open(os.path.join(d, 'keep.stl'), 'w') as fh:
            fh.write('keep')
        rc, o = tool('stl a keep.stl\npng a x.png 1,2,3\n')
        with open(os.path.join(d, 'keep.stl')) as fh:
            kept = fh.read() == 'keep'
        case('bad manifest line: exit 2, existing OUT untouched',
             rc == 2 and 'line 2' in o and kept and not has('x.png'), f'rc={rc} {o}')

        rc, o = tool('stl - p.stl\n', '-D', 'part="a"')
        rc2, o2 = tool('stl - p2.stl\n', '-D', 'part="bad"')
        case('-D passthrough (part "-" takes the define)',
             rc == 0 and 'ok   p.stl' in o and rc2 == 1 and 'FAIL p2.stl' in o2,
             f'rc={rc} {o} / rc={rc2} {o2}')

        rc, o = tool(f'png none n.png {cam}\n')
        case('png of a part that draws nothing: FAIL blank, file removed',
             rc == 1 and 'FAIL n.png blank image' in o and not has('n.png'),
             f'rc={rc} {o}')

        rc, o = tool(f'png a f1.png {fcam} fixed\npng a f2.png {fcam} fixed o\n'
                     f'png a f3.png {fcam} o fixed\n')
        case('fixed, fixed o and o fixed parse and render',
             rc == 0 and o.count('ok   f') == 3, f'rc={rc} {o}')
        rc, o = tool(f'png a g.png {cam} o p\n')
        rc2, o2 = tool(f'png a g.png {cam} fit fixed\n')
        case('two tokens of one kind: exit 2 with line number',
             rc == 2 and 'line 1' in o and rc2 == 2 and 'line 1' in o2,
             f'rc={rc} {o} / rc={rc2} {o2}')

        rc, o = tool(f'png a r.png {cam}\n')
        case('real png still ok', rc == 0 and 'ok   r.png' in o and has('r.png'),
             f'rc={rc} {o}')

        # OUT containment: each refusal is exit 2 and deletes nothing.
        sub = os.path.join(d, 'sub')
        os.mkdir(sub)
        outside = os.path.join(d, 'outside.stl')
        far = os.path.join(d, 'far')
        os.mkdir(far)
        victim = os.path.join(far, 'x.stl')
        for pth in (outside, victim):
            with open(pth, 'w') as fh:
                fh.write('victim')
        mdir = os.path.join(d, 'mdir')
        os.mkdir(mdir)
        mpath = os.path.join(mdir, 'm.scad')
        shutil.copy(model, mpath)
        os.symlink(far, os.path.join(mdir, 'link'))
        os.mkdir(os.path.join(mdir, 'adir'))
        with open(os.path.join(mdir, 'keep.stl'), 'w') as fh:
            fh.write('keep')

        def alive(pth):
            try:
                with open(pth) as fh:
                    return fh.read() in ('victim', 'keep', SELFTEST_MODEL)
            except OSError:
                return False

        rc, o = tool('stl a keep.stl\nstl a ../outside.stl\n', mdl=mpath)
        case('OUT ../ escape: exit 2, file and earlier OUT survive',
             rc == 2 and 'outside' in o and alive(outside)
             and alive(os.path.join(mdir, 'keep.stl')), f'rc={rc} {o}')
        rc, o = tool(f'stl a {outside}\n', mdl=mpath)
        case('absolute OUT: exit 2, file survives',
             rc == 2 and 'absolute' in o and alive(outside), f'rc={rc} {o}')
        rc, o = tool('stl a link/x.stl\n', mdl=mpath)
        case('OUT through a directory symlink out of the model dir: exit 2',
             rc == 2 and 'outside' in o and alive(victim), f'rc={rc} {o}')
        m2 = os.path.join(mdir, 'm2.scad')
        shutil.copy(model, m2)
        rc, o = tool('stl a m2.scad\n', mdl=m2)
        case('OUT equal to the model: exit 2, model survives',
             rc == 2 and 'model file' in o and alive(m2), f'rc={rc} {o}')
        rc, o = tool('stl a adir\n', mdl=mpath)
        case('OUT naming an existing directory: exit 2',
             rc == 2 and 'directory' in o and os.path.isdir(os.path.join(mdir, 'adir')),
             f'rc={rc} {o}')

        with open(os.path.join(mdir, 'notes.txt'), 'w') as fh:
            fh.write('keep')
        rc, o = tool('stl a notes.txt\n', mdl=mpath)
        case('OUT without the job\'s extension: exit 2, the file survives',
             rc == 2 and '.stl' in o and alive(os.path.join(mdir, 'notes.txt')),
             f'rc={rc} {o}')
        rc, o = tool('png a pic.stl 0,0,0,55,0,25,0\n', mdl=mpath)
        case('png line with a .stl OUT: exit 2', rc == 2 and '.png' in o, f'rc={rc} {o}')

        # Inert terminal output.
        emodel = os.path.join(d, 'esc.scad')
        with open(emodel, 'w') as fh:
            fh.write('echo(str("TAG ", chr(27), "[2K"));\ncube(1);\n')
        rc, o = tool('stl - e.stl TAG\n', mdl=emodel)
        case('ECHO line with ESC prints as \\x1b text, no raw ESC',
             rc == 0 and '\\x1b[2K' in o and '\x1b' not in o, f'rc={rc} {o!r}')
        rc, o = tool('stl - bad\x1b[2Kname.stl\n', mdl=emodel)
        case('OUT with a control character: no raw ESC in output',
             '\x1b' not in o, f'rc={rc} {o!r}')

    for ok in results:
        if not ok:
            print('selftest FAILED', file=sys.stderr)
            return 1
    print(f'selftest: {len(results)} cases passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
