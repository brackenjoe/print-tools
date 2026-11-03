# print-tools

Command-line tools for checking 3D-printable parts. Each tool is one Python
file: clone the repo and run it with `python3`.

| tool | what it does | needs |
|---|---|---|
| `check_mesh.py` | Checks one STL, or compares two. It reports triangle count, bounds, volume, surface area and open edges, then measures horizontal sections at chosen heights and fits the small round loops it finds at one height. Given two meshes, it measures how far each section differs, pairs the holes by position and compares their diameters. Use it to check that a rebuilt model matches the mesh it was rebuilt from, or that a mounting face survived an edit. | Python 3, `pip install numpy matplotlib` |
| `scad-render.py` | Renders the STLs and PNG previews of one OpenSCAD model from a short manifest, and fails any output that is not real. A failed `assert()` in a PNG render, or a part that draws nothing, still exits 0 and writes a blank image (OpenSCAD 2026.10.29 nightly); this tool reads each render's log and checks every file it writes, down to the pixels. | Python 3 (stdlib only), OpenSCAD, `xvfb-run` for PNGs on a machine with no display |

## check_mesh.py

    python3 check_mesh.py A.stl [B.stl] [options]
    python3 check_mesh.py --selftest

| option | default | meaning |
|---|---|---|
| `--z Z1,Z2,...` | 8 heights, evenly spaced | section heights in mm |
| `--hole-z Z` | mid-height | height at which holes are fitted |
| `--hole-max-d D` | 6 | largest diameter in mm counted as a hole |
| `--shift-a DX,DY,DZ`, `--shift-b DX,DY,DZ` | none | translate a mesh before checking (no rotation); a negative value works as given, e.g. `--shift-a -219,-20.3,0` |
| `--tol-area A` | 0.5 | largest section XOR area in mm² that passes |
| `--tol-hole T` | 0.02 | largest hole centre offset, and diameter change, in mm that passes |
| `--res R` | 0.02 | raster step in mm for the XOR area; must be above 0 |
| `--strict` | off | exit 1 when the result is FAIL |

Binary and ASCII STL are both read. Units are the file's, taken as mm.

**What is judged.** Only the sections and the holes in two-mesh mode, and
the open edges in one-mesh mode, decide PASS or FAIL. Triangles, bounds,
volume and surface area are printed for you to read; a part 0.3 mm taller
than its reference still passes if no chosen height crosses the difference.
Pick heights that cross the features you care about.

**One mesh.** It prints the stats, then each section's area and loop count
(reported, not judged), then the holes at `--hole-z`. An open edge, one not
shared by exactly two triangles (the `manifold:` line), makes the result
FAIL: such a mesh is not a closed solid, and a slicer may repair it in ways
you did not intend.

**Two meshes.** It prints both meshes' stats and the volume difference. At
each height it gives both section areas and their XOR area: the area inside
one section but not the other, rasterised at `--res`. A height passes when
the XOR area is at most `--tol-area`. Holes are paired by nearest centre. A
pair passes when its centres are within `--tol-hole` and its `fit d` values
differ by at most that much. An unpaired hole fails, and unequal hole counts
fail as a separate item, so one missing hole counts twice in the FAIL total.
Open edges are reported here, not judged: a mesh downloaded from elsewhere
is often not closed, and that is not the question being asked.

**Holes.** A hole is any closed section loop within 0.05 mm of a circle and
smaller than `--hole-max-d`. A round pin or boss is listed too, since the
tool does not tell inside from outside. Each line gives `vertex d`, the
diameter through the loop's vertices; `fit d`, the least-squares circle's
diameter, which is the one compared; and `fit resid`, the worst distance of
a vertex from that circle. Bores of `--hole-max-d` or more are not listed.
Only the section XOR covers them, unless you raise the limit.

**Heights.** Left out, the tool takes 8 evenly spaced heights over the
meshes' combined height, and mid-height for holes, and prints a line saying
which it used. A section exactly at a horizontal face is ambiguous, so
avoid heights where a face sits. A section with an open loop is noted under
warnings; that does not change the result.

**Large parts.** The XOR area is computed on a full raster grid, so memory
grows with the square of part width over `--res`. A 150 mm part at the
default 0.02 took about 2.3 GB. For parts over about 100 mm, use `--res
0.05` or coarser.

**Exit status.** 0 on PASS, and also on FAIL unless `--strict`. 1 on FAIL
with `--strict`, or on an unexpected error such as running out of memory. 2
on a usage error or an unreadable file. A mesh with any NaN or infinite
vertex coordinate is refused with exit 2, as is a NaN or infinite option
value or a `--res` that is not above 0. A single NaN vertex would otherwise
make every default section height NaN, and two different parts would compare
as a PASS. `--selftest` builds a few meshes in
a temporary directory, checks them and exits 0 only if every case passes.

## scad-render.py

    python3 scad-render.py MODEL.scad MANIFEST [-D 'name=value']...
    python3 scad-render.py --selftest

The manifest lists one output per line; `#` starts a comment:

    # kind  part      file              extra
    stl     body      body.stl          BODY
    png     body      body.png          0,0,0,55,0,35,0
    png     assembly  assembly_top.png  0,0,0,0,0,0,0  o
    png     section   section.png       -4,0,0,0,0,0,125  o fixed

Each render gets the define `part="PART"`, so the model chooses what to draw
from a `part` variable; a part of `-` passes no such define. For an `stl`
line, the optional last field is a tag: each distinct `ECHO:` line
containing it is printed once, so a model can report its own measurements.
For a `png` line, the camera is OpenSCAD's `--camera` value (6 or 7
numbers), then up to two words in either order: `p` or `o` (perspective,
the default, or orthographic) and `fit` or `fixed`. `fit`, the default,
adds `--viewall --autocenter`, so OpenSCAD frames the part itself; `fixed`
keeps the camera as given. A manifest of `-` is read from standard input,
so a script can compute cameras and pipe them in.

Paths in the manifest are relative to the model's directory, and OpenSCAD
runs there. An output `file` is refused, as a bad manifest line, if it is
absolute, if its directory resolves (symlinks followed) outside the model's
directory, if it is the model itself, if it names an existing directory, or
if it does not end in `.stl` for an `stl` line or `.png` for a `png` line
(any case).
Every listed file is deleted before the first render, so a manifest you did
not write can still remove `.stl` and `.png` files under the model's
directory. Run a manifest you did not write only against a model in a directory whose
`.stl` and `.png` files you can lose. Every `-D` on the command line goes to every render, after the
part define. Its value is OpenSCAD syntax, so a string needs quotes that
survive the shell: `-D 'label="v2"'`, not `-D label=v2`.

| option | default | meaning |
|---|---|---|
| `--openscad BIN` | `$OPENSCAD`, else `openscad`, else `openscad-nightly` | the OpenSCAD binary |
| `--imgsize W,H` | 1400,1050 | PNG size in pixels |
| `--colorscheme NAME` | Tomorrow | OpenSCAD colour scheme for PNGs |
| `--min-png BYTES` | 0 (off) | fail PNGs smaller than this |
| `--no-xvfb` | off | render PNGs without `xvfb-run`, for a machine with a display |

It checks the whole manifest before rendering anything; a bad line is
reported with its line number. Then it deletes every listed output and
renders them in order. PNGs render under `xvfb-run -a` unless `--no-xvfb`,
with `LIBGL_ALWAYS_SOFTWARE=1` set. Every distinct `WARNING:` and `ERROR:`
line from any render is printed once. Those lines, tagged `ECHO:` lines,
failure reasons and file names are printed with control characters shown as
`\xNN` text, so a model or manifest cannot drive your terminal. Each output then prints `ok` or
`FAIL` with the reason. An output fails, and is deleted, when:

- its log has an `ERROR:` line, or says the top-level object is empty (what
  an STL render of nothing logs);
- OpenSCAD exits non-zero, or the file is missing;
- an STL is not a valid binary or ASCII STL;
- a PNG is not a PNG, is one colour throughout (a blank render), or is
  smaller than `--min-png`.

**Exit status.** 0 if every output is ok, 1 if any fails, 2 for a bad
manifest or arguments, or a missing model, OpenSCAD or `xvfb-run`.
`--selftest` renders a tiny model in a temporary directory, including a
failing assertion and an empty part. It needs OpenSCAD and `xvfb-run`,
exits 2 without them, and exits 0 only if every case behaves.

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE).
