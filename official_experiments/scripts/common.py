"""What every script in this directory needs, in one place.

Before the cleanup these constants lived in ``src/nspm/visualization``, and the
scripts reached them through a stub that faked the ``nspm`` package so that
importing a palette would not drag in torch and pm4py. The stub was repeated in
three files and the palette was repeated in none of them only because the import
hid it. Both are gone: the drawing constants live here, beside the figures they
draw, and ``official_experiments/`` no longer depends on ``src/`` to produce a
figure.

Two scripts still import from ``src/``, and for a reason that is not style:
``dataset_table.py`` reads the event logs through ``nspm.data.loader`` and
``artifacts_table.py`` mines the symbolic artifacts through the experiment code.
Those two describe the inputs of the experiment, so they have to use the same
code the experiment used.

The palette is the one the thesis figures have always used. A model keeps its
colour across every figure, which is why the mapping is by model name and not by
position: a chart that reorders its series must not recolour them.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

FIGURES = ROOT / "official_experiments" / "figures"
GRID_CSV = ROOT / "official_experiments" / "all_grids.csv"

#: One directory per knowledge source, which is what the protocols differ in.
#: The letters B and C are how the runs were launched and survive only in the
#: grids; these are the names the thesis uses.
PROTOCOL_DIR = {"B": "protocol-test", "C": "protocol-train"}

#: The thesis reads its figures from its own directory, so every figure exists
#: twice. The copy is made by :func:`copy_to_thesis` rather than by hand: a
#: figure regenerated in one place and forgotten in the other is a figure that
#: disagrees with the table beside it, and nothing in the build would say so.
THESIS_IMAGES = ROOT / "docs" / "Thesis" / "images"

PROTOCOLS = {"B": "test", "C": "train"}

#: CSV key -> the name of the model in the thesis, in the order of the legend:
#: the baseline, the four losses, the three features.
VARIANTS = {
    "baseline": "Baseline",
    "lll": "Local loss",
    "gll": "Global loss",
    "checker_net": "Projected net",
    "checker_net_state": "State-indexed net",
    "marking": "Marking",
    "gnn": "Graph encoder",
    "seq": "Marking sequence",
}

#: Panel and column order. BPIC 2012 is last because it is the log the reference
#: work never saw.
DATASET_ORDER = ("Sepsis_Case", "BPIC_2013_incidents",
                 "BPIC_2020_DomesticDeclarations", "BPI_Challenge_2012")

#: Short name of a log, for a panel label or a column header.
SHORT = {
    "Sepsis_Case": "Sepsis",
    "BPIC_2013_incidents": "BPIC13",
    "BPIC_2020_DomesticDeclarations": "BPIC20",
    "BPI_Challenge_2012": "BPIC12",
}

#: Model -> colour. Fixed categorical slots: the colour follows the model, not
#: its rank. The two methods taken from Mezini et al. inject their knowledge
#: through the objective, like the two net-derived penalties, so they sit on the
#: warm side of the palette with them rather than on the green of the features:
#: the hue names the channel. ``checker`` is the directly-follows variant, which
#: is out of the thesis but still in the grids.
COLORS = {
    "baseline": "#2a78d6",           # blue
    "checker": "#eb6834",            # orange
    "checker_net": "#4a3aa7",        # violet
    "checker_net_state": "#008300",  # green
    "marking": "#1baf7a",            # aqua
    "gnn": "#eda100",                # yellow
    "seq": "#e87ba4",                # magenta
    "lll": "#a8201a",                # dark red
    "gll": "#7d4f00",                # brown
}

#: Secondary encoding: the shape separates the series in black and white, under
#: colour vision deficiency, and where two lines overlap.
MARKERS = {
    "baseline": "o", "checker": "s", "checker_net": "^", "checker_net_state": "D",
    "marking": "v", "gnn": "P", "seq": "X",
    "lll": "*", "gll": "h",
}

# --- ink and surface --------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def save_figure(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    """Write ``fig`` to ``path`` if there is one, and return it either way."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def copy_to_thesis(path: Path) -> Path:
    """Mirror a written figure into the thesis image directory."""
    THESIS_IMAGES.mkdir(parents=True, exist_ok=True)
    copy = THESIS_IMAGES / path.name
    shutil.copyfile(path, copy)
    return copy
