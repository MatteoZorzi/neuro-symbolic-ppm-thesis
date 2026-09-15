# Draw the three architecture figures of the thesis from the code that ran

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from common import (FIGURES, INK, INK_SECONDARY as MUTED, ROOT,  # noqa: E402
                    copy_to_thesis, save_figure)

GRID = Path(__file__).resolve().parent / "matrix.py"
CONFIG = ROOT / "src" / "nspm" / "config.py"


# =====================================================================
#  The drawing half. One reading aid is deliberate: the shared neural
#  skeleton is drawn in neutral grey and what the variant adds is the
#  only coloured block, so "each model changes exactly one thing" is
#  visible before a word is read.
#
#  ``P`` is the number of places of the mined Petri net (26 on Sepsis)
#  and ``|A|`` the size of the activity vocabulary.
# =====================================================================

NEUTRAL_FILL = "#eeede9"
NEUTRAL_EDGE = "#b9b8b1"
ARROW = "#8a8981"

# Rounded block. ``color=None`` means shared skeleton (neutral grey)
def _box(ax, x, y, w, h, text, *, color=None, fontsize=8, bold=False):
    face = NEUTRAL_FILL if color is None else to_rgba(color, 0.20)
    edge = NEUTRAL_EDGE if color is None else color
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=face, edgecolor=edge, linewidth=2.0 if color else 1.1, zorder=2,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK,
            zorder=3, fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color=ARROW, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=1.3,
                                shrinkA=2, shrinkB=2))


#: The trunk, identical in all eight variants of the benchmark. Mirrors
#: ``RECURRENT_HIDDEN`` and ``RECURRENT_LAYERS`` in ``matrix.py`` and
#: the ``ModelConfig`` defaults for embedding and dropout. They are Mezini et
#: al.'s numbers, adopted for every variant and not only for his two, so that
#: the comparison is between channels and not between architectures.
#:
#: ``official_experiments/scripts/architecture_figures.py`` asserts these
#: against the grid configuration before it draws anything: a figure claiming a
#: dimension the runs did not use is worse than no figure at all.
EMBEDDING_DIM = 32
HIDDEN_DIM = 100
RECURRENT_LAYERS = 2
DROPOUT = 0.20

#: How each objective is assembled, from ``learning/training.py``. The two
#: net-derived penalties are *added* to the task loss with weight
#: ``LOGIC_WEIGHT``; Mezini's two losses *mix* supervision and logic with
#: ``alpha`` and ``1 - alpha``, so the two families are not on the same scale
#: and the figures state their coefficients rather than leaving a bare lambda.
LOGIC_WEIGHT = 0.5
ALPHA_LOCAL = 0.25
ALPHA_GLOBAL = 0.75
GUMBEL_TEMPERATURE = 0.5
GLOBAL_SAMPLES = 10

#: One hue family per figure, neighbouring hues inside it.
#:
#: Colour carries the channel, not the variant: a reader sees at a glance that
#: the panels of one figure belong together and that the figures do not. Shades
#: of a single colour would read as "more of the same thing", and the panels
#: are different things inside one channel. The baseline is left grey in every
#: sense -- it adds nothing, and an empty symbolic column is the whole
#: statement. The result charts keep the per-model palette of
#: :data:`common.COLORS`, where naming a single model is the job.
LOSS_SHADES = ("#eb6834", "#c1272d")                 # orange, red
FEATURE_SHADES = ("#1baf7a", "#0f9ab4", "#2f6fd0")   # green, teal, blue
AXEL_SHADES = ("#8e59c9", "#4f3a9e")                 # violet, indigo

LOSS_ORDER = ("baseline", "net", "net_state")
FEATURE_ORDER = ("marking", "gnn", "seq")
AXEL_ORDER = ("local", "global")

#: Panel names, taken from ``tab:exp-cost`` of Chapter 5. These are the names
#: the thesis uses; the keys above are the ones the result CSVs carry, and the
#: mapping between the two belongs to the reproducibility section, not here.
PANEL_TITLES = {
    "baseline": ("Baseline", "no knowledge"),
    "net": ("Projected net", "mask indexed by the last activity"),
    "net_state": ("State-indexed net", "mask indexed by the automaton state"),
    "marking": ("Marking", "the marking, as a flat feature"),
    "gnn": ("Graph encoder", "the marking and the structure of the net"),
    "seq": ("Marking sequence", "the marking, the structure, the history"),
    "local": ("Local loss", "one step, the mass the automaton forbids"),
    "global": ("Global loss", "a whole rollout, the acceptance at its end"),
}

PANEL_COLORS = {
    "baseline": None,
    "net": LOSS_SHADES[0],
    "net_state": LOSS_SHADES[1],
    **dict(zip(FEATURE_ORDER, FEATURE_SHADES)),
    **dict(zip(AXEL_ORDER, AXEL_SHADES)),
}

#: What each feature rung does to the marking: the transformation, the width it
#: hands to the concatenation, and the shape it reads. The inner recurrence of
#: the marking sequence is a GRU and stays one whatever the trunk is: it is
#: hard-coded in :class:`~nspm.learning.models.MarkingSequenceEncoder`, which
#: never sees the architecture of the trunk.
FEATURE_BODIES = {
    "marking": ("no transformation\n(the raw vector is the feature)",
                "P", "(B, P)"),
    "gnn": ("heterogeneous GNN, 2 hops\nplaces → transitions → places\n"
            "readout: max over places", str(HIDDEN_DIM), "(B, P)"),
    "seq": ("heterogeneous GNN per step\n(weights shared over time)\n↓\n"
            f"inner GRU {HIDDEN_DIM} (packed)", str(HIDDEN_DIM), "(B, L, P)"),
}

#: The two masks of RQ3: where the constraint comes from, how its rows are
#: indexed, and its shape. Everything else in the two arms is identical.
LOSS_ARMS = {
    "net": (
        "reachability automaton\nof the discovered net",
        "projected onto activity pairs:\nthe union of what the states\n"
        "an activity reaches allow",
        "M$^{\\mathrm{net}}$   (|T|, |A|)",
    ),
    "net_state": (
        "reachability automaton\nof the discovered net",
        "memory kept: the row is\nchosen by the state the\n"
        "automaton reaches",
        "M$^{\\mathrm{state}}$   (|Q|, |A|)",
    ),
}

#: The two logic losses of the state of the art. Source of the constraint, what
#: the method does with it, what reaches the objective.
AXEL_ARMS = {
    "local": (
        "reachability automaton\nof the discovered net,\nprojected",
        "row indexed by the last activity;\nthe cross-entropy is switched off\n"
        "where the target itself is forbidden",
        "M$^{\\mathrm{net}}$   (|T|, |A|)",
    ),
    "global": (
        "reachability automaton\nof the discovered net,\ntensorized",
        "transition matrices over states\nand activities, traversed on\n"
        "distributions instead of symbols",
        "q$_0$   (B, |Q|)",
    ),
}

_TRUNK = f"LSTM {HIDDEN_DIM}, {RECURRENT_LAYERS} layers\n(packed)"
_HEAD = f"dropout {DROPOUT:.1f}\nLinear → |A|"


# Curved arrow, for the one place where the flow doubles back
def _loop_arrow(ax, x0, y0, x1, y1, color, rad=0.3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3,
                                linestyle=(0, (4, 2)), shrinkA=4, shrinkB=4,
                                connectionstyle=f"arc3,rad={rad}"))


# Prefix, embedding, recurrence, hidden state: the part nobody varies
def _draw_trunk(ax, x, w, *, bottom=5.7):
    _box(ax, x, 9.3, w, 0.9, "token prefix\n(B, L)")
    _arrow(ax, x, 8.85, x, 8.55)
    _box(ax, x, 8.1, w, 0.9, f"Embedding {EMBEDDING_DIM}")
    _arrow(ax, x, 7.65, x, 7.35)
    _box(ax, x, 6.9, w, 0.9, _TRUNK)
    _arrow(ax, x, 6.45, x, bottom + 0.45)
    _box(ax, x, bottom, w, 0.8, f"h  (B, {HIDDEN_DIM})")


def _set_panel_title(ax, key):
    title, subtitle = PANEL_TITLES[key]
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, color=INK, pad=10)


# One feature rung: neural path left, symbolic path right, joined
def draw_feature_rung(ax, variant: str) -> None:
    colour = PANEL_COLORS[variant]
    body, out_dim, shape = FEATURE_BODIES[variant]
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")

    x_n, w, x_s = 2.65, 4.3, 7.35
    _draw_trunk(ax, x_n, w)

    _box(ax, x_s, 9.3, 4.5, 0.9, f"Petri net marking\n{shape}", color=colour)
    _arrow(ax, x_s, 8.85, x_s, 8.55)
    _box(ax, x_s, 7.5, 4.3, 1.9, body, color=colour, fontsize=7.0)
    _arrow(ax, x_s, 6.55, x_s, 6.15)
    _box(ax, x_s, 5.7, 4.3, 0.8, f"s  (B, {out_dim})", color=colour)

    _arrow(ax, x_n, 5.3, x_n, 4.75)
    _arrow(ax, x_s, 5.3, x_s, 4.75)
    _box(ax, 5.0, 4.3, 8.6, 0.8,
         f"concatenation  (B, {HIDDEN_DIM} + {out_dim})", color=colour)
    _arrow(ax, 5.0, 3.9, 5.0, 3.5)

    _box(ax, 5.0, 3.05, 6.4, 0.9, _HEAD)
    _arrow(ax, 5.0, 2.6, 5.0, 2.3)
    _box(ax, 5.0, 1.85, 6.4, 0.8, "softmax over activities")
    _arrow(ax, 5.0, 1.45, 5.0, 1.15)
    _box(ax, 5.0, 0.7, 6.4, 0.8, "loss = cross-entropy")

    _set_panel_title(ax, variant)


# The three feature rungs side by side
def plot_feature_channel(path: str | Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 8.4))
    for ax, variant in zip(axes, FEATURE_ORDER):
        draw_feature_rung(ax, variant)
    fig.suptitle(
        "The feature channel: the knowledge enters the input",
        fontsize=15, y=0.99,
    )
    fig.text(
        0.5, 0.015,
        "shared neural skeleton in grey · in colour what the variant adds",
        ha="center", fontsize=10, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return save_figure(fig, path)


# One loss arm, laid out exactly like a feature rung
def draw_loss_arm(ax, arm: str) -> None:
    colour = PANEL_COLORS[arm]
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")

    x_n, w, x_s = 2.65, 4.3, 7.35
    _draw_trunk(ax, x_n, w)

    if arm != "baseline":
        source, body, mask = LOSS_ARMS[arm]
        _box(ax, x_s, 9.3, w, 0.9, source, color=colour, fontsize=7.4)
        _arrow(ax, x_s, 8.85, x_s, 8.55)
        _box(ax, x_s, 7.5, w, 1.9, body, color=colour, fontsize=7.0)
        _arrow(ax, x_s, 6.55, x_s, 6.15)
        _box(ax, x_s, 5.7, w, 0.8, mask, color=colour)

    # The long unbroken arrows are the claim: nothing merges before the loss.
    _arrow(ax, x_n, 5.3, x_n, 3.5)
    _box(ax, x_n, 3.05, w, 0.9, _HEAD)
    _arrow(ax, x_n, 2.6, x_n, 2.3)
    _box(ax, x_n, 1.85, w, 0.8, "softmax over activities")
    _arrow(ax, x_n, 1.45, x_n, 1.05)

    if arm == "baseline":
        _box(ax, x_n, 0.6, w, 0.8, "loss = cross-entropy")
    else:
        _arrow(ax, x_s, 5.3, x_s, 1.05, color=colour)
        _box(ax, 5.0, 0.6, 8.6, 1.0,
             "loss = cross-entropy + λ · forbidden mass\n"
             f"λ = {LOGIC_WEIGHT}",
             color=colour, fontsize=8.0, bold=True)

    _set_panel_title(ax, arm)


# The baseline and the two net-derived penalties, side by side
def plot_loss_channel(path: str | Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 8.4))
    for ax, arm in zip(axes, LOSS_ORDER):
        draw_loss_arm(ax, arm)
    fig.suptitle(
        "The loss channel: the knowledge enters the objective",
        fontsize=15, y=0.99,
    )
    fig.text(
        0.5, 0.015,
        "shared neural skeleton in grey · the two penalties differ only "
        "in the mask that feeds them",
        ha="center", fontsize=10, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return save_figure(fig, path)


# The local logic loss: one step, and the mass the automaton forbids
def draw_local_loss(ax) -> None:
    colour = PANEL_COLORS["local"]
    source, body, mask = AXEL_ARMS["local"]
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")

    x_n, w, x_s = 2.65, 4.3, 7.35
    _draw_trunk(ax, x_n, w)

    _box(ax, x_s, 9.3, w, 0.9, source, color=colour, fontsize=7.4)
    _arrow(ax, x_s, 8.85, x_s, 8.55)
    _box(ax, x_s, 7.5, w, 1.9, body, color=colour, fontsize=7.0)
    _arrow(ax, x_s, 6.55, x_s, 6.15)
    _box(ax, x_s, 5.7, w, 0.8, mask, color=colour)

    _arrow(ax, x_n, 5.3, x_n, 3.5)
    _box(ax, x_n, 3.05, w, 0.9, _HEAD)
    _arrow(ax, x_n, 2.6, x_n, 2.3)
    _box(ax, x_n, 1.85, w, 0.8, "softmax over activities")
    _arrow(ax, x_n, 1.45, x_n, 1.05)
    _arrow(ax, x_s, 5.3, x_s, 1.05, color=colour)

    _box(ax, 5.0, 0.6, 8.6, 1.0,
         "loss = α · masked cross-entropy\n"
         "+ (1 − α) · [−log(1 − forbidden mass)]"
         f"        α = {ALPHA_LOCAL}",
         color=colour, fontsize=8.0, bold=True)

    _set_panel_title(ax, "local")


# The global logic loss: the model runs on, and the automaton judges
def draw_global_loss(ax) -> None:
    colour = PANEL_COLORS["global"]
    source, body, state = AXEL_ARMS["global"]
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")

    x_n, w, x_s = 2.65, 4.3, 7.35
    _draw_trunk(ax, x_n, w)

    _box(ax, x_s, 9.3, w, 0.9, source, color=colour, fontsize=7.4)
    _arrow(ax, x_s, 8.85, x_s, 8.55)
    _box(ax, x_s, 7.5, w, 1.9, body, color=colour, fontsize=7.0)
    _arrow(ax, x_s, 6.55, x_s, 6.15)
    _box(ax, x_s, 5.7, w, 0.8, state, color=colour)

    _arrow(ax, x_n, 5.3, x_n, 5.0)
    _box(ax, x_n, 4.55, w, 0.9, f"{_HEAD}\nsoftmax over activities", fontsize=7.4)

    # The logits go two ways: the true next activity scores the cross-entropy,
    # the sampled one carries the rollout.
    _arrow(ax, x_n, 4.1, x_n, 2.35)
    _box(ax, x_n, 1.9, w, 0.8, "cross-entropy\non the true next activity")

    _arrow(ax, x_n + w / 2, 4.4, x_s - w / 2, 3.9, color=colour)
    _arrow(ax, x_s, 5.3, x_s, 4.1, color=colour)
    _box(ax, x_s, 3.3, w, 1.5,
         f"rollout: horizon steps,\n{GLOBAL_SAMPLES} samples per prefix\n"
         f"Gumbel-Softmax (τ = {GUMBEL_TEMPERATURE})\n"
         "over the logits\n→ one soft activity",
         color=colour, fontsize=6.6)
    # The dashed return is the autoregression: the sampled activity is the next
    # input, which is why a rollout costs horizon sequential steps.
    _loop_arrow(ax, x_s - w / 2, 3.3, x_n + w / 2, 6.9, colour, rad=0.25)

    _arrow(ax, x_s, 2.55, x_s, 2.35)
    _box(ax, x_s, 1.9, w, 0.8, "acceptance  (B,)", color=colour)

    _arrow(ax, x_n, 1.5, x_n, 1.15)
    _arrow(ax, x_s, 1.5, x_s, 1.15, color=colour)
    _box(ax, 5.0, 0.6, 8.6, 1.0,
         "loss = α · cross-entropy\n"
         "+ (1 − α) · (−log acceptance)"
         f"        α = {ALPHA_GLOBAL}",
         color=colour, fontsize=8.0, bold=True)

    _set_panel_title(ax, "global")


# The two logic losses of the state of the art, side by side
def plot_logic_losses(path: str | Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 8.4))
    draw_local_loss(axes[0])
    draw_global_loss(axes[1])
    fig.suptitle(
        "The logic losses of the state of the art",
        fontsize=15, y=0.99,
    )
    fig.text(
        0.5, 0.015,
        "from Mezini et al. · same trunk and same automaton "
        "as the rest of the benchmark, only the objective changes",
        ha="center", fontsize=10, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return save_figure(fig, path)

FIGURES_TO_DRAW = (
    ("loss-channel.png", plot_loss_channel),
    ("feature-channel.png", plot_feature_channel),
    ("logic-losses.png", plot_logic_losses),
)

#: Every number the figures print, and where it has to come from. Left is the
#: constant defined above, right is the name it has in the configuration the
#: grid ran with.
CHECKS = (
    ("hidden units of the trunk", "HIDDEN_DIM", "grid", "RECURRENT_HIDDEN"),
    ("stacked recurrent layers", "RECURRENT_LAYERS", "grid", "RECURRENT_LAYERS"),
    ("weight of the net-derived penalty", "LOGIC_WEIGHT", "grid", "LOGIC_WEIGHT"),
    ("alpha of the local loss", "ALPHA_LOCAL", "grid", "AXEL_ALPHA_LOCAL"),
    ("alpha of the global loss", "ALPHA_GLOBAL", "grid", "AXEL_ALPHA_GLOBAL"),
    ("Gumbel-Softmax temperature", "GUMBEL_TEMPERATURE", "grid", "AXEL_TEMPERATURE"),
    ("rollout samples", "GLOBAL_SAMPLES", "grid", "AXEL_SAMPLES"),
    ("embedding width", "EMBEDDING_DIM", "model", "embedding_dim"),
    ("dropout before the classifier", "DROPOUT", "model", "dropout"),
)


# Module-level literal assignments of ``path``, without importing it
def literals(path: Path, inside: str | None = None) -> dict[str, object]:
    body = ast.parse(path.read_text(encoding="utf-8")).body
    if inside is not None:
        body = next(node for node in body
                    if isinstance(node, ast.ClassDef) and node.name == inside).body

    values: dict[str, object] = {}
    for node in body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names, value = [node.target.id], node.value
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        else:
            continue
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except ValueError:
            continue
        values.update(dict.fromkeys(names, literal))
    return values


# Refuse to draw a figure that disagrees with the grid it describes
def verify() -> None:
    sources = {"grid": literals(GRID), "model": literals(CONFIG, inside="ModelConfig")}

    mismatches = []
    for description, drawn_name, source, ran_name in CHECKS:
        drawn = globals()[drawn_name]
        ran = sources[source][ran_name]
        print(f"  {'ok' if drawn == ran else 'MISMATCH':>8}  {description}: "
              f"figure {drawn}, configuration {ran}")
        if drawn != ran:
            mismatches.append(
                f"{drawn_name}: the figures say {drawn}, {ran_name} is {ran}")

    if mismatches:
        raise SystemExit(
            "The figures do not describe the models that were trained:\n  "
            + "\n  ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw the three architecture figures of the thesis from the code that ran.")
    parser.add_argument(
        "--out", type=Path, default=FIGURES,
        help="where to write the three PNGs (default: official_experiments/figures)")
    parser.add_argument("--no-thesis-copy", action="store_true",
                        help="do not mirror the figures into the thesis")
    args = parser.parse_args()
    out = args.out

    print("Checking the figures against the grid configuration:")
    verify()

    out.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES_TO_DRAW:
        draw(path=out / name)
        print(f"wrote {(out / name).relative_to(ROOT)}")

        if not args.no_thesis_copy:
            print(f"  mirrored to {copy_to_thesis(out / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
