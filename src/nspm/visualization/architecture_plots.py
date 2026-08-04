"""Block diagrams of the five model variants (the ablation ladder).

These are **structure** figures, not data figures: they show where the symbolic
knowledge enters each model, which is the distinction the whole thesis turns on
(a loss term for rung 2, an input feature for rungs 3-5).

Two reading aids are deliberate:

* the shared neural skeleton is drawn in neutral grey and **what the rung adds
  is the only coloured block** -- so "each step changes exactly one thing" is
  visible before a word is read;
* the colours are the same ones :mod:`nspm.visualization.matrix_plots` gives the
  variants in the result charts, so a reader who has seen the results recognises
  the model by hue.

Dimensions come from :class:`nspm.config.ModelConfig` defaults (embedding 32,
hidden 64); ``P`` is the number of places of the mined Petri net (26 on Sepsis)
and ``|A|`` the size of the activity vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

from .benchmark_plots import _maybe_save
from .matrix_plots import VARIANT_COLORS, VARIANT_ORDER

NEUTRAL_FILL = "#eeede9"
NEUTRAL_EDGE = "#b9b8b1"
ARROW = "#8a8981"
INK = "#0b0b0b"
MUTED = "#52514e"

#: One line per rung: what it adds, and whether the addition is a loss term or
#: an input feature. Mirrors ``VARIANT_KNOWLEDGE`` in :mod:`matrix_plots`.
VARIANT_TITLES = {
    "baseline": ("1 · baseline", "nessuna conoscenza"),
    "checker": ("2 · checker", "vincolo nella loss"),
    "marking": ("3 · marking", "stato, come feature"),
    "gnn": ("4 · gnn", "stato + struttura"),
    "seq": ("5 · seq", "stato + struttura + tempo"),
}


def _box(ax, x, y, w, h, text, *, color=None, fontsize=8, bold=False):
    """Rounded block. ``color=None`` means shared skeleton (neutral grey)."""
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


def draw_variant(ax, variant: str, arch: str = "gru") -> None:
    """Draw one variant's data flow on ``ax``.

    The neural path runs down the left, the symbolic path down the right, and
    they meet at the concatenation: that meeting point *is* the architectural
    claim of rungs 3-5, and its absence is the claim of rungs 1-2.
    """
    colour = VARIANT_COLORS[variant]
    symbolic = variant in ("marking", "gnn", "seq")
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")

    x_n, w = (2.7, 4.5) if symbolic else (5.0, 6.0)
    x_s = 7.3

    # ---- percorso neurale (identico in tutte e cinque le varianti)
    _box(ax, x_n, 9.3, w, 0.9, "prefisso di token\n(B, L)")
    _arrow(ax, x_n, 8.85, x_n, 8.55)
    _box(ax, x_n, 8.1, w, 0.9, "Embedding 32")
    _arrow(ax, x_n, 7.65, x_n, 7.35)
    _box(ax, x_n, 6.9, w, 0.9, f"{arch.upper()} 64\n(packed)")
    _arrow(ax, x_n, 6.45, x_n, 6.15)
    _box(ax, x_n, 5.7, w, 0.8, "h  (B, 64)")

    # ---- percorso simbolico (solo i gradini feature)
    if symbolic:
        shape = "(B, L, P)" if variant == "seq" else "(B, P)"
        _box(ax, x_s, 9.3, 4.5, 0.9, f"marking della Petri net\n{shape}", color=colour)
        _arrow(ax, x_s, 8.85, x_s, 8.55)
        if variant == "marking":
            body, out_dim = "nessuna trasformazione\n(il vettore grezzo è la feature)", "P"
        elif variant == "gnn":
            body, out_dim = ("GNN eterogenea, 2 hop\nposti → transizioni → posti\n"
                             "readout: max sui posti"), "64"
        else:
            body, out_dim = ("GNN eterogenea per passo\n(pesi condivisi nel tempo)\n"
                             "↓\nGRU interna 64 (packed)"), "64"
        _box(ax, x_s, 7.5, 4.5, 1.9, body, color=colour, fontsize=7.0)
        _arrow(ax, x_s, 6.55, x_s, 6.15)
        _box(ax, x_s, 5.7, 4.5, 0.8, f"s  (B, {out_dim})", color=colour)

        _arrow(ax, x_n, 5.3, x_n, 4.75)
        _arrow(ax, x_s, 5.3, x_s, 4.75)
        _box(ax, 5.0, 4.3, 8.6, 0.8, f"concatenazione  (B, 64 + {out_dim})", color=colour)
        _arrow(ax, 5.0, 3.9, 5.0, 3.5)
    else:
        # niente si unisce qui: la freccia lunga e' il punto
        _arrow(ax, x_n, 5.3, x_n, 3.5)

    # ---- testa di classificazione e loss
    _box(ax, 5.0, 3.05, 6.4, 0.9, "dropout 0.2\nLinear → |A|")
    _arrow(ax, 5.0, 2.6, 5.0, 2.3)
    _box(ax, 5.0, 1.85, 6.4, 0.8, "softmax sulle attività")
    _arrow(ax, 5.0, 1.45, 5.0, 1.15)
    if variant == "checker":
        _box(ax, 5.0, 0.7, 8.6, 0.9,
             "loss = cross-entropy + λ · massa vietata dal DFG",
             color=colour, bold=True)
    else:
        _box(ax, 5.0, 0.7, 6.4, 0.8, "loss = cross-entropy")

    title, subtitle = VARIANT_TITLES[variant]
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, color=INK, pad=10)


def plot_variant_architecture(
    variant: str, arch: str = "gru", path: str | Path | None = None
) -> plt.Figure:
    """One variant, full page -- the figure for a slide dedicated to that rung."""
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    draw_variant(ax, variant, arch)
    fig.tight_layout()
    return _maybe_save(fig, path)


def plot_ablation_ladder(
    arch: str = "gru", variants=VARIANT_ORDER, path: str | Path | None = None
) -> plt.Figure:
    """All five side by side: the ladder in one image.

    Read left to right, only the coloured blocks change -- which is exactly the
    experimental design: one ingredient added per step, so a delta between two
    adjacent columns measures that ingredient and nothing else.
    """
    fig, axes = plt.subplots(1, len(variants), figsize=(4.1 * len(variants), 8.4))
    for ax, variant in zip(axes, variants):
        draw_variant(ax, variant, arch)
    fig.suptitle(
        "La scala di ablazione: ogni gradino aggiunge una cosa sola",
        fontsize=15, y=0.99,
    )
    fig.text(
        0.5, 0.015,
        "in grigio lo scheletro neurale condiviso · a colori ciò che il gradino aggiunge",
        ha="center", fontsize=10, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return _maybe_save(fig, path)
