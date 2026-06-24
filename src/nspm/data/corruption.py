"""Training-set perturbations for robustness studies.

These helpers deliberately touch **only the training partition** (applied after
``split_traces``), so validation and test always measure honest performance.

The label-noise study replaces the next-activity *target* of a fraction of
training prefix examples with a random activity drawn from the DFA alphabet (the
set of activities the process is known to contain). The prefix *history* is left
intact, isolating label noise. Crucially, the empirical DFA and the mined LTLf
constraints are still built from the **clean** traces: the question is whether
clean symbolic knowledge keeps a model trained on dirty labels conformant and
accurate.
"""

from __future__ import annotations

from dataclasses import replace
import random

from .prefixes import ActivityVocabulary, PrefixExample


def corrupt_targets(
    examples: list[PrefixExample],
    fraction: float,
    vocabulary: ActivityVocabulary,
    rng: random.Random,
) -> tuple[list[PrefixExample], int]:
    """Replace the target of ``fraction`` of examples with a random DFA activity.

    The replacement is drawn uniformly from the activity alphabet and is always
    different from the true target, so the number of changed labels equals the
    requested count. Returns the (new) example list and how many were corrupted.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")
    number_of_classes = len(vocabulary.activities)
    if fraction == 0.0 or number_of_classes < 2:
        return list(examples), 0

    count = round(len(examples) * fraction)
    corrupt_indices = set(rng.sample(range(len(examples)), count))

    perturbed: list[PrefixExample] = []
    changed = 0
    for index, example in enumerate(examples):
        if index in corrupt_indices:
            new_target = rng.randrange(number_of_classes)
            if new_target == example.target_id:  # guarantee an actual change
                new_target = (new_target + 1) % number_of_classes
            perturbed.append(replace(example, target_id=new_target))
            changed += 1
        else:
            perturbed.append(example)
    return perturbed, changed


def subsample_traces(
    traces: dict[str, tuple[str, ...]],
    fraction: float,
    seed: int = 42,
) -> dict[str, tuple[str, ...]]:
    """Keep a random ``fraction`` of cases (for the future data-ablation study)."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if fraction == 1.0:
        return dict(traces)
    case_ids = sorted(traces)
    keep = max(1, round(len(case_ids) * fraction))
    chosen = random.Random(seed).sample(case_ids, keep)
    return {case_id: traces[case_id] for case_id in chosen}
