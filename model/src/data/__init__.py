"""Data loading and preprocessing for runtime BHM notebooks.

Two stages, two sets of loaders, sharing a common RuntimeDataset container
and design-array convention. See common.py, heuristic_filtering.py,
string_normalisation.py for stage-specific details.

All paths and corpus lists are explicit arguments — nothing is hardcoded.
"""
from .common import (
    parse_runtime,
    reader_block,
    strip_task_suffix,
    make_design_arrays,
    RuntimeDataset,
)
from .heuristic_filtering import (
    load_hf_runs,
    load_hf_ear,
    load_hf_dataset,
)
from .string_normalisation import (
    load_str_norm_jobs,
    load_str_norm_ear,
    load_str_norm_dataset,
    DEFAULT_EXCLUDE_RUN_ID_PATTERN,
)

__all__ = [
    "parse_runtime", "reader_block", "strip_task_suffix",
    "make_design_arrays", "RuntimeDataset",
    "load_hf_runs", "load_hf_ear", "load_hf_dataset",
    "load_str_norm_jobs", "load_str_norm_ear", "load_str_norm_dataset",
    "DEFAULT_EXCLUDE_RUN_ID_PATTERN",
]