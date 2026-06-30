## Adding a new corpus (anchoring)

A new corpus gets a usable prediction from a small probe instead of a full run. Filtering needs 4 probe runs (one per config sweep point, to identify a slope); string normalisation needs 3 (one per form). See thesis Sections 2.f/3.c.

1. Run the probe, process it with the matching loader
   (`load_hf_dataset` / `load_str_norm_dataset`), save as CSV.
2. Call `fit_anchored(base_csv, probe_csv, model, extra_design_cols, save_to=...)`:

```python
   from models import model_offset_hf, model_offset_interaction_no_ntasks, fit_anchored

   # filtering
   dataset, idata = fit_anchored(
       "results/df_train_hf.csv", "results/probe_hf.csv",
       model_offset_hf, extra_design_cols=["alpha_cfg"],
       save_to="results/fit_hf_production.nc",
   )

   # string normalisation
   dataset, idata = fit_anchored(
       "results/df_train_str_norm.csv", "results/probe_sn.csv",
       model_offset_interaction_no_ntasks, extra_design_cols=["form_idx"],
       save_to="results/fit_str_norm_production.nc",
   )
```

3. Predict with `run_energy_mc_hf` / `run_energy_mc_str_norm` using `dataset.corpora` and `dataset.corp2idx` for the new corpus's index.

`save_to` overwrites the base CSV and `.nc` fit in place — copy them first if you want a snapshot before anchoring.