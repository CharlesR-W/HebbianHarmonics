"""Precomputed big runs (N >> B) for the explorer: films and summaries in site/data/big/.

See site/data/big/README.md for the data format and the list of runs, and
docs/findings_v2.md "Big runs (N >> B), 2026-11-19" for the numbers.
Engines are imported, never edited: hebb/core.py (v1 covariance rule), hebb/onsite.py
(v2 distance rule). hebb/big/rect_torch.py is a batched re-implementation of
onsite.equilibrate validated against it (hebb/big/validate.py).
"""
