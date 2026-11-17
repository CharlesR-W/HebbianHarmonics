# HebbianHarmonics

An interactive explorer of Hebbian learning on a resistor network (the author's note is in
`docs/note.md`). Published as `HebbianHarmonics/explorer`.

Version 2 (in progress) rebuilds the explorer around a new model: neurons with on-site
potentials (a nonlinear transfer function), whole response functions as the objects that
cluster, and a distance-based plasticity rule that is the gradient of an energy. The first
model, a raw covariance rule on a linear resistor network, is condensed into one chapter
("The old rule", `#old`).

- Page: `site/` (static, no build step). Preview: `python3 -m http.server 8000 --bind 127.0.0.1 -d site`.
- Spec: `SPEC.md` (v2). Design contract for chapter authors: `site/DESIGN.md`.
- v2 model: `hebb/onsite.py` (numpy reference) and `site/js/onsite_engine.js` (browser);
  sweeps in `hebb/sweeps2/` → `site/data/sweeps2/`; findings in `docs/findings_v2.md`.
- v1 model (used only by the old-rule chapter): `hebb/core.py` and `site/js/engine.js`
  (`site/js/ENGINE.md`); parity test `uv run --with pytest pytest -q tests/test_engine_parity.py`.
- v1 documents: `docs/v1/` (`spec_v1.md`, `findings.md`, `findings_private_noise.md`,
  `mnist.md`, `nonlin.md`; `review_corrections.md` overrides the others where they differ).
  The v1 sweep, MNIST and rectified-output code and data were removed from the tree after
  commit `f613d68`; git history keeps them.
- All tests: `uv run --with pytest pytest -q tests/`.
