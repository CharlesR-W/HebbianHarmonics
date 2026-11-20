# HebbianHarmonics

An interactive explorer of Hebbian learning on a resistor network (the author's note is in
`docs/note.md`). Published as `HebbianHarmonics/explorer`.

The current redesign focuses on linear neurons. All data are synthetic. See `docs/PLAN_redesign.md`.

Top tabs, in order: `#talk` (The talk, home), `#basics` (The basics), `#perron` (Perron clusters),
`#readout` (Reading the weights), and `#experiments` (Experiments). The Experiments landing page links to
`#fusion` (Fusion), `#many` (Many inputs), `#allocation` (Who gets the neurons?), and `#old` (More on the covariance rule).

Nonlinear-neuron chapters (nonlinear, diversity) and 'Choices & caveats' were removed on 2026-11-20; the last commit containing them is 40cb3ff.

- Page: `site/` (static, no build step). Preview: `python3 -m http.server 8000 --bind 127.0.0.1 -d site`.
- Spec: `SPEC.md` (v2). Design contract for chapter authors: `site/DESIGN.md`.
- v2 model: `hebb/onsite.py` (numpy reference) and `site/js/onsite_engine.js` (browser);
  sweeps in `hebb/sweeps2/` -> `site/data/sweeps2/`; findings in `docs/findings_v2.md`.
- Big precomputed runs (N >> B, too large for the browser): scripts in `hebb/big/` write films
  and summaries to `site/data/big/` (format in `site/data/big/README.md`), replayed by
  `site/js/filmplayer.js`; findings in `docs/findings_v2.md`, section "Big runs (N >> B)".
- v1 model (covariance rule; used by `#perron` and `#old`): `hebb/core.py` and
  `site/js/engine.js` (`site/js/ENGINE.md`); parity test
  `uv run --with pytest pytest -q tests/test_engine_parity.py`.
- v1 documents: `docs/v1/` (`spec_v1.md`, `findings.md`, `findings_private_noise.md`,
  `mnist.md`, `nonlin.md`; `review_corrections.md` overrides the others where they differ).
  They are archival: the v1 sweep, MNIST and rectified-output code and data were removed from
  the tree after commit `f613d68` (git history keeps them), and MNIST is no longer used anywhere.
- All tests: `uv run --with pytest pytest -q tests/`.
