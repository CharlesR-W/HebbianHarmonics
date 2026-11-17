# HebbianHarmonics

An interactive explorer of "Hebbian learning on a resistor network" (the note is in
`docs/note.md`). Published as `HebbianHarmonics/explorer`.

- Page: `site/` (static, no build step). Preview: `python3 -m http.server 8000 --bind 127.0.0.1 -d site`.
- Model: `hebb/core.py` (numpy reference) and `site/js/engine.js` (browser); parity tests: `uv run --with pytest pytest -q tests/`.
- Experiments: `hebb/sweeps/` (synthetic sweeps, `run_all.sh`), `hebb/mnist/`, `hebb/nonlin/` (Modal).
- What was found: `docs/findings.md`, `docs/findings_private_noise.md`, `docs/mnist.md`, `docs/nonlin.md`;
  `docs/review_corrections.md` overrides them where they differ.
- Spec and design contract: `SPEC.md`, `site/DESIGN.md`, `site/js/ENGINE.md`.
