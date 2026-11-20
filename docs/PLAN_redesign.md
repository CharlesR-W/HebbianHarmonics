# Plan: readable redesign (2026-11-20)

Supersedes `docs/PLAN_curation.md` and, where they differ, `SPEC.md` §4 and `site/DESIGN.md` §3.

## What the author asked for

- The site is a working experiments notebook, aimed at a reader like the average LessWrong
  reader: curious, technical, has not seen this project. Message: here is how Hebbian learning
  can work without a loss function, and what it gives you as a model of behaviour and as an
  interpretability tool.
- Too much of the current site is clutter, and almost no section has a figure that shows the
  thing the section is about. Fix that: **good, relevant, easily readable figures first**.
- Liked: the slide-style opening ("The talk"); the triangle plots (neurons drawn at their
  mixture of three inputs); the figures showing the input covariance recovered from the weights;
  harmonic embeddings as an idea.
- Interesting material gets its own page: the basics, Perron clustering, reading the covariance
  back from the weights. Everything else goes under one **Experiments** tab.
- **Removed for now:** nonlinear neurons (on-site potentials, rectifiers, response-function
  diversity). Linear networks only.
- **Removed for good:** status badges (DERIVED / SHOWN / CONJECTURE), boxed "key point" claim
  statements with their qualifiers, the long "Plotted: ... Values: ..." captions, the
  "Choices & caveats" tab, lists of what was and was not obtained, and every aside that argues
  against an earlier idea or an earlier version of the project ("the note expected X", "v1 did
  Y", "why not Z"). The reader has never heard of those ideas.
- Add a note on subtractive versus multiplicative weight decay: they give different objectives
  (see `docs/perron_variational_and_related_work.md` §1 and §4). The author thinks the result
  is important but somewhat sensitive to this model choice, and the page should say so once,
  plainly.

## Site structure

| tab id | nav label | content |
|---|---|---|
| `talk` | The talk | the current Primer slides, renamed back; nonlinear slide removed |
| `basics` | The basics | the model: resistor network, each neuron is a mixture of the inputs (its harmonic embedding), Hebb's rule written in those variables, watching it learn |
| `perron` | Perron clusters | where clusters sit: non-negative Perron vectors of pieces of the thresholded covariance; the threshold cuts the correlation tree; clusters are not eigenvectors |
| `readout` | Reading the weights | recovering the input covariance from the learned weights; what is invisible; how model-sensitive it is |
| `experiments` | Experiments | landing page of cards linking to sub-pages `fusion`, `many`, `allocation`, `old` (linear parts only) |

## Rules for every page

1. **Figure first.** A page is 4 to 6 figures with short text between them. Each section opens
   with its figure or reaches it within one short paragraph. If a section has no figure that
   shows its point, cut the section or make the figure.
2. **A figure carries its result alone.** Title: a plain sentence stating what the figure shows
   ("Neurons that listen to the same inputs end up wired together"), not a label ("Voltage
   covariance"). Axes labelled in words, with the symbol in brackets if useful. Caption: at
   most two sentences, how to read it and what to notice. Anything else (parameters, seeds,
   solver details) goes in a collapsed `<details class="fig-details">` titled "Details".
3. **One idea per figure.** No eight-panel dashboards. At most three controls per figure;
   every control must change something the caption talks about. Sensible defaults: the figure
   must already show its point before the reader touches anything.
4. **Evolution gets a filmstrip.** Where the point is change over training or over a
   parameter, show five or six small frames side by side (fixed axes) next to or instead of a
   scrubber; a row of small frames is read at a glance.
5. **Text.** Plain words, short sentences, second person or "we". Define each symbol where it
   first appears. No project history, no hedging lists. A page should read in five minutes.
   Maths only where it is the point; at most a handful of displayed equations per page.
6. **Keep the two-space convention**, lightly: a figure is either a wiring diagram (nodes are
   neurons) or drawn in input space (the triangle, heatmaps over inputs). Say which in the
   title or caption in words; the coloured badge is optional.
7. **Honesty without ledgers.** If something only holds under a condition, say it in the
   sentence that states the result. One "Limits" paragraph at the end of a page is fine.
8. Linear networks only. All inputs are synthetic; say so once per page, not on every figure.

## Process

1. Shell restructure (tabs, removals).
2. Draft round: one author per main page builds the page with its candidate figures, real
   computations, screenshots checked.
3. The author picks figures from the candidates; pages are tightened around the picks.
4. Independent review (GPT) of maths, code and claims; fixes.
