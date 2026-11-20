# Perron parts: variational reading, local Lyapunov function, related work

Source: a GPT reply the user shared (2026-11-19), distilled. The derivations below are **GPT's and are not yet checked
against our code**, except where noted. The intended home is the Perron chapter (`site/content/perron.html`).
Suggested placement is given in each section.

Notation as in the chapter: $A = \Sigma_g - \theta\mathbf 1\mathbf 1^\top$, embeddings $h_i \ge 0$ with
$\mathbf 1^\top h_i = 1$, and parts $A_{SS}h_S = c\,h_S$ with $(Ah)_b \le 0$ off $S$. GPT's version of the rule
is $\dot w = \eta([K]_+ - \alpha w)$ with $K = HAH^\top$: the drive is rectified, and there is no floor. Its
zero-floor equilibria are the same as ours (signed drift clipped at $w_{\min}$).

## 1. What a part optimises (place in §"Where a cluster can sit", after eq. 3)

Parts are exactly the KKT points of the **non-negative Rayleigh quotient**

$$R_\theta(h) = \frac{h^\top A h}{h^\top h}, \qquad h \ge 0 .$$

The quotient is scale-invariant, so with $z = h/\|h\|_2$ the problem is the same as

$$\max_{z \ge 0,\ \|z\|_2 = 1}\ z^\top \Sigma_g z - \theta\,\|z\|_1^2 .$$

- Stationarity on the support gives $A_{SS}h_S = c\,h_S$, with $c = R_\theta(h)$. Complementary slackness off the
  support gives $(Ah)_b \le 0$. This is eq. 3 exactly.
- A *stable* part (top simple eigenvalue on $S$, strict outside inequality) is a strict local maximum.
  Conversely, any local maximum has to be the top eigenvector on its support. So "stable part" means
  "local max of $R_\theta$". I checked this KKT step myself; it is elementary.
- **Reading of $\theta$:** it is an $\ell_1^2$ sparsity penalty on a non-negative principal component. This is
  the penalty in the SDP relaxation of sparse PCA ($\rho\,\mathbf 1^\top|X|\mathbf 1$ with $X = zz^\top$), and
  for $z \ge 0$ it is exact. So the covariance rule does **non-negative sparse PCA**, one local optimum per
  cluster.
- Uniform $h = \mathbf 1_S/|S|$ gives $R_\theta = \frac{1}{|S|}\sum_{a,b\in S}\Sigma_{ab} - \theta|S|$, which is
  **ratio association** (a standard spectral-clustering objective) with a size penalty. A part is its
  continuous non-negative relaxation, for one cluster at a time. Covering all the inputs (a full partition) is
  not part of the objective, which fits the allocation failures we see.
- The chapter's "never straddles two pieces of $G_\theta$" argument (flip the sign on one group and $h^\top Ah$
  rises) is this Rayleigh argument.

## 2. Local Lyapunov function and stability rates (place in §"Which parts get populated", at "Why can a cluster not split?")

Take a configuration of compatible stable parts: cluster $r$ has $n_r$ copies on part $v_r$ with eigenvalue
$\lambda_r$, $q_r = v_r^\top A v_r$, and $v_r^\top A v_s < 0$ between parts. GPT claims that near this
equilibrium, in weight space,

$$V(W) = \sum_r \sum_{i \in C_r}\big[\lambda_r - R_\theta(h_i(W))\big] \ \ge 0$$

is non-increasing under the **full** dynamics, with $H(W)$ recomputed and not frozen. Equivalently, the sum of
the neurons' Rayleigh quotients rises. On the support, $\lambda - R_\theta(h_i) = \sum_{k\ge2}(\lambda-\mu_k)a_{ik}^2$:
spectral error weighted by the eigengaps. Unwanted synapses, even when initially positive, decay as $e^{-\eta\alpha t}$.

Linearised rates for one cluster, where $\mu$ is another eigenvalue of $A_{SS}$:

| perturbation | rate |
|---|---|
| all copies move together along the $\mu$-direction (coherent) | $\eta\alpha(\mu/\lambda - 1)$ |
| copies separate along it (zero mean) | $\eta\alpha\big(\mu/(\lambda + nq) - 1\big)$ |
| weight changes that leave the embeddings fixed | $-\eta\alpha$ |

**Why this matters for us:** the table predicts the ramp result in the chapter. Just past a level, the parent's
positive vector is no longer on top: some contrast eigenvalue $\mu$ exceeds $\lambda$. The *coherent* mode goes
unstable first, at $\mu > \lambda$, so the cluster moves as a body towards one child. The *separating* mode
needs $\mu > \lambda + nq$, and the lateral synapses make that threshold large: $nq$ grows with cluster size
and, in our model, with the lateral gain $\gamma$. This is the "falls as a body into one child" behaviour we
observed: 1 of 2, 1 of 4, … at $\gamma = 1$, partly fixed by small $\gamma$. It is a candidate analytic
explanation of the observation, and it is **worth checking against `hebb/perron2/ramps.py`**. Our floor
$\omega$ adds an extra coupling term that GPT's model does not have.

**Limits:**
- The result is local only. GPT gives a counterexample to global monotonicity of $\sum_i R_\theta(h_i)$:
  $\Sigma = \mathrm{diag}(1, 0.8)$, $\theta = 0.25$, $W_{IB} = [[1, .001],[10, .1]]$, $w_{12} = .01$,
  $\eta=\alpha=1$, giving $\frac{d}{dt}\sum R \approx -0.0021$. Not checked.
- It does not settle allocation or basin selection.

## 3. Contrast with the distance rule (one sentence, perhaps in the idea chapter)

$\mathcal E(W) = \tfrac12\mathbb E\sum w_{ij}(u_i-u_j)^2$ has the exact envelope gradient
$\partial\mathcal E/\partial w_{ij} = \tfrac12\mathbb E(u_i-u_j)^2 = \tfrac{K_{ii}+K_{jj}}2 - K_{ij}$. We already
show this in the idea chapter. Raw covariance Hebb is $K_{ij}$ alone, without the endpoint-variance terms. That is
why it has no global energy and only the local $V$ above. It fits the existing "a local rule with a global
objective" callout.

## 4. Normalisation: subtractive vs multiplicative (place as a sidenote at eq. 1/2)

- **Miller & MacKay (1994)**: with *multiplicative* (proportional) normalisation, Hebbian learning finds the
  principal eigenvector, as in Oja. With *subtractive* normalisation, a uniform amount is subtracted from every
  synapse's update. The weights then run to their bounds, and the outcome is in general **not** an eigenvector
  of the correlation matrix. Our $\theta$ is a fixed (not budget-enforcing) uniform subtraction, and our result
  has the same shape: the outcome is not an eigenvector of $\Sigma_g$, it is set by which synapses sit at the
  floor, and it is the top eigenvector of a constrained problem. This is the closest prior result to "parts,
  not eigenvectors" and **should be cited** in the lede or at eq. 2.
- **Scale invariance:** $H(cW) = H(W)$. A *global* multiplicative budget on $W$ therefore leaves every
  embedding trajectory unchanged up to a time change, so it is harmless. Per-neuron budgets are a different
  model and would need their own analysis.
- $\theta$ is needed to create negative affinities. If $A$ is entrywise positive, no proper support can satisfy
  the off-support condition. The chapter already says this implicitly.

## References to add

- K. D. Miller & D. J. C. MacKay (1994). The role of constraints in Hebbian learning. *Neural Computation*
  6(1):100–126. <https://ctn.zuckermaninstitute.columbia.edu/sites/default/files/content/PDFs%20Miller/The%20Role%20of%20Constraints%20in%20Hebbian%20Learning.pdf>
- E. Oja (1982). A simplified neuron model as a principal component analyzer. *J. Math. Biol.* 15:267–273.
- A. d'Aspremont, L. El Ghaoui, M. I. Jordan, G. R. G. Lanckriet (2007). A direct formulation for sparse PCA
  using semidefinite programming. *SIAM Review* 49(3). <https://arxiv.org/abs/cs/0406021>
- A. Montanari & E. Richard (2016). Non-negative principal component analysis: message passing algorithms and
  sharp asymptotics. *IEEE Trans. Inf. Theory* 62(3). <https://arxiv.org/abs/1406.4775>
- I. S. Dhillon, Y. Guan, B. Kulis (2004). A unified view of kernel k-means, spectral clustering and graph
  cuts. UTCS TR-04-25 (§4.3, ratio association). <https://www.cs.utexas.edu/ftp/techreports/tr04-25.pdf>
- M. N. Galtier, O. D. Faugeras, P. C. Bressloff (2012). Hebbian learning of recurrent connections: a
  geometrical perspective. *Neural Computation* 24(9). <https://arxiv.org/abs/1102.0166>. It gives exact
  Lyapunov functions for recurrent Hebbian networks with fixed leak; not directly transferable to
  weight-dependent degrees.
