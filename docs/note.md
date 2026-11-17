# The user's note (verbatim apart from equations repaired from a mangled paste)

Here I'll explain something I worked out today about how Hebbian learning works.  Hebbian learning is the rule that "neurons that fire together wire together" commonly used in neuroscience.  It's remarkable that this works because there *is no global loss function*.  How then, or what, is the brain doing?  Presumably you can guess my answer - let's see how.

We'll consider our neural network as being a network of resistors; each neuron is a node with voltage u_i.  The weights between neurons are conductances w_ij.  We will assume that there are a number of *input* neurons, whose voltages are *clamped*, representing the input of the rest of the brain.  We'll show that Hebbian learning can only learn the second moments of this input distribution.  Therefore we will assume that the input has been lifted by a fixed feature-map, so that g = phi(stimulus) for some rich vector of functions phi; in the limit of a rich kernel, the covariance of g is sufficient to determine the *whole* distribution over the stimulus, p(stimulus)

The Hebb rule for a synaptic weight is

$$\dot w_{ij} = \rho(w_{ij}) + \mathrm{Cov}(u_i, u_j)$$

(\rho(w_ij) is a regularizer, usually with some weight decay but also some term that prevents weights getting too small; the author's starting point was \rho(w) = w_0 - \lambda w, and they want \rho to depend on the weight only.) From the weighted graph, we can calculate the response matrix H which relates the clamped voltages to the interior voltages - it is a function of the weights (specifically the Kron reduction of the graph laplacian)

$$u_i = H_{ib} g_b$$

(b for boundary nodes; H is the 'discrete poisson kernel')
We can therefore describe a neuron in terms of its harmonic embedding h_i, which is a function only of the weights (in fact it is a probability distribution over the boundary-nodes).  The voltage covariance from Hebb's rule becomes

$$\Sigma_u = H \Sigma_g H^T, \qquad (\Sigma_u)_{ij} = \langle h_i, h_j\rangle_{\Sigma_g}$$

That is, the Hebb rule reads

$$\dot w_{ij} = \text{regularizer} + \text{const} \cdot \langle h_i, h_j\rangle_{\Sigma_g}$$

(the paste had a minus sign here; the sentence that follows fixes the intended sign.) We cut the connection between two neurons to the extent that their harmonic embeddings misalign.  The regularizer prevents weights from going either too low or too high.

The following is conjecture that I haven't quite worked out the details for:
At a fixed point, the embedding vectors should separate into discrete clusters, with the number of those clusters set by the homeostatic parameters.  Assuming there are many more interior neurons than boundary ones, each direction in embedding space will have many representatives.

One can now do *weight-space interpretability*!!!
Given some set of weights wij, we can calculate the harmonic embeddings.  Then we can *solve* for Cov(g)!!  Specifically we have the set of quadratic equations for C

$$h_{ib} C_{bc} h_{jc} = -\rho(w_{ij}) \;\Leftrightarrow\; H C H^T = R, \qquad C = H^+ R H^{+T}$$

(inserting ridge regularizers as you like).
We have recovered the covariance of g.

if g=\phi(stim) is a very rich set of feature functions, this covariance is sufficient to recover the distribution p(stimulus) (assuming we know phi, though).

We can see how Hebbian learning works!  The Hebb rule organises neurons into a discrete set of clusters.  The privileged basis for interpretability is the *harmonic embedding* of the neurons.  Each population becomes isomorphic to the eigenmode of g which it represents.

The Hebb rule accomplishes two things - first, it makes an arbitrary population of neurons statistically isomorphic to its inputs.  Second, it produces an *expansion code* of those inputs - for each eigenmode of g, there will be N neurons whose harmonic embeddings have high alignment, each of which individually has the same statistics as g.  We have implemented a reservoir code! - a family of random functions of the input, usable for downstream computations (coding like this makes sense because neurons have limited read-write, so you hope that *one* of those functions will expose the right useful feature).  And!  We have arrived at a way to *generate* the rich set of feature functions in the first place!

We can interpret the synaptic regularizers in terms of their effects - i.e., we can reverse-engineer what tradeoff they navigate!!  Without concerning ourselves with the equations, we can see that there are two knobs: how many eigenmodes of g do I want to resolve, and, for each eigenmode, how many random functions do I think I'll need?  (we didn't include noise but surely one could)

Magnificently, we now have a precise theory of how the plasticity rules navigate the resolution-expressivity tradeoff!!  The more neurons you have, the more surface area you give downstream computations.

Real neurons of course do not have symmetric weights, etc. but there are directed graph laplacians, and a corresponding generalization of the harmonic embedding.  At a fundamental level, this is how the Hebb rule organizes computation in the brain: neuron soup organizes neurons into spectral clusters corresponding to eigenmodes of the input stimulus, and each neuron within such a cluster represents a random feature applied to that eigenmode.

I haven't worked it out yet but something that seems important is how the eigenmode energy relates to the number of neurons allocated to it - this would tell you if the soup assigns neurons proportionally to energy or not.  By default I'd guess you want neurons in proportion to energy, and different regularizers will set an effective spectral filter that your network will apply - i.e., the homeostatic parameters may be naturally understood as encoding a spectral filter which sets num_neurons_dedicated_to_mode(energy_of_mode).

## What the author asked for

A visual explorer, both exploratory and pedagogical, "nice and well explained":
1. 3 boundary neurons + many interior neurons: watch the harmonic embeddings change over time in 3D.
2. Random input statistics with more boundary neurons.
3. MNIST (or CIFAR) with an appropriate kernel lift (tunable parameter count), train, possibly with tunable homeostatic parameters; extract the statistics from the weights as outlined and compare with the true lifted statistics in an appropriate PSD metric; show the embedding similarity matrix over training; show the embeddings as images; plot the graph, maybe as UMAPs.
4. An explorer for the effect of the homeostatic parameters on the clustering (assumed covariance is fine; real data would be nice).
"Wherever I'm being dumb, use your good judgement and create something smart and interesting. Make any controversial/strange engineering choices clear."
