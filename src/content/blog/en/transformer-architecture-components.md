---
title: "The parts of a modern Transformer"
description: 'The parts of an LLM architecture: attention variants, positional encoding, KV-cache and MoE — what got added and what got dropped since the 2017 design.'
pubDate: '2026-08-01'
---

## Abstract

When a language model's architecture gets explained today, the thing drawn on
the board is still 2017's original Transformer diagram. Yet since that day the
Transformer has changed beyond recognition next to that first design.

The starkest proof of it hides in the enormous gap opened by one specific
component. While most current models look for ways to optimize memory,
DeepSeek-V3 made a radical move and adopted MLA (Multi-Head Latent Attention),
which, rather than storing the key/value matrices at all, reduces them to a
single compressed vector. The result? In DeepSeek-V3 a single request's KV-cache
(key/value cache) at 128K context is only 8.6 GiB. Built with 2017's standard
attention, a model of the same size would have sat at 488 GiB.

Open the config file of a model published today and you find the insides renewed
from top to bottom. Almost none of the ingredients of that first 2017 recipe are
still in place: the sinusoidal positional encoding by which the model memorized
word order, the feed-forward network built on ReLU (Rectified Linear Unit), the
normalization applied to sub-layer outputs, and heads owning their own key/value
matrices have all passed into history. What is left of the original design is
only the load-bearing skeleton: the attention operation itself, residual
connections, and the idea of stacking layers on top of one another.

This piece works through the Transformer's eight-year evolution part by part,
and ties every change to one of four root causes — decode memory, compute per
parameter, training stability, context length — because collapsing the whole
transformation into a single cause is the most common mistake in the field. Yet
putting all those root causes and the evolution side by side, no universal
"modern Transformer recipe" emerges; on the contrary, the five open-weight
models examined here each chart their own path, giving five entirely different
answers to the same four structural pressures.

---

## 1. In 2017 there was a single block

June 2017. *Attention Is All You Need* publishes one architecture with one
recipe: a six-layer encoder, a six-layer decoder, cross-attention between them.
The insides of the blocks are fixed too — eight-head attention, $d_k = d_v = 64$
per head; sinusoidal positional encoding **added** to the input embedding; a
feed-forward network 2048 wide inside, with ReLU. And around every sub-layer a
residual connection, followed by normalization. The paper writes it on one line
(Vaswani et al. 2017, §3.1): the output of each sub-layer is
$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$.

<aside class="sidenote">

Vaswani et al., *Attention Is All You Need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The block recipe is in §3.1, attention in §3.2, feed-forward in §3.3, positional encoding in §3.5. Whenever this piece says 2017, it means this paper.

</aside>

Eight years on, the items of that recipe have been displaced one by one:

- **Sinusoidal encoding is gone.** RoPE (Rotary Position Embedding) took its
  place — encoding position by rotating rather than adding. Qwen3, Gemma 3,
  OLMo 2, gpt-oss and DeepSeek all use RoPE.
- **ReLU is gone.** SwiGLU (Swish-Gated Linear Unit) took its place (Qwen3,
  gpt-oss, OLMo 2).
- **LayerNorm is gone.** RMSNorm (Root Mean Square Normalization), which never
  computes the mean at all, took its place — in all five of the models covered
  in this piece.
- **Normalization moved.** In 2017 it was applied to the sub-layer's output;
  today it is applied to its input. Gemma 3 does both, and OLMo 2 has gone back
  to 2017's placement.
- **In most models the feed-forward network is not a single network.** In
  DeepSeek-V3 every MoE layer holds 1 shared plus 256 routed experts, and nine
  of them run per token (§6.2).
- **Attention heads do not own their key and value matrices.** Either they share
  them (Qwen3, Gemma 3, gpt-oss), or they collapse into one compressed vector
  (DeepSeek-V3).

So what survived? The skeleton: the attention operation itself, residual
connections, and the idea of stacking the same block on top of itself.

### 1.1 The scope of this piece

2017's single design split into three families before long — encoder-only,
decoder-only, encoder-decoder — and
[decoder-only became the dominant paradigm over time](/en/blog/why-decoder-only/).
*Why* that split happened is the subject of a separate study.

This piece covers what came after the split: the inside of the winning block,
taken part by part. Which part is left over from 2017, which one arrived later,
and what the arrivals are answering.

### 1.2 Four pressures

Behind nearly every new part that entered the block stands a defined structural
pressure — the single exception is §8's subject:

- **Decode memory and bandwidth:** while the model produces a token, it has to
  read the key and value of every token so far out of memory. The real cost at
  inference is not the computation but this memory traffic. MQA (Multi-Query
  Attention), GQA (Grouped-Query Attention), MLA and sliding window attention
  were all developed to reduce that read load.

- **Compute per parameter:** the real cost of growing a model is measured not by
  total parameter count but by the number of parameters that *run* per token.
  Mixture-of-Experts (MoE) separates the two notions.

- **Training stability at scale:** above a certain size, training loss can
  diverge for no reason. Pre-LN (pre-layer normalization), RMSNorm, QK-Norm
  (query–key normalization) and attention sinks are the measures taken against
  that instability.

- **Context length:** a positional encoding designed for 512 tokens does not
  work at a context of 128,000 tokens. Rescaling RoPE's base frequency and the
  NoPE (No Positional Encoding) debate were both born of that need.

These four pressures have to be kept apart. MoE has no direct connection to the
KV-cache, nor QK-Norm to context length. Reducing an entire architecture to a
single cause is the most common mistake in the field.

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-1-timeline.svg" alt="A timeline running from 2016 to 2025. Components are placed by publication date and grouped by four pressures: decode memory, compute per parameter, training stability, context length. Sparsely-gated MoE sits five months to the left of the Transformer.">

<figcaption><b>Figure 1.</b> The parts of the block and when they entered (arXiv v1 submission dates). The bands show which pressure each part answers. The red mark on the left: sparsely-gated <b>MoE</b> was published five months before the Transformer — older than the block it lives in.</figcaption>

</figure>

That date in the figure is not a coincidence but a warning: not every part in
this piece is a child of 2017. Some are older ideas recalled once the right
hardware and the right scale arrived.

Start the inventory with the part that did not change.

## 2. Attention: four variants, one operation

"Attention" is the name of a single operation. The four variants in this section
are not separate mechanisms — they are the same operation invoked with different
arguments. The differences between them collect into just two things: **which
positions a query ($Q$) can look at**, and **where the key and the value ($K$,
$V$) come from**.

MQA, GQA and MLA also get called "attention variants", but they sit on a
different axis: they do not change who can look at whom, they change **how the
key and value are stored**. That axis did not exist in 2017; it opened once the
cost of generation became a problem. All three are in §4 for that reason. First,
§3 defines the problem.

### 2.1 Self-attention

This is the part that did not budge in eight years. The definition, as it stood
in 2017:

<figure>

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

<figcaption><b>Figure 2.</b> Scaled dot-product attention (Vaswani et al. 2017, §3.2.1). When query, key and value come from the same sequence, this is called self-attention.</figcaption>

</figure>

Each row of $QK^{\top}$ is a query position. That position's query is multiplied
against every key, the resulting scores pass through a softmax and turn into a
probability distribution, and that distribution takes a weighted average of the
values. In one sentence: **every position decides for itself how much
information to take from the others.**

The arguments are the frame for this whole section. Over a sequence $X$ of $n$
tokens, $Q = XW_Q$, $K = XW_K$, $V = XW_V$ — all three produced from the
**same** sequence, which is where "self" comes from. That is why the score
matrix is square: $n \times n$, every token against every token, with no cell
closed. What the remaining three variants do is depart from this setup — either
by changing the source of one of the three arguments, or by adding a mask to the
score matrix before the softmax.

On the cost side: Table 1 of the paper gives attention a per-layer complexity of
$O(n^2 \cdot d)$ — **quadratic** in the sequence, but fully parallel, with a
constant-length path between any two positions. In 2017 the winning side was
parallelism. That quadratic term has come back today, but not from the expected
direction: the binding constraint is not compute, it is memory traffic. §3 and
§4 are about that.

One last caveat: the sentence "attention has not changed since 2017" is almost
true. In 2025 gpt-oss adds a learned term to the **denominator** of the softmax
and thereby changes the operator itself. The rationale is §7.4's subject.

### 2.2 Multi-head attention

A single attention operation produces a single "point of view". 2017 multiplies
it: the same operation runs $h$ times in parallel, with different learned
projections.

The critical detail is this — the heads do not **widen** the model, they
**divide** it. The paper's own configuration (Vaswani et al. 2017, §3.2.2):
*"In this work we employ $h=8$ parallel attention layers, or heads. For each of
these we use $d_k = d_v = d_{\text{model}}/h = 64$."*

$512 = 8 \times 64$. Eight heads run at roughly the same parameter and compute
cost as one wide head; what is bought is not capacity but **diversity** — eight
different relations can be learned in eight different subspaces.

At the level of arguments there is no departure from §2.1: $Q$, $K$ and $V$
still come from the same $X$, and there is still no mask. What gets divided are
the projections. Each head runs Figure 2 independently with its own triple
$Q_i = XW_Q^{(i)}$, $K_i = XW_K^{(i)}$, $V_i = XW_V^{(i)}$, and the $h$ outputs
are concatenated and passed through a single $W_O$. Per head the score matrix is
again $n \times n$ — but computed in a subspace of dimension
$d_{\text{head}} = d_{\text{model}}/h$. Which is why §2.2 is not a "variant": it
is $h$ copies of the same operation.

The limit of that trade was measured in 2017 as well: in the ablation table the
number of heads and the width of a head are varied separately
(ibid., §6.2): *"In Table 3 rows (B), we observe that reducing the attention key
size $d_k$ hurts model quality."*

**There are models still running plain MHA (Multi-Head Attention) today.** OLMo
2 is one of them: the hyperparameter table in its technical report has a single
"Attention Heads" row (32 for 7B, 40 for 13B), no separate key/value head count,
and neither GQA nor MQA appears anywhere in the paper.

<aside class="sidenote">

OLMo team, *2 OLMo 2 Furious*, [arXiv:2501.00656](https://arxiv.org/abs/2501.00656). The configuration is in Table 4. This is a configuration observation: the paper contains **no** ablation measuring MHA against GQA, and justifies all of its architectural changes with training stability. The sentence "OLMo 2 keeps MHA because the ablations support it" is not in this source.

</aside>

That limit in the ablation is the ground for all of §4: the obvious way to make
heads cheaper is to cut $d_k$, and that road was closed in 2017. MQA and GQA
will press a different button.

### 2.3 Causal attention and prefix-invariance

The second variant moves the first axis: it restricts the positions a query can
look at. Every token sees only what came before it and itself, never what comes
after. Why this mask was chosen and how it separates the three architecture
families is [the other piece's subject](/en/blog/why-decoder-only/); the only
thing that matters here is its consequence for this one.

This time it is not the arguments that change but the scores. $Q$, $K$ and $V$
still come from the same $X$; before the softmax a mask is added to the score
matrix: $M_{ij} = 0$ if $j \le i$, and $M_{ij} = -\infty$ otherwise. With the
upper triangle at $-\infty$ the softmax produces exact zeros there and the
forward-looking path closes — the matrix is still $n \times n$, but half of it
is dead. In the code below this is a single line:
`np.where(j <= i, S, -np.inf)`.

That consequence is called **prefix-invariance**, and half of the rest of this
piece comes out of it. Plainly: **a token's key and value do not depend on what
comes after it.** Appending a new token to the sequence does not disturb the
arithmetic of the old ones.

> **Lemma.** In a stack with a causal mask, the key and value at position $j$ of
> layer $\ell$ are a function of $x_{\le j}$ alone. Therefore appending token
> $x_{t+1}$ to the sequence **does not change** $k_j^{(\ell)}$ or
> $v_j^{(\ell)}$ for any $j \le t$ and any $\ell$.

Notation: $x_{\le j}$ is the sequence's prefix up to token $j$, and
$k_j^{(\ell)}$ the key that layer $\ell$ produces at position $j$.

The proof runs by induction, layer by layer. **Base:** the input of the first
layer at position $j$ is just the embedding of $x_j$ — nothing from its
neighbours mixes in. **Step:** the output of layer $\ell$ at $j$ looks only at
the previous layer's states at positions $j' \le j$; that is exactly what the
mask does. By the induction hypothesis those states depended on $x_{\le j}$, so
the output at $j$ does too. The only place that could break the chain is the
feed-forward layer, and because it is applied to each position separately it
never mixes positions.

What happens without a mask matters too: in a stack with no mask
($M = \mathbf{0}$) the state at position $j$ depends on **all** positions —
including those to its right. Appending one token changes the key and value of
every position from the second layer onward. There is nothing stable to store.

A three-layer stack, random weights, six tokens: it runs once on the first five
and once with the sixth appended; the key/values of the first five positions are
then compared.

```python
def stack(X, causal):
    """Her katmanın K ve V'sini biriktirip döndürür."""
    kv = []
    for w in W:
        H = norm(X)
        Q, K, V = H @ w["q"], H @ w["k"], H @ w["v"]
        kv.append((K.copy(), V.copy()))
        S = Q @ K.T / np.sqrt(d)
        if causal:
            i, j = np.indices(S.shape)
            S = np.where(j <= i, S, -np.inf)
        A = np.exp(S - S.max(-1, keepdims=True))
        X = X + (A / A.sum(-1, keepdims=True)) @ V @ w["o"]
    return kv

for causal in (True, False):
    a, b = stack(x[:5], causal), stack(x[:6], causal)   # 6. token eklendi
    sapma = max(max(np.abs(bk[:5] - ak).max(), np.abs(bv[:5] - av).max())
                for (ak, av), (bk, bv) in zip(a, b))
    print(f"{'causal    ' if causal else 'çift yönlü'}  sapma: {sapma:.2e}")
```

```
causal      sapma: 0.00e+00
çift yönlü  sapma: 1.38e+00
```

That the drift is not *approximately* but **exactly** zero is the argument
itself: the causal path never reads the new token, so the arithmetic stays
identical down to the bit.

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-3-prefix-invariance.svg" alt="Two panels, each a key-value grid of three layers and six positions. Under the causal mask the first five columns are unchanged, drift zero; without the mask every column has changed, drift 1.38.">

<figcaption><b>Figure 3.</b> The same stack, before and after appending a sixth token. Under the causal mask the key and value of the first five positions stay identical <b>down to the bit</b>; remove the mask and all of them change. The zero on the left is the very thing that makes the KV-cache possible.</figcaption>

</figure>

This opens the door to a family of optimizations: if the earlier keys and values
do not change, you can store them instead of recomputing them.

### 2.4 Cross-attention

The third variant moves the second axis: the query comes from one stack, the key
and value from **another**. 2017's recipe (Vaswani et al. 2017, §3.2.3):

> "In "encoder-decoder attention" layers, the queries come from the previous
> decoder layer, and the memory keys and values come from the output of the
> encoder. This allows every position in the decoder to attend over all
> positions in the input sequence."

So what changes here is not the mask but **the key set itself**. The natural
design for translation: read the source sentence once, and return to that
reading at every step of writing the target sentence.

On the algorithm side nothing changes — only the arguments. In self-attention
$Q$, $K$ and $V$ are produced from the same $X$; in cross-attention the query
comes from the decoder's own state, the key and value from the encoder's output:
$\operatorname{Attention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$. The
product, the scaling, the softmax, the weighted sum — all of it is the formula
in Figure 2. The dimensions do not have to match either: the query sequence can be
$m$ long and the key sequence $n$ long, giving an $m \times n$ score matrix.
There is no causal mask either — because the source sequence is present from
beginning to end, every position of the decoder can see all of it.

The cost side is asymmetric: the encoder's output is fixed throughout
generation, so its keys and values can be computed **once** and stored. The
encoder runs once per input, not once per token.

That asymmetry carried cross-attention far beyond 2017's translation setting.
Query and key being able to come from separate stacks means the two can come
from separate **modalities**:

- **Audio → text.** Whisper uses 2017's encoder-decoder as it stands (Radford et
  al. 2022, §2.2): an 80-channel log-Mel spectrogram passes through an audio
  encoder, *"The encoder and decoder have the same width and number of
  transformer blocks"*, and the decoder attaches to that output in the way this
  section describes.
- **Image → text.** Flamingo interleaves cross-attention layers between the
  layers of a frozen language model (Alayrac et al. 2022, §2.2): *"We freeze the
  pretrained LM blocks, and insert gated cross-attention dense blocks... trained
  from scratch."* Visual features enter language here, and the language model's
  own weights are never touched.
- **Text → image.** Latent diffusion wires conditioning straight into this
  operation (Rombach et al. 2022, §3.3): *"We turn DMs into more flexible
  conditional image generators by augmenting their underlying UNet backbone with
  the cross-attention mechanism"*. The distribution of the arguments is written
  out there in full: $Q = W_Q^{(i)} \varphi_i(z_t)$ from the image latent,
  $K = W_K^{(i)} \tau_\theta(y)$ and $V = W_V^{(i)} \tau_\theta(y)$ from the text
  encoder. This is where Stable Diffusion listens to the prompt.

<aside class="sidenote">

Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper), [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) · Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) · Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) · Liu et al., *Visual Instruction Tuning* (LLaVA), [arXiv:2304.08485](https://arxiv.org/abs/2304.08485). Whisper's §2.2 describes the architecture as encoder-decoder but does not separately define cross-attention; the phrase "the decoder attaches to the encoder's output" here follows from 2017's encoder-decoder recipe, and is not a verbatim quotation from the Whisper paper. Flamingo's gating mechanism is in the same section: the output is multiplied by $\tanh(\alpha)$ before being added to the residual, and because $\alpha$ is initialized at zero the model at the outset produces exactly what the frozen language model itself would.

</aside>

But most of today's vision-language models do not take that road. LLaVA carries
image features straight into the word embedding space with a projection (Liu et
al. 2023, §4.1): *"We apply a trainable projection matrix $W$ to convert $Z_v$
into language embedding tokens $H_v$, which have the same dimensionality as the
word embedding space in the language model"*. The image is thereby lined up
alongside the text tokens, and everything after that is plain self-attention.
The two are two answers to one question: should a modality be **kept in a
separate stack and visited by a query**, or **joined into the same sequence and
left to a single self-attention**? The first settles for computing the encoder's
keys and values once. The second writes the image into the context length — and
hands the bill to the KV-cache, the subject of the next section.

The sentence *"only decoder-only models can use a KV-cache"* is wrong. **Every**
decoder with a causal mask uses a cache — including the decoder half of an
encoder-decoder; T5 decodes with a cache too. The correct, narrow statement is
this: **a bidirectional encoder running over a growing sequence** cannot use a
cache, because §2.3's lemma does not hold there. Encoder-decoder architectures
are not penalized on the cache side.

### 2.5 Sliding window

The fourth variant moves the first axis again, narrowing the causal mask once
more: every token can look only at the last $W$ tokens. The idea gets
systematized in Longformer (Beltagy et al. 2020) — in the paper's own words,
*"an attention mechanism that scales linearly with sequence length"* and a
component that is *"a drop-in replacement for the standard self-attention."*

What changes is again only the mask, and its pattern is §2.3's narrowed by one
step: the condition $j \le i$ becomes $i - W < j \le i$. In the causal mask the
upper triangle was closed; here the far part of the lower triangle closes too,
and what stays open is a band of width $W$ along the diagonal. $Q$, $K$ and $V$
stay where they were. Because the number of open cells per row is fixed at $W$
instead of growing with $n$, the cost stops being quadratic — this is what
Longformer means by *"scales linearly"*.

<aside class="sidenote">

Beltagy, Peters and Cohan, *Longformer: The Long-Document Transformer*, [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) · Jiang et al., *Mistral 7B*, [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) · Gemma team, *Gemma 3 Technical Report*, [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) · Brown et al., *Language Models are Few-Shot Learners* (GPT-3), [arXiv:2005.14165](https://arxiv.org/abs/2005.14165). Gemma 3's ratio ablation is in Figure 3, its window-size ablation in Figure 4; both are measured through validation-set perplexity, and by the caption's own record *"This ablation is run with text-only models."* Perplexity does not directly measure whether a distant token can be recalled when needed — the ratio's effect on long-context **retrieval** is outside the scope of that ablation.

</aside>

The counterintuitive part: the window is narrow, but the range the model can see
does not stay narrow. Mistral 7B works it out explicitly (Jiang et al. 2023,
§2): if information can travel $W$ tokens in one attention layer, then after $k$
layers it has traveled $k \times W$ tokens. With their own configuration —
$W = 4096$, 32 layers — *"At the last layer, using a window size of $W = 4096$,
we have a theoretical attention span of approximately 131K tokens."*

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-4-sliding-window.svg" alt="The querying token on the right, six bars extending to its left. 4096 tokens at one layer, 8192 at two, 16,384 at four, 32,768 at eight, 65,536 at sixteen, 131,072 at thirty-two.">

<figcaption><b>Figure 4.</b> A single layer looks directly at 4096 tokens, but the range multiplies with the layer count. At the end of Mistral's 32 layers a theoretical range of 131,072 tokens remains. The scale is linear — 24 pixels per 4096 tokens.</figcaption>

</figure>

In exchange, the number of keys/values that must be stored is capped — Mistral
does this with a *rolling buffer cache*: the key of position $i$ is written to
slot $i \bmod W$ of the cache, over the old one.

But that range is **theoretical**, and that is where the mixtures come from.
Gemma 3 interleaves global layers among the local ones, and gives its rationale
in one sentence (Gemma team 2025, §2): *"only the global layers attend to long
context, and we have 1 global for every 5 local layers."* The indirect
$k \times W$ range is not what the design relies on; the job of looking far away
still sits with the full-attention layers.

What determines how far the ratio can be pushed is not quality. The report's own
ablation varies the local:global ratio (ibid., Figure 3): *"Impact of
Local:Global ratio on the perplexity on a validation set. The impact is minimal,
even with 7-to-1 local to global."* So the mixture is not a quality rescue
operation: a few global layers are enough to carry the long context, perplexity
barely reacts when the remaining layers are turned local, and the gain collects
on the memory side — the subject of the next section. The mixtures shipping
today:

- **Gemma 3**: one global layer for every five local ones (5:1), local window
  1024 tokens. A detail that connects to §5.3: the local layers and the global
  layers use different RoPE base frequencies.
- **gpt-oss**: a 128-token banded window alternating with full dense attention.
  The pattern is inherited from GPT-3 — which says it took its
  architecture from GPT-2 and lists a single exception (Brown et al. 2020,
  §2.1): *"with the exception that we use alternating dense and locally banded
  sparse attention patterns in the layers of the transformer."* Five years
  later the same pattern comes back.

Why it is done — the cost of the cache — is the subject of the next section.

## 3. KV-cache: the trade that turns compute into memory

§2.3's lemma made storing possible; this section itemizes the bill.

### 3.1 The memory bill

The price of not storing is plain: you have to process the whole sequence from
scratch for every new token; producing $n$ tokens means attention work on the
order of $O(n^3 d)$. If you do store, at each step you only compute the new
token's projections and multiply them against the accumulated keys. This is the
optimization that makes generation possible.

The price is memory. The formula:

<figure>

$$
\text{KV bytes} = 2 \times b \times s \times L \times h_{\text{kv}} \times d_{\text{head}} \times \text{bytes}_{\text{dtype}}
$$

<figcaption><b>Figure 5.</b> The leading 2 is for key and value; <i>b</i> batch size, <i>s</i> the number of tokens so far, <i>L</i> layer count, <i>h<sub>kv</sub></i> the number of <b>key/value</b> heads, <i>d<sub>head</sub></i> the per-head dimension.</figcaption>

</figure>

Because the vLLM paper spells the product out, the formula can be repeated from
the outside (Kwon et al. 2023, §3):

> "for the 13B parameter OPT model, the KV cache of a single token demands 800
> KB of space, calculated as 2 (key and value vectors) × 5120 (hidden state
> size) × 40 (number of layers) × 2 (bytes per FP16). Since OPT can generate
> sequences up to 2048 tokens, the memory required to store the KV cache of one
> request can be as much as 1.6 GB."

<aside class="sidenote">

Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).

</aside>

```python
per_tok = 2 * 5120 * 40 * 2      # 2(K,V) × gizli boyut × katman × bayt(fp16)
print(f"token başına : {per_tok} bayt = {per_tok/1024:.0f} KiB")
print(f"2048 token   : {per_tok*2048/1e9:.3f} GB")
```

```
token başına : 819200 bayt = 800 KiB
2048 token   : 1.678 GB
```

The formula reproduces the paper's number: 819,200 bytes per token, exactly the
800 KB the paper rounds to. A small discrepancy remains in the total — when the
paper says 1.6 GB it is multiplying its own rounded 800 KB by 2048, while the raw
product is 1.678 GB. For a single request against a 13-billion-parameter model,
that is the order of magnitude.

### 3.2 The real bottleneck is bandwidth, not compute

<aside class="sidenote">

Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150). The section numbers below belong to Shazeer's paper, not to this piece. The memory/arithmetic ratio for incremental decode is in §2.4.1 and the one for training in §2.3.1; the multi-query proposal is in §3 and its own ratio in §3.1; the experiments are in §4 (Table 1 quality, Table 2 speed). The same author's two other papers in this piece are separate sources: sparsely-gated MoE (Shazeer et al. 2017) and GLU variants (Shazeer 2020); their references are in §6.2 and §8 of this piece.

</aside>

So far the cache has looked like a capacity problem; the real issue is
finer. Shazeer (2019, §2.4.1) works out the ratio of arithmetic to memory access
for incremental generation: the ratio of memory access to operation count is
$\Theta\!\left(\frac{n}{d}+\frac{1}{b}\right)$, and

> "When $n \approx d$ or $b \approx 1$, the ratio is close to 1, causing memory
> bandwidth to be a major performance bottleneck on modern computing hardware."

The notation here is Shazeer's: $n$ sequence length, $d$ model dimension, $b$
batch size, $k$ the key dimension per head — that is, Figure 5's $s$ and
$d_{\text{model}}$.

For comparison: run the same analysis for **training** and the ratio comes out
$O(1/k + 1/bn)$ (ibid., §2.3.1) — that is, very small. **Incremental decode is a
fundamentally worse hardware regime than training:** in training the chip
computes, in decode it waits.

An H100 SXM does 989.5 teraflops per second (TFLOPS) in bf16 without sparsity
and reads its HBM (High Bandwidth Memory) at 3.35 TB/s; the ratio says the chip
needs 295 operations per byte read to be saturated — the quantity the code calls
**arithmetic intensity**. With OLMo 2 7B:

<aside class="sidenote">

The chip numbers come from the spec table on NVIDIA's [H100 product page](https://www.nvidia.com/en-us/data-center/h100/). The footnote to the table's 1,979 TFLOPS says *"With sparsity"*; the dense counterpart, 989.5, is a derived value and is not listed separately. The model numbers come from the `config.json` in §8. Weight traffic is assumed read once per step; on a real server sharding and caching change this, but not the order of magnitude.

</aside>

```python
denge = 989.5e12 / 3.35e12          # H100 SXM: yoğun bf16 ÷ HBM. Çip doymak için bu kadar ister.
P, L, h_kv, d_h, s = 7.30e9, 32, 32, 128, 4096      # OLMo 2 7B — MHA, config.json

for b in (1, 16, 64):
    agirlik = 2 * P                                  # adım başına bir kez okunuyor
    kv = 2 * b * s * L * h_kv * d_h * 2              # her dizi kendi cache'ini taşıyor
    yogunluk = 2 * P * b / (agirlik + kv)
    print(f"b={b:<3d} yoğunluk {yogunluk:5.2f} FLOP/bayt"
          f"   hesap kullanımı %{100*yogunluk/denge:4.2f}"
          f"   KV trafiği %{100*kv/(agirlik+kv):.0f}")
print(f"çipi doyuran nokta: {denge:.0f} FLOP/bayt")
```

```
b=1   yoğunluk  0.87 FLOP/bayt   hesap kullanımı %0.30   KV trafiği %13
b=16  yoğunluk  4.77 FLOP/bayt   hesap kullanımı %1.62   KV trafiği %70
b=64  yoğunluk  6.15 FLOP/bayt   hesap kullanımı %2.08   KV trafiği %90
çipi doyuran nokta: 295 FLOP/bayt
```

Serving a single request, the chip uses **three thousandths** of its compute
capacity. The rest of the time it waits on memory. And the compute-utilization
column says why the obvious fix is not enough: growing the batch raises the
arithmetic intensity, but even at 64 it stays around 2%, because **KV traffic
grows with the batch too** — the weights are read once and shared across all
sequences, the cache is not. At b=64, ninety of every hundred bytes read are KV.

Pope et al. make this concrete at 500B+ scale, in a model with **multi-head
attention** (2022, §2.1): with batch 512 and
2048 tokens of context the KV-cache totals **3TB** — *"3 times the size of the
model's parameters"* — and this cache is re-read for every token generated,
*"during which the computational core of the chip is essentially idle."*

<aside class="sidenote">

Pope et al., *Efficiently Scaling Transformer Inference*, [arXiv:2211.05102](https://arxiv.org/abs/2211.05102). The 3TB figure is the value they report; the paper gives the batch and the context length, but because it does not write the model's layer count and hidden size in that sentence the product cannot be repeated from the outside — so use the vLLM example above as the arithmetic you can verify yourself.

</aside>

One common misreading has to be avoided here: **the sentence "the KV-cache makes
decode cheaper" is incomplete.** The cache *converts* a compute problem into a
memory problem. The entire MQA/GQA/MLA/PagedAttention literature exists because
the cache is expensive.

That literature is the next section.

## 4. MQA, GQA, MLA: three attacks on one factor

Look at §3's formula and the factors that can be cut are limited — $L$ and
$d_{\text{head}}$ are the model itself, $b$ is the thing that already raises
revenue. The only way to trim $s$ is §2.5's sliding window — and that is a mask
change, which does not touch how things are stored. That leaves
**$h_{\text{kv}}$**: the number of key/value heads. This section covers three
attacks on that factor, in chronological order.

### 4.1 MQA — let all heads share a single key set

Shazeer's 2019 proposal is the simplest move available (Shazeer 2019, §3):
*"Multi-query attention is identical except that the different heads share a
single set of keys and values."*

In the formula that means $h_{\text{kv}} = 1$: the cache shrinks by a factor of
$h$. The new memory/arithmetic ratio is
$\Theta\!\left(\frac{1}{d}+\frac{n}{dh}+\frac{1}{b}\right)$ (ibid., §3.1) — in
their own words, *"We have reduced the offensive $\frac{n}{d}$ by a factor of
$h$."*

The measured gain in the translation experiment (Table 2): in the baseline model
a decoder step costs **46 µs** per token, in multi-query **3.8 µs**. The encoder
barely moves (1.7 → 1.5 µs) — the gain is entirely on the decode side, which is
where the problem was.

And quality? There is a price, and the paper does not hide it (Table 1; ln(PPL)
is log-perplexity, BLEU a translation quality score — low PPL and high BLEU are
good):

| Attention | $h$ | $d_k, d_v$ | $d_{ff}$ | ln(PPL) | BLEU (dev) | BLEU test (beam 1 / 4) |
|---|---|---|---|---|---|---|
| multi-head | 8 | 128 | 4096 | 1.424 | 26.7 | 27.7 / 28.4 |
| **multi-query** | 8 | 128 | 5440 | 1.439 | 26.5 | 27.5 / **28.5** |
| multi-head | 2 | 64 | 6784 | 1.480 | 26.2 | 26.8 / 27.9 |
| multi-head | 8 | 16 | 6784 | 1.513 | 25.8 | — |

The $d_{ff}$ column is no accident: Shazeer gives back to the feed-forward what
he cuts from attention, so all four rows sit at the same parameter budget. The
quality difference does not come from a difference in size.

What the table really says is not how much MQA gives up — it is **how much the
alternatives give up.** Shazeer's own reading (ibid., §4.2): the multi-query
model *"seems to be slightly worse than the baseline, but much closer than any
of the alternatives involving decreasing $h$, $d_k$ and $d_v$."*

The obvious ways to shrink the cache are cutting the number of heads or the
width of a head; 2017 had measured that cutting $d_k$ hurts quality (§2.2). MQA
finds a third button: keep the head count, make only the **key/value** heads
singular. That is why it wins.

### 4.2 GQA — the middle ground

MQA is an aggressive cut, and in some contexts the quality loss is
unacceptable. GQA opens up every point in between (Ainslie et al. 2023, §2.2):

> "Grouped-query attention divides query heads into $G$ groups, each of which
> shares a single key head and value head. GQA-1, with a single group and
> therefore single key and value head, is equivalent to MQA, while GQA-H, with
> groups equal to number of heads, is equivalent to MHA."

A single integer, with a familiar structure at each end. And the rationale for
the chosen middle value is scale: *"larger models generally scale the number of
heads, such that multi-query attention represents a more aggressive cut in both
memory bandwidth and capacity. GQA lets us keep the same proportional decrease
in bandwidth and capacity as model size increases."*

The paper's second contribution is practical: an MHA checkpoint can be converted
to GQA without training from scratch — by **mean-pooling** the key and value
projections within each group and continuing for about 5% of the original
training steps.

The "8 groups" number you now see everywhere comes from here too — and its
grounding is weaker than assumed (ibid., §3.3): *"increasing the number of
groups from MQA only results in modest slowdowns initially, with increasing cost
as we move closer to MHA. We selected 8 groups as a favorable middle ground."*
A middle point one lab picked on its own curve; not a derived optimum.

**And the paper's own limitations section is the most honest quote in this
piece** (ibid., *Limitations*)**:**

> "Due to limited computation, we also do not compare our XXL GQA model to a
> comparitive model trained from scratch, so we do not know the relative
> performance of uptraining vs training from scratch. Finally, we evaluate the
> impact of uptraining and GQA only on **encoder-decoder** models. Recently,
> decoder-only models are extremely popular, and since these models do not have
> separate self-attention and cross-attention, we **expect** GQA to have a
> stronger advantage over MQA."

So every decoder-only use of GQA from Llama 2 onward rests on a result that was
never measured on decoder-only models. The paper says so plainly; the field
repeats the result without the caveat.

<aside class="sidenote">

Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) · Touvron et al., *Llama 2*, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — Table 1 caption: *"Bigger models — 34B and 70B — use Grouped-Query Attention (GQA) for improved inference scalability."*

</aside>

### 4.3 MLA — never store the key and the value at all

<aside class="sidenote">

DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) — **2024a** in this piece. The definition of MLA is in §2.1.2, the RoPE incompatibility and the decoupled fix in §2.1.3. DeepSeek-V3's technical report is a separate source (2024b, reference in §6.2).

</aside>

DeepSeek takes a third road. MQA and GQA reduce the number of key/value
**heads**; MLA stores neither key nor value, and keeps a single low-rank
**latent vector** in their place (DeepSeek-AI 2024a, §2.1.2): *"During inference,
MLA only needs to cache $\mathbf{c}_t^{KV}$, so its KV cache has only $d_c l$
elements, where $l$ denotes the number of layers."*

Their own comparison table, in elements per token:

| | KV cache per token | Model capability (their own labels) — not an independent measurement |
|---|---|---|
| MHA | $2 n_h d_h l$ | Strong |
| GQA | $2 n_g d_h l$ | Moderate |
| MQA | $2 d_h l$ | Weak |
| MLA | $(d_c + d_h^R)\, l \approx \frac{9}{2} d_h l$ | Stronger |

With their own settings ($d_c = 4 d_h$, $d_h^R = d_h/2$) this means:
*"its KV cache is equal to GQA with only 2.25 groups, but its performance is
stronger than MHA."* The report's abstract gives the figures: compared with
DeepSeek 67B, it *"saves 42.5% of training costs, reduces the KV cache by 93.3%,
and boosts the maximum generation throughput to 5.76 times."*

DeepSeek-V3's published configuration makes this concrete: 61 layers, 128
attention heads, 128 dimensions per head, $d_c = 512$, $d_h^R = 64$.

```python
L, n_h, d_h, d_c, d_hR = 61, 128, 128, 512, 64
mha = 2 * L * n_h * d_h          # aynı model MHA olsaydı
mla = (d_c + d_hR) * L           # gerçekte saklanan

print(f"MHA olsaydı  : {mha:,} eleman/token")
print(f"MLA (gerçek) : {mla:,} eleman/token   -> {mha/mla:.1f}x küçük")
print(f"128K bağlam, bf16, b=1 : "
      f"MLA {mla*131072*2/2**30:.1f} GiB  vs  MHA {mha*131072*2/2**30:.0f} GiB")
```

```
MHA olsaydı  : 1,998,848 eleman/token
MLA (gerçek) : 35,136 eleman/token   -> 56.9x küçük
128K bağlam, bf16, b=1 : MLA 8.6 GiB  vs  MHA 488 GiB
```

A single request's cache at 128K context: 8.6 GiB instead of 488 GiB.

<aside class="sidenote">

The "if it were MHA" row is an upper bound, not a literally realistic alternative: DeepSeek-V3's `config.json` has 128 heads × 128 dimensions per head = 16,384, while the model's `hidden_size` is 7168. In classic MHA those two numbers are kept equal (§2.2: heads do not widen the model, they divide it); since only a design that compresses key and value can afford this many heads, a model building the same head count with MHA would never have been made in the first place.

</aside>

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-6-kv-footprint.svg" alt="Four horizontal bars: MHA 1,998,848 elements and 488 GiB, GQA-8 124,928 elements and 30.5 GiB, MQA 15,616 elements and 3.8 GiB, MLA 35,136 elements and 8.6 GiB. MLA's bar is one fifty-seventh of MHA's.">

<figcaption><b>Figure 6.</b> The same model under four mechanisms. Only <b>MLA</b> shipped; the other three are that mechanism applied to DeepSeek-V3's configuration. The bars are elements per token; the GiB values are for 128K context and bf16.</figcaption>

</figure>

### 4.4 The collision with RoPE

The real detail in MLA's story is here: the method is **mathematically
incompatible** with another component chosen for entirely independent reasons.

DeepSeek-AI 2024a, §2.1.3:

> "However, RoPE is incompatible with low-rank KV compression. To be specific,
> RoPE is position-sensitive for both keys and queries. If we apply RoPE for the
> keys $k_t^C$, $W^{UK}$ in Equation 10 will be coupled with a
> position-sensitive RoPE matrix. In this way, $W^{UK}$ cannot be absorbed into
> $W^Q$ any more during inference, since a RoPE matrix related to the currently
> generating token will lie between $W^Q$ and $W^{UK}$ and matrix multiplication
> does not obey a commutative law. As a result, we must recompute the keys for
> all the prefix tokens during inference, which will significantly hinder the
> inference efficiency."

The source of MLA's speed is being able to multiply two matrices ahead of time
and store them as one. RoPE inserts a position-dependent rotation in between,
and because matrix multiplication is not commutative, that merge breaks.

Their fix is **decoupled RoPE**: carve out a separate, small group of dimensions
that carries the position information, and rotate only those. The $d_h^R = 64$
in DeepSeek-V3's configuration is exactly this — a cavity left outside the
compression, reserved for RoPE.

How an architecture actually gets designed is visible right here: two components
justified separately, colliding, and the settlement left behind as a number in a
config file.

## 5. Positional encoding: from adding to rotating

The attention operation has no notion of position. Shuffle the rows in
$\operatorname{softmax}(QK^\top)V$ and the output shuffles the same way; the
operation itself treats the sequence as a set. Position has to be injected from
outside.

### 5.1 Sinusoidal encoding, and 2017's own ablation

The original solution: build a vector of sines and cosines at different
frequencies for each position and **add it to the embedding** (Vaswani et al.
2017, §3.5):

<figure>

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

<figcaption><b>Figure 7.</b> Sinusoidal positional encoding. The wavelengths form a geometric progression from 2&pi; to 10000 &middot; 2&pi;.</figcaption>

</figure>

Why this function? The paper's rationale is relative position: *"We chose this
function because we hypothesized it would allow the model to easily learn to
attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can
be represented as a linear function of $PE_{pos}$."*

That sentence deserves underlining: what is wanted is **relative** position,
what is obtained is an absolute signal the model **might learn** to use that
way. A hope, not a guarantee.

The sentence that comes right after is a surprising one (ibid., §3.5):

> "We also experimented with using learned positional embeddings instead, and
> found that the two versions produced **nearly identical** results (see Table 3
> row (E)). We chose the sinusoidal version because it may allow the model to
> extrapolate to sequence lengths longer than the ones encountered during
> training."

So the choice was not made on quality grounds — the ablation found no
difference. It was made on **extrapolation** grounds: an axis that was entirely
speculative and unmeasurable in 2017. Eight years later it is the only axis that
matters.

### 5.2 RoPE — derived from a requirement

What makes RoPE worth telling is that it does not propose a method and then test
it. It writes a **requirement** first, then derives the one solution that
satisfies it (Su et al. 2021, §3.1). The requirement: let the query-key inner
product depend on position only through the **difference**.

$$
\langle f_q(\boldsymbol{x}_m, m),\, f_k(\boldsymbol{x}_n, n)\rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n)
$$

In two dimensions the solution is a phase factor — which in real form is plainly
a **rotation** (ibid., §3.2.1):

$$
f_{\{q,k\}}(\boldsymbol{x}_m, m) =
\begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix}
\boldsymbol{W}_{\{q,k\}}\,\boldsymbol{x}_m
$$

In their own words: *"simply rotate the affine-transformed word embedding vector
by amount of angle multiples of its position index."* Position becomes an angle.
That is the whole idea.

Generalizing to $d$ dimensions means splitting the space into $d/2$ planes and
rotating each at its own frequency: $\boldsymbol{R}^d_{\Theta,m}$ is a
block-diagonal rotation matrix and the frequencies are
$\Theta = \{\theta_i = 10000^{-2(i-1)/d}\}$.

**The constant is familiar: $10000$, Vaswani's constant.** RoPE did not change the
sines — it changed *how they enter the computation*. Multiplication instead of
addition, applied to the query and the key rather than to the embedding.

What is gained is §5.1's hope turned into an identity (ibid., §3.2.2):

$$
\boldsymbol{q}_m^{\top}\boldsymbol{k}_n =
(\boldsymbol{R}^d_{\Theta,m}\boldsymbol{W}_q\boldsymbol{x}_m)^{\top}
(\boldsymbol{R}^d_{\Theta,n}\boldsymbol{W}_k\boldsymbol{x}_n) =
\boldsymbol{x}_m^{\top}\boldsymbol{W}_q\,\boldsymbol{R}^d_{\Theta,\,n-m}\,\boldsymbol{W}_k\boldsymbol{x}_n
$$

Absolute positions go in, only $n-m$ comes out. In numbers:

```python
def R(m, d, theta=10000.0):
    """RoPE'un blok-köşegen rotasyon matrisi (Su ve ark. 2021, Denk. 15)."""
    M = np.zeros((d, d))
    for i in range(d // 2):
        a = m * theta ** (-2 * i / d)
        c, s = np.cos(a), np.sin(a)
        M[2*i:2*i+2, 2*i:2*i+2] = [[c, -s], [s, c]]
    return M

rng = np.random.default_rng(0)
d, m, n = 64, 7, 23
q, k = rng.normal(0, 1, d), rng.normal(0, 1, d)

dogrudan = (R(m, d) @ q) @ (R(n, d) @ k)   # konumları ayrı ayrı kodla
bagil    = q @ R(n - m, d) @ k             # sadece n-m'yi kullan
```

```
(R_m q)·(R_n k) = -2.3518311212
  q·R_(n-m) k   = -2.3518311212
fark            = 1.78e-15
‖q‖ = 7.315344   ‖R_m q‖ = 7.315344
```

The last line shows the second property: rotation does not change the norm. As
the paper stresses, $\boldsymbol{R}^d_\Theta$ is orthogonal — *"which ensures
stability during the process of encoding position information."*

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-8-rotate-vs-add.svg" alt="Two circles. On the left, adding a position vector to the embedding pushes the tip of the sum outside the circle. On the right, the vector is rotated at the same radius and the tip stays on the circle. Both measured norms read 7.315344.">

<figcaption><b>Figure 8.</b> Adding changes the length of the vector, rotating does not. Because RoPE's rotation matrix is <b>orthogonal</b>, norm preservation is not an observation but a guarantee.</figcaption>

</figure>

<aside class="sidenote">

Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

</aside>

### 5.3 The base frequency became a tuning knob

In RoPE's design, $10000$ was a fixed number. Today it is a hyperparameter — and
there is exactly one reason it gets moved: long context.

- **OLMo 2** (OLMo team 2025, §2.1): raises $\theta$ from $10^4$ to
  $5 \times 10^5$; the rationale is that it *"increases the resolution of
  positional encoding."*
- **Qwen3** (Qwen team 2025, §3.2): *"we increase the base frequency of RoPE from
  10,000 to 1,000,000 using the ABF [Adjusted Base Frequency] technique"*, plus
  YaRN (Yet another RoPE extensioN) and Dual Chunk Attention (DCA) at inference.
- **Gemma 3** (Gemma team 2025, §2.2): *"We increase RoPE base frequency from 10k
  to 1M on global self-attention layers, and keep the frequency of the local
  layers at 10k."*

That last item connects back to §2.5 and is interesting on its own: **two
different $\theta$ values inside one model.** The local layers see only 1024
tokens, so they have no need for a stretched frequency. Positional encoding is
no longer a global property but a setting chosen per layer type.

What exactly does raising $\theta$ buy? OLMo 2's rationale said it
*"increases the resolution"*; write the frequencies out and it turns out that is
not what happens. RoPE's fastest-rotating plane has frequency $\theta^{0} = 1$,
i.e. a wavelength of $2\pi$ — **6.28 tokens, whatever $\theta$ is.** Only the
other end moves:

```python
# theta_i = theta^(-2i/d), i = 0 .. d/2-1  ->  dalga boyu 2*pi / theta_i
band = lambda th, d: (2*np.pi, 2*np.pi * th ** ((d-2)/d))

for ad, th, d, ctx in (("Gemma 3 yerel ", 1e4, 128,   1024), ("OLMo 2 7B     ", 5e5, 128,   4096),
                       ("Gemma 3 global", 1e6, 128, 131072), ("Qwen3-235B    ", 1e6, 128,  40960),
                       ("DeepSeek-V3   ", 1e4,  64, 163840)):   # decoupled RoPE: yalnız 64 boyut
    ince, kaba = band(th, d)
    print(f"{ad} θ={th:>9.0f}  d={d:3d}   en hızlı {ince:.2f} token"
          f"   en yavaş {kaba:11,.0f}   bağlam/en yavaş {ctx/kaba:5.2f}")
```

```
Gemma 3 yerel  θ=    10000  d=128   en hızlı 6.28 token   en yavaş      54,410   bağlam/en yavaş  0.02
OLMo 2 7B      θ=   500000  d=128   en hızlı 6.28 token   en yavaş   2,559,196   bağlam/en yavaş  0.00
Gemma 3 global θ=  1000000  d=128   en hızlı 6.28 token   en yavaş   5,063,256   bağlam/en yavaş  0.03
Qwen3-235B     θ=  1000000  d=128   en hızlı 6.28 token   en yavaş   5,063,256   bağlam/en yavaş  0.01
DeepSeek-V3    θ=    10000  d= 64   en hızlı 6.28 token   en yavaş      47,117   bağlam/en yavaş  3.48
```

What is bought is not resolution but **range**: going from 10k to 1M moves the
slowest plane's wavelength from 54 thousand to 5 million, and leaves the fastest
one untouched.

The last row is a separate story, and the continuation of §4.4. DeepSeek-V3
never moved the base frequency at all — in the shipped configuration $\theta$ is
still 10,000, Vaswani's constant. On top of that, because decoupled RoPE rotates
only 64 dimensions, the exponent shrinks too and the slowest wavelength drops to
47 thousand. Across a 163,840-token context that means **3.48 full turns**: the
only one of the five whose context exceeds its slowest wavelength. The second
half of the settlement §4.4 called "a number left behind in a config file" is
here — that cavity was borrowed from the width of the position band: halving the
number of rotating planes clips the range in half too.

<aside class="sidenote">

The frequency definition is Su et al. 2021, Eq. 15. The configurations come from the models' published `config.json` files: `deepseek-ai/DeepSeek-V3` (`rope_theta` 10000, `qk_rope_head_dim` 64, `max_position_embeddings` 163840), `google/gemma-3-27b-it` (`rope_theta` 1e6, `rope_local_base_freq` 1e4), `allenai/OLMo-2-1124-7B` (5e5), `Qwen/Qwen3-235B-A22B` (1e6). A ratio below 1 does not mean YaRN is unnecessary — Qwen3 uses it too, because of training length. DeepSeek-V3 compensates at runtime: `rope_scaling` YaRN, `factor` 40, `original_max_position_embeddings` 4096.

</aside>

### 5.4 QK-Norm — and a measured price

The only component in this section that belongs to the **stability** pressure
rather than the context-length one. The idea is simple: normalize the query and
the key before the dot product.

Its origin is low-resource translation (Henry et al. 2020), but the citation the
field uses is ViT-22B (Vision Transformer). What it solves there is very clear
(Dehghani et al. 2023, §2):

> "In scaling ViT beyond prior works, we observed divergent training loss after
> a few thousand steps. In particular, this instability was observed for models
> with around 8B parameters. It was caused by extremely large values in
> attention logits, which lead to (almost one-hot) attention weights with
> near-zero entropy."

Appendix B gives the number too: without normalization the attention logits
*"quickly grow to over 50000 in magnitude."* Feed 50,000 into a softmax and what
comes out is not a distribution but a selection — and the gradient vanishes.

Qwen3, OLMo 2 and Gemma 3 all three adopt QK-Norm, and all three cite stability
as the reason. In Gemma 3's phrasing, it replaces Gemma 2's soft-capping.

**But it is not free**, and there is a controlled study that shows it. Yang et
al. (2025) train RoPE, NoPE and QK-Norm variants at 8 billion parameters under
the same recipe and compare them. Their finding (ibid., §2.2.1):

> "the RoPE and QK-Norm variants exhibit comparable performance on standard
> benchmarks... For long context evaluations, QK-Norm performs the **worst**
> among the three variants, despite its decent performance in other
> capabilities."

They explain the reason as well: normalization files down the magnitude
information in the query-key dot product; the attention logits move closer
together and the distribution flattens. The entropy they measure is markedly
higher in the QK-Norm variant — that is, attention is diffuse.

<aside class="sidenote">

Yang et al., *Rope to Nope and Back Again: A New Hybrid Attention Strategy*, [arXiv:2501.18795](https://arxiv.org/abs/2501.18795) · Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*, [arXiv:2302.05442](https://arxiv.org/abs/2302.05442) · Henry et al., *Query-Key Normalization for Transformers*, [arXiv:2010.04245](https://arxiv.org/abs/2010.04245).

</aside>

The pattern is familiar: QK-Norm rescues training, and takes something away from
long context.

### 5.5 NoPE — no encoding at all

The last possibility is to remove positional encoding entirely. It looks
counterintuitive, but there is a rationale: the causal mask already carries
position information. The first position sees one token, the second two, the
third three — the mask is itself a counter.

This is not an analogy. Kazemnejad et al. (2023) do not merely prove that the
mask *can* carry position, they construct the weights explicitly (App. C.1):
let one dimension of the embedding be 1 at every token and another be 1 only at
the first token; let the key read the first and the value read the second. Then
all keys become identical, the softmax flattens over the causal prefix, and the
output comes out exactly the reciprocal of the prefix length. With no positional
encoding:

```python
rng = np.random.default_rng(0); T, d = 8, 16
H = rng.normal(0, 1, (T, d))
H[:, 0] = 1                                   # 1. boyut: her token'da 1
H[:, 1] = [1] + [0] * (T - 1)                 # 2. boyut: yalnız ilk token'da 1

Q = H @ rng.normal(0, 1, (d, 4))              # sorgu projeksiyonu keyfî olabilir
K = H[:, [0]] @ np.ones((1, 4))               # anahtar 1. boyutu okuyor -> hepsi aynı
V = H[:, [1]]                                 # değer 2. boyutu okuyor -> yalnız ilk token 1

S = np.where(np.tril(np.ones((T, T))) == 1, Q @ K.T, -np.inf)   # causal maske, konum kodu YOK
A = np.exp(S - S.max(-1, keepdims=True)); A /= A.sum(-1, keepdims=True)
print("attention çıktısı :", np.round((A @ V).ravel(), 4))
print("1 / çıktı         :", np.round(1 / (A @ V).ravel()).astype(int))
```

```
attention çıktısı : [1.     0.5    0.3333 0.25   0.2    0.1667 0.1429 0.125 ]
1 / çıktı         : [1 2 3 4 5 6 7 8]
```

A single attention layer, given no positional signal whatsoever, carries the
rank of every position in readable form.

Kazemnejad et al. (2023) measure this systematically: in decoder-only models
trained from scratch they compare APE (Absolute Position Embedding — absolute
encoding added to the embedding), T5's relative position bias, ALiBi (Attention
with Linear Biases), RoPE and no encoding at all — NoPE — on length
generalization. The conclusion of their abstract:

<aside class="sidenote">

Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers*, [arXiv:2305.19466](https://arxiv.org/abs/2305.19466). The quote is from the abstract; the experimental setup is in §3, the comparison in §4.

</aside>


> "Our findings reveal that the most commonly used positional encoding methods,
> such as ALiBi, Rotary, and APE, are not well suited for length generalization
> in downstream tasks. More importantly, **NoPE outperforms other explicit
> positional encoding methods while requiring no additional computation.**"

The scope has to be read correctly: these are task-specific models trained from
scratch. The finding does not mean production models should drop RoPE. In Yang
et al.'s 8B comparison the NoPE variant also trails RoPE on standard benchmarks.

Even so, the same study shows that mixing the two approaches **layer by layer**
does work. Measured layer-wise, NoPE layers and RoPE layers behave in clearly
distinct ways (ibid., §2.2.3): the NoPE layers are strong at retrieval —
they put high attention mass on the token they are looking for; the RoPE layers
show a strong recency bias, leaning on the last tokens.

In the same place, the price of §5.3's tuning knob is measured too: as $\theta$
grows the recency bias of the RoPE layers decreases, but this time the widened
receptive field brings noise, and **the retrieval ability of the NoPE layers
degrades.** When $\theta$ goes from 10,000 to 4 million, the attention on the
sought token drops from 0.0765 to 0.0369 and the evaluation score from 8.036 to
6.203. Raising the base frequency is not free.

## 6. Mixture-of-Experts: separating parameters from compute

From here the pressure changes. §3–§5 were about decode memory and context
length; MoE answers something else entirely, with no connection to the KV-cache.

The question it answers: **how do you grow a model without growing the compute
done per token at the same rate?**

The idea is old: sparsely-gated MoE was published in January 2017, five months
before the Transformer; its implementation at the time sat between LSTM (Long
Short-Term Memory) layers.
In their own abstract (Shazeer et al. 2017):

> "The capacity of a neural network to absorb information is limited by its
> number of parameters. Conditional computation, where parts of the network are
> active on a per-example basis, has been proposed in theory as a way of
> dramatically increasing model capacity without a proportional increase in
> computation... achieving greater than **1000x** improvements in model capacity
> with only minor losses in computational efficiency on modern GPU clusters."

The mechanism is simple: replicate the feed-forward layer into $N$ copies
("experts"), let a small router network decide which experts run for each token,
and compute only the selected ones. Parameters go up by a factor of $N$, compute
per token stays flat.

### 6.1 How many experts, and how many on?

Switch Transformer (Fedus et al. 2021) makes the most radical simplification
(ibid., §2.1): *"Contrary to these ideas, we instead use a simplified strategy
where we route to only a single expert. We show this simplification preserves
model quality, reduces routing computation and performs better."* They report a
pre-training speedup of up to 7x over T5-Base and T5-Large at the same compute
budget.

DeepSeekMoE (Dai et al. 2024) goes the other way — more, smaller experts — and
proposes two ideas (ibid., §1): *"(1) finely segmenting the experts into $mN$
ones and activating $mK$ from them, allowing for a more flexible combination of
activated experts; (2) isolating $K_s$ experts as shared ones, aiming at
capturing common knowledge and mitigating redundancy in routed experts."*

The argument for fine granularity is not a claim but a number. Split every
expert into four and the number of selectable combinations explodes:

```python
from math import comb
print(f"C(16,2)  = {comb(16,2):,}")      # kaba taneli
print(f"C(64,8)  = {comb(64,8):,}")      # her uzman dörde bölünmüş
```

```
C(16,2)  = 120
C(64,8)  = 4,426,165,368
```

Identical to the figure the paper published. The same parameter budget, the same
compute, 36 million times more combinations.

### 6.2 What shipped

The real configurations:

| Model | Experts per MoE layer | Active per token |
|---|---|---|
| Switch Transformer (2021) | up to 2048 | **1** |
| DeepSeek-V3 (2024) | 1 shared + 256 routed | 8 + 1 = **9** |
| Qwen3-235B-A22B (2025) | 128, no shared | **8** |
| gpt-oss-120b (2025) | 128 | **4** |
| gpt-oss-20b (2025) | 32 | **4** |

DeepSeek-V3's number has to be stated correctly, because it is usually reported
wrong. The report says (DeepSeek-AI 2024b, §4.2): *"Each MoE layer consists of 1
shared expert and 256 routed experts... Among the routed experts, 8 experts will
be activated for each token."* Because the shared expert is always on, nine
experts run in total (ibid., §2.1.2). So it is not "256 experts, 9 active": there
are **257** experts per layer, and 8 routed plus the 1 always-on shared expert
run. Of the 671 billion total parameters, 37 billion run per token — **5.5%**.

<aside class="sidenote">

Shazeer et al., *Outrageously Large Neural Networks*, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) · Fedus et al., *Switch Transformers*, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) · Dai et al., *DeepSeekMoE*, [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) · DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — **2024b** in this piece · Qwen team, *Qwen3 Technical Report*, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) · OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, [arXiv:2508.10925](https://arxiv.org/abs/2508.10925).

</aside>

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-9-moe-configs.svg" alt="Scatter plot: total experts per layer on a logarithmic horizontal axis, active experts per token on the vertical axis. Switch 1 of 2048, gpt-oss-20b 4 of 32, gpt-oss-120b 4 of 128, Qwen3 8 of 128, DeepSeek-V3 9 of 257.">

<figcaption><b>Figure 9.</b> Five models, five configurations; the horizontal axis is logarithmic. The points are scattered: there is no shared decision on the total expert count or on the active count. Switch's entire thesis was top-1, and nobody is there today — but they are not with each other either.</figcaption>

</figure>

### 6.3 The contested point: the shared expert

Look at the table and it is tempting to say "the field converged on fine
granularity". The sources do not support that.

**On the shared expert, two strong labs decided the opposite way.**
DeepSeekMoE makes it one of its two main ideas. Qwen3 removes it in a single
sentence (Qwen team 2025, §2): *"Unlike Qwen2.5-MoE, the Qwen3-MoE design
excludes shared experts."* They publish no ablation for their own removal
decision.

But a third lab measures it without being the side that proposed the idea. OLMoE
trains two models side by side: active
parameters, total parameters and FLOPs identical; the only difference is that
one has 4 of 32 routed experts on, the other 1 always-on shared expert plus 3 of
31 routed. The result (Muennighoff et al. 2024, §4.1.3):

> "While both settings lead to similar performance, sharing an expert performs
> slightly worse."

What is really interesting is their reasoning, because it is §6.1's combination
argument itself — running in the opposite direction this time:

```python
from math import comb
print(f"32 uzmandan 4 aktif      : {comb(32,4):>7,}")
print(f"1 paylaşılan + 31'den 3  : {comb(31,3):>7,}")
print(f"kaybedilen kombinasyon   : {1 - comb(31,3)/comb(32,4):>7.1%}")
```

```
32 uzmandan 4 aktif      :  35,960
1 paylaşılan + 31'den 3  :   4,495
kaybedilen kombinasyon   :   87.5%
```

Both numbers match what the paper published exactly; in their own words, taking
one of the routed experts and making it shared *"eliminates almost 90% of
possible combinations."*

The pattern: whatever argument DeepSeekMoE built to **defend** fine granularity,
OLMoE turns **against** the shared expert. The same study also measures the
granularity side and confirms DeepSeekMoE (ibid., §4.1.2): quartering the expert
size and raising the count from 8 to 32 — again at constant active parameters
and constant compute — yields an improvement of *"around 10%"* on HellaSwag and
MMLU (at around 130 billion tokens). §6.1's combinatorial explosion is not a
claim but a measured effect.

<aside class="sidenote">

Muennighoff et al., *OLMoE: Open Mixture-of-Experts Language Models*, [arXiv:2409.02060](https://arxiv.org/abs/2409.02060). The shared-expert ablation is in §4.1.3 (Figure 6), fine granularity in §4.1.2 (Figure 5). In both experiments the active and total parameters and the FLOPs are held constant.

</aside>

Its limit has to be stated too: OLMoE is at 1B active / 7B total, two orders of
magnitude below DeepSeek-V3, and again on a single lab's own recipe. The other
side is not unmeasured either: DeepSeekMoE's own ablation (Dai et al. 2024, §4.4)
reports that isolating a shared expert *"yields improved performance across a
majority of benchmarks"* — but that measurement sits on the recipe of the lab
that proposed the idea. Nothing closes the question: both sides have published
their own experiment, and both inside their own recipe.

Routing has a problem of its own: it has to be balanced, or a handful of experts
take the whole load. The standard method is to add an auxiliary loss, and it is
known to cost quality — one of DeepSeek-V3's contributions is precisely getting
past this: *"an auxiliary-loss-free strategy for load balancing"*, with the
stated aim of *"minimizing the adverse impact on model performance that arises
from the effort to encourage load balancing."*

## 7. Normalization: both its place and its form changed

Normalization existed in 2017 and still does; what changed is where it sits and
what it computes. Its place changing (§7.1, §7.3) is a consequence of the
stability pressure; its form changing (§7.2) of the speed calculation, and the
term added to the softmax denominator (§7.4) of long context.

### 7.1 Post-LN and Pre-LN

Both names answer a single question: **does normalization sit on the main path,
or on the copy that goes down the branch?** Both blocks contain the same three
parts — a sub-layer (attention or FFN — feed-forward network), a residual add,
a normalization. Only the order differs.

**Post-LN** (post-layer normalization, the 2017 original): run the sub-layer,
add the residual, *then* normalize. Normalization sits **on top of** the main
path, between blocks.

$$x_{l+1} = \mathrm{LayerNorm}\big(x_l + \mathrm{Sublayer}(x_l)\big)$$

**Pre-LN** (what ships today): normalize a *copy* first, apply the sub-layer to
it, then add to the untouched residual. Normalization is **inside the branch**;
the residual path stays a clean identity from end to end.

$$x_{l+1} = x_l + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(x_l)\big)$$

That is the whole difference.

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-10-postln-preln.svg" alt="Two block diagrams, each of two layers. In Post-LN the normalization box sits on the main path and cuts the vertical residual line. In Pre-LN the normalization is inside the branch and the vertical line runs through uncut from end to end.">

<figcaption><b>Figure 10.</b> The only difference is the <b>position</b> of the normalization box. On the main path in Post-LN, inside the branch in Pre-LN — in the right-hand diagram no box cuts the vertical line. The winning form won not on quality, but because it made the warm-up stage removable.</figcaption>

</figure>

Xiong et al. (2020) show this theoretically (ibid., abstract):

> "Specifically, we prove with mean field theory that at initialization, for the
> original-designed Post-LN Transformer, which places the layer normalization
> between the residual blocks, the expected gradients of the parameters near the
> output layer are large. Therefore, using a large learning rate on those
> gradients makes the training unstable. The warm-up stage is practically
> helpful for avoiding this problem. On the other hand, our theory also shows
> that if the layer normalization is put inside the residual blocks (recently
> proposed as Pre-LN Transformer), the gradients are well-behaved at
> initialization."

The practical consequence: with Pre-LN the learning-rate warm-up stage can be
**removed**.

Pre-LN did not win on quality; it won because it eliminated a hyperparameter
whose tuning cost grows with the model. Some of the changes in this piece are
like that — not an architectural victory, an operational convenience.

Pre-LN has a visible price too: because the residual stream is never normalized
along the way, a final LayerNorm has to be added at the very top of the stack.

<aside class="sidenote">

Xiong et al., *On Layer Normalization in the Transformer Architecture*, [arXiv:2002.04745](https://arxiv.org/abs/2002.04745). The step-by-step comparison of the two blocks is in Table 1.

</aside>

### 7.2 RMSNorm

The second change is inside normalization itself. LayerNorm does two things:
subtracts the mean (re-centering) and divides by the standard deviation
(re-scaling). Zhang and Sennrich (2019) argue the first is unnecessary
(ibid., abstract):

> "we hypothesize that re-centering invariance in LayerNorm is dispensable and
> propose root mean square layer normalization, or RMSNorm... RMSNorm achieves
> comparable performance against LayerNorm but reduces the running time by
> **7%~64%** on different models."

Not computing the mean removes one statistics pass; the gain is entirely speed.
Qwen3, Gemma 3, OLMo 2 and gpt-oss all four use RMSNorm.

The range itself says something — 7%–64%, a ninefold spread, and measured on
2019's hardware, on that era's models. The paper's quality claim is not "better"
either, but
"comparable". None of the four models shipping RMSNorm publishes an ablation
against LayerNorm at its own scale — so today's ubiquity rests not on a measured
advantage but on a speedup that looks free.

### 7.3 2024–25: the placement reopened

**OLMo 2 goes back** (OLMo team 2025, §2.2): *"We normalize the outputs to the
attention and feedforward (MLP) layers within each transformer block, instead of
the inputs."* Which is close to 2017's placement — but with RMSNorm rather than
LayerNorm, and again with stability as the rationale.

**Gemma 3 does both** (Gemma team 2025, §2): *"We use a Grouped-Query
Attention (GQA) with post-norm and pre-norm with RMSNorm."* Two normalizations
in the same block.

So the sentence "Pre-LN won" was right for 2020 and too definite for today.

### 7.4 Attention sink — the counterpart of §2.1's caveat

Xiao et al. (2023) start from a strange observation: in trained models,
surprisingly high attention goes to the **first** tokens of the sequence —
regardless of whether they are meaningful. They call these tokens *attention
sinks*.

Their explanation is the most elegant piece in this article (Xiao et al. 2023,
§3.1):

> "We attribute the reason to the Softmax operation, which requires attention
> scores to sum up to one for all contextual tokens. Thus, even when the current
> query does not have a strong match in many previous tokens, the model still
> needs to allocate these unneeded attention values somewhere so it sums up to
> one."

That the softmax row sums to 1 is not a preference but the definition itself.
The models find their own way out: they use the first tokens as a garbage bin.
So much so that when using a sliding window, keeping the key/value of only
**four** initial tokens is enough to bring performance back.

And in 2025 gpt-oss walks straight out of the constraint (OpenAI 2025, §2.2):

> "Each attention head has a learned bias in the denominator of the softmax,
> similar to off-by-one attention and attention sinks, which enables the
> attention mechanism to pay no attention to any tokens."

The term shifts here: in Xiao et al. the sink is a *token*, in gpt-oss it is a
*learned number* that takes over that token's job. The denominator is no longer
just the sum of the token scores; it also contains a learned term. As a result
the weights going to real tokens **do not sum to 1.** The model can now say "I
am looking at none of them."

A constraint set in 2017, two consequences within eight years: first (2023) a
compensation the models found on their own, then an architectural change that
makes it unnecessary.

<aside class="sidenote">

Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) · Zhang and Sennrich, *Root Mean Square Layer Normalization*, [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).

</aside>

## 8. FFN: the one change whose rationale was never written

The less-discussed half of the block, but the side that holds most of the
parameters. 2017's recipe is plain (Vaswani et al. 2017, §3.3): two linear
layers with a ReLU between them, applied to each position separately. The
dimensions are $d_{\text{model}} = 512$, $d_{ff} = 2048$.

Today's counterpart is SwiGLU. Shazeer's 2020 formulation, a variant of gated
linear units (GLU):

$$
\mathrm{SwiGLU}(x, W, V) = \mathrm{Swish}_1(xW) \otimes xV, \qquad
\mathrm{Swish}_\beta(x) = x\,\sigma(\beta x)
$$

A third weight matrix arrives — the input is multiplied by both $W$ and $V$ and
the results are multiplied element-wise. And out of this comes a detail that
explains the odd numbers in modern configurations (Shazeer 2020, §2):

> "All of these layers have **three** weight matrices, as opposed to two for the
> original FFN. To keep the number of parameters and the amount of computation
> constant, we reduce the number of hidden units $d_{ff}$ (the second dimension
> of $W$ and $V$ and the first dimension of $W_2$) by a factor of $\frac{2}{3}$
> when comparing these layers to the original two-matrix version."

This is why 2017's ratio of $4 \times d_{\text{model}}$ looks more like
$\approx 8/3$ today.

In a shipped configuration both claims can be checked at once: does the FFN
really hold most of the parameters, and does the $\frac{2}{3}$ correction really
keep the budget flat. OLMo 2 7B's `config.json` gives $d_{\text{model}} = 4096$,
$d_{ff} = 11008$ and 32 layers — and its `num_key_value_heads` equals the head
count, so the MHA claim from §2.2 shows up here too.

```python
d, d_ff = 4096, 11008              # OLMo 2 7B, sevk edilen config.json

attn = 4 * d * d                   # q, k, v, o — MHA, dördü de d × d
ffn  = 3 * d * d_ff                # gate, up, down — SwiGLU üç matris
eski = 2 * d * (4 * d)             # 2017: iki matris, d_ff = 4d

print(f"attention   : {attn:>12,}   ({attn/(attn+ffn):.1%})")
print(f"FFN         : {ffn:>12,}   ({ffn/(attn+ffn):.1%})")
print(f"2017 tarifi : {eski:>12,}   SwiGLU'ya göre {ffn/eski-1:+.1%}")
print(f"8/3 · d = {8/3*d:.1f}  ->  128'in katı: {d_ff}")
```

```
attention   :   67,108,864   (33.2%)
FFN         :  135,266,304   (66.8%)
2017 tarifi :  134,217,728   SwiGLU'ya göre +0.8%
8/3 · d = 10922.7  ->  128'in katı: 11008
```

Two results. The first earns the section's opening sentence: **two thirds** of
the block's parameters sit in the FFN. Attention is the side that gets
discussed, but not the side that carries the weight.

The second is more interesting. 11008 is not an arbitrary number — it is
$\frac{8}{3} \times 4096 = 10922.7$ rounded up to a multiple of 128. The result
lands only **0.8%** away from 2017's two-matrix recipe. The third matrix does
not come free; its cost is cut out of the hidden unit count.

<aside class="sidenote">

The configuration was verified from two independent places. OLMo team 2025, Table 4 and §2.2: *"set the corresponding hidden size to approximately $\frac{8}{3}d$, but increased to the closest multiple of 128 (11,008 for our 7B model) to improve throughput."* And the model's published [`config.json`](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json): `hidden_size` 4096, `intermediate_size` 11008, `num_hidden_layers` 32, `num_attention_heads` = `num_key_value_heads` = 32. The presence of the three matrices is explicit in `transformers`' [`Olmo2MLP`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py) implementation too: `gate_proj`, `up_proj`, `down_proj` — all three with `bias=False`. The count covers only the matrices, not the embedding and output layers.

</aside>

The measured difference, with parameters and computation matched (Shazeer 2020,
six of the eight rows of Table 1; heldout log-perplexity, standard deviation in
parentheses):

| FFN variant | 65,536 steps | 524,288 steps |
|---|---|---|
| ReLU (baseline) | 1.997 (0.005) | 1.677 |
| GELU | 1.983 (0.005) | 1.679 |
| Swish | 1.994 (0.003) | 1.683 |
| GLU (gated, no Swish) | 1.982 (0.006) | 1.663 |
| GEGLU | 1.942 (0.004) | 1.633 |
| SwiGLU | 1.944 (0.010) | 1.636 |

The gain is real but modest — and its source is visible in the table. On the
long run the three ungated variants collect between 1.677 and 1.683, while the
three gated ones drop to 1.633–1.663. The difference comes not from the
activation but from the **gate**. And the 0.002 between GEGLU and SwiGLU sits
inside the measurement's own noise: SwiGLU's standard deviation is 0.010.

The real point is the paper's **closing sentence** (ibid., §4):

> "We offer no explanation as to why these architectures seem to work; we
> attribute their success, as all else, to divine benevolence."

The sentence reads like a joke, but it is one of the most honest sentences in
the field. Every *other* component in this inventory has a named pressure behind
it — cache, compute, stability, context. This one has a table and a shrug. And
it went into everything anyway, from Llama to Qwen3, from gpt-oss to OLMo 2.

Not every part of the modern block is explained. The field knows this too.

<aside class="sidenote">

Shazeer, *GLU Variants Improve Transformer*, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).

</aside>

## 9. The block as a whole: five models, five answers

All eight years show up in two things: the diagram of today's block, and how
little the five models shipping it resemble one another.

### 9.1 Today: the decoder block

2017's skeleton, filled with the parts toured in this piece:

```
x ──┬─────────────────────────────────────────┐
    │   RMSNorm  →  attention (§2, §4)         │   ← position: RoPE (§5)
    │              + causal mask (§2.3)        │
    └─────────────────────────────────────── ⊕ ┘
    │
    ├─────────────────────────────────────────┐
    │   RMSNorm  →  FFN: SwiGLU or MoE         │      (§6, §8)
    └─────────────────────────────────────── ⊕ ┘
                                              × L layers
```

Compared with 2017's decoder: the cross-attention sub-layer is gone (there is no
encoder to look at), normalization has moved and changed form, positional
encoding has migrated from the embedding into attention, the feed-forward has
gone sparse. What did not change is the skeleton itself: two sub-layers, two
residual connections, and the same block stacked on top of itself.

### 9.2 No convergence

If there were a single recipe called "the modern transformer", the
configurations of five open-weight models would resemble each other. They do
not:

| | DeepSeek-V3 | Qwen3-235B | Gemma 3 | gpt-oss-120b | OLMo 2 13B |
|---|---|---|---|---|---|
| Attention | MLA | GQA 64/4 | GQA + 5:1 local/global | GQA 64/8, banded ↔ dense | **MHA** (40 heads) |
| Position | RoPE (decoupled) | RoPE θ=1M + YaRN | RoPE 1M global / 10k local | RoPE + YaRN | RoPE θ=5e5 |
| QK-Norm | — | yes | yes | — | yes |
| Normalization | RMSNorm | RMSNorm, pre | RMSNorm, pre **and** post | RMSNorm, pre | RMSNorm, **on the output** |
| FFN | SwiGLU + MoE | SwiGLU + MoE | dense | SwiGLU + MoE | SwiGLU, dense |
| MoE | 1+256, 9 active | 128, 8 active, no shared | none | 128, top-4 | none |

Every cell comes from the model's own report. The Gemma 3 column describes not a
single size but the family; the report gives these choices as common to all
sizes. Five models, five different answers — and the points where they conflict
are not random: one keeps MHA, two use no MoE, one drops the shared expert, one
puts normalization in two places at once, one returns to 2017's placement.

What converged is not the component list. What converged is the **problem
list**: all five are wrestling with the same four pressures, and making
different trades in different places.

### 9.3 Where this piece is weak

Almost all of the table above and of the 2024–25 numbers in this piece come from
**technical reports**, not controlled experiments. Model cards say what shipped,
not what was isolated.

There are places where an ablation exists — Gemma 3's local/global ratio,
DeepSeek-V2's MHA/GQA/MQA comparison, SwiGLU's Table 1, Yang et al.'s 8B
RoPE/NoPE/QK-Norm study — but each of those is on a single lab's own recipe. An
independent, broad study comparing the components at equal budget does **not
exist** in this piece's evidence base. Because nobody has done it: the cost of
training a frontier model twice, changing one component, is higher than the
price of wondering what the answer is.

So this piece can claim the following: every change has a named pressure and a
primary source documenting that pressure — except SwiGLU (§8), whose primary
source explicitly declines to give one. It cannot claim this: that the choices
made are the **best** answers available to those pressures. That second one was
never measured.
