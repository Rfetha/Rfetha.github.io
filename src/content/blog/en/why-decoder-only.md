---
title: 'Why is everyone using decoder-only?'
description: 'What decoder-only is, and why the field converged on it — five candidate answers tested against the primary sources.'
pubDate: '2026-09-02'
---

## Abstract

Every large language model you use today shares one architecture:
decoder-only. The usual reason given is "because it is better." But that
answer has a problem — the one clean experiment that tested this question at
equal compute (T5, 2019) found the opposite: encoder-decoder won seven tasks
out of seven, and decoder-only was the worst structure tested. This post
answers two questions in order. First, *what exactly decoder-only is* — that
the difference between three architecture families comes down to a single
attention mask, concrete enough to be shown in twelve lines of code. Then,
*if it is not better, why is it the one in use* — by testing five candidate
answers in turn (BERT cannot generate, encoder-decoder is worse, decoder-only
scales better, bidirectionality is harmful, decoder-only is simpler) against
the primary sources. One survives, but not for the reason usually given; two
collapse; one turns out partly right; one was never demonstrated at all.
What remains is the real answer: not quality, but option value.

---

## 1. The losing experiment

October 2019. A team at Google Research publishes a vast paper sweeping
transfer learning end to end. It is called T5. The famous part is this: you
can cast every NLP task into "text in, text out."

But in §3.2 of that paper there is something almost nobody talks about
today. There the team runs an experiment nobody had done properly until
then: it puts the architectures side by side at **equal compute budget**.
Encoder-decoder, decoder-only, prefix-LM. Same FLOPs, same data, same
evaluation.

The result is clear. Encoder-decoder wins seven tasks out of seven.
Decoder-only is the **worst** structure on the table — 19.7 points behind on
SQuAD.

<aside class="sidenote">

Raffel et al., *Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer* (T5),
[arXiv:1910.10683](https://arxiv.org/abs/1910.10683). The architecture
comparison is in §3.2, the numbers in Table 2, the quoted result in §3.2.4.

</aside>

The team does not hide it; they write it into the paper (Raffel et al. 2019,
§3.2.4): *"For all tasks, the encoder-decoder architecture with the denoising
objective performed best."* And they give their own recommendation: use
encoder-decoder.

Four years later, nearly every large language model in production was the
architecture that lost that experiment.

When you talk to a chat model today, when you use a coding assistant, when
you send a request to an API — the thing on the other side is decoder-only.
GPT, Gemini, Claude, Llama, DeepSeek, Qwen. All from the same family.

This post takes a simple question seriously: **if it lost that experiment,
why is it the one in use?**

Before getting to the answer, there is a more basic question to settle: what
exactly is decoder-only?

---

## 2. What does decoder-only mean?

<aside class="sidenote">

The three families: Vaswani et al., *Attention Is All You Need*,
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · Devlin et al., *BERT*,
[arXiv:1810.04805](https://arxiv.org/abs/1810.04805) · Radford et al.,
*Improving Language Understanding by Generative Pre-Training* (GPT-1) — not
on arXiv, [OpenAI technical
report](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

</aside>

In 2017 there was a single Transformer — the encoder-decoder of *Attention Is
All You Need*, designed for machine translation. The next two years split it
into three families. All three need to be understood properly, because the
rest of this post rests on the difference between them.

Start with the original, since everything after it is a cut of this:

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-0-transformer-2017.svg" alt="The full architecture of the 2017 Transformer: self-attention and feed-forward sublayers in the encoder stack; masked self-attention, cross-attention reading the encoder output, and feed-forward in the decoder stack; linear and softmax on top.">

<figcaption><b>Figure 0.</b> The original Transformer (Vaswani et al., 2017). Two stacks, with a residual connection and normalization after each sublayer. The <b>cross-attention</b> sublayer in the middle of the decoder reads the encoder's output as keys and values — that box is the one this post will discuss most from here on.</figcaption>

</figure>

### Encoder-only (BERT, 2018)

A single self-attention stack. Its defining property: **every token can see
every token in the sequence** — the ones before it and the ones after it.

You give it a sentence; the model produces, for each word, a representation
*enriched by context from both directions*. In "I went to the bank," the
representation of "bank" is fed by the words before it and after it — money
bank or river bank, the whole sentence decides.

Its output is not token generation but **representation**: one vector per
position in the sequence. You put a small head on top of those vectors and
classify, or you use the vectors directly for search and retrieval. BERT was
designed for this and is still used for it.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1a-encoder-only.svg" alt="Encoder-only architecture: input tokens pass through a fully-visible self-attention stack and produce one representation vector per position, with a classification head on top.">

<figcaption><b>Figure 1a.</b> Encoder-only. One stack, fully-visible mask: every token sees every token. The output is not a token but a representation.</figcaption>

</figure>

### Decoder-only (GPT-1, 2018)

One stack again — but with a constraint: **every token can see only the ones
before it.** In GPT-1's own description (Radford et al. 2018, §4.1): "a
12-layer decoder-only transformer with masked self-attention heads."

The constraint is not arbitrary; it defines the model's job. At every
position the model answers a single question: *I have seen this much, what is
the next token?* Training is nothing more than that — over a vast pile of
text, predict the next token at every position. Generation is the same
operation repeated from the outside: produce a token, append it to the input,
ask again.

Two things to notice: first, the training signal comes from **every token** —
every position in the sequence is both context and target at once. Second,
the operation during training and the operation during generation are **the
same operation** — the model works by doing exactly what it was trained on.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1b-decoder-only.svg" alt="Decoder-only architecture: input tokens pass through a causal self-attention stack and produce a next-token distribution; the sampled token is appended to the input and the loop repeats.">

<figcaption><b>Figure 1b.</b> Decoder-only. One stack, causal mask. The operation during training and the operation during generation are identical: predict the next token, append, ask again.</figcaption>

</figure>

Opening it up makes the comparison with the decoder column of Figure 0
easier:

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1e-decoder-block.svg" alt="The full interior of the decoder-only block: token embedding plus positional encoding, then N times masked self-attention, add and norm, feed-forward, add and norm; linear and softmax on top. The cross-attention sublayer of the 2017 decoder is absent.">

<figcaption><b>Figure 1e.</b> The decoder-only block in full. Identical to Figure 0's decoder — with one sublayer missing: <b>cross-attention</b> has been removed, because there is no encoder to look at.</figcaption>

</figure>

### Encoder-decoder (the original Transformer, later T5)

Two separate stacks, dividing the labor. The **encoder** reads the input —
fully-visible, every token sees every token, just like BERT. The **decoder**
produces the output — causal, just like GPT. A third mechanism links them:
**cross-attention**. At every generation step the decoder looks not only at
what it has produced but also at the encoder's output — the query comes from
the decoder, the keys and values from the encoder.

This is the natural design for translation: read the source sentence once,
bidirectionally, in full context; write the target sentence left to right; at
every word while writing, turn back and look at the source. "Read" and
"write" are separate modules.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1c-encoder-decoder.svg" alt="Encoder-decoder architecture: a fully-visible encoder reads the input, a causal decoder produces the output, and the decoder looks at the encoder representations through cross-attention.">

<figcaption><b>Figure 1c.</b> Encoder-decoder. Two stacks and one bridge: in cross-attention the query comes from the decoder, the keys and values from the encoder. "Read" and "write" are separate modules.</figcaption>

</figure>

### And a hybrid: prefix-LM

One stack, mixed mask: in the "prefix" at the start of the sequence everyone
sees everyone (like an encoder), after that it is causal (like a decoder).
For now it looks like a footnote; in §4 it will turn out to be the most
important row in the table.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1d-prefix-lm.svg" alt="Prefix-LM architecture: one stack, mixed mask — fully-visible over the prefix tokens, causal from there on.">

<figcaption><b>Figure 1d.</b> Prefix-LM. One stack, mixed mask. A single parameter — <code>prefix_len</code> — determines this hybrid.</figcaption>

</figure>

### All three are cuts of the same design

I described the four separately, but they are not separate designs. They are
cuts taken from the same 2017 block:

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1f-three-cuts.svg" alt="The three families as cuts of a single design: encoder-only keeps the encoder block as is, decoder-only keeps the decoder block with the cross-attention sublayer removed, encoder-decoder keeps both stacks together with cross-attention.">

<figcaption><b>Figure 1f.</b> Three cuts. None of them invents a new mechanism — encoder-only takes the encoder block, decoder-only takes the decoder block and throws away cross-attention, encoder-decoder keeps both intact.</figcaption>

</figure>

### All the same formula

Now the surprising part. All four of these structures use the **same**
attention formula:

$$
\operatorname{Attention}_M(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V
$$

This is exactly what the literature calls **masked attention** — the "masked
self-attention" in GPT-1's own description is this operator too. The
difference sits in one place: $M$, the mask that says who may look at whom.
To the disallowed positions you add $-\infty$, **before** the softmax.

The detail here matters: adding $-\infty$ is not the same as zeroing the
weights afterwards. When you add it beforehand, the softmax **renormalizes
over the allowed subset**; had you zeroed afterwards, the row sum would stay
below 1. Two lines of code show it:

```python
def attention_weights(Q, K, M):
    s = Q @ K.T / np.sqrt(Q.shape[-1]) + M
    w = np.exp(s - s.max(-1, keepdims=True))
    return w / w.sum(-1, keepdims=True)

n = 4
i, j = np.indices((n, n))
causal = np.where(j <= i, 0.0, -np.inf)

dogru  = attention_weights(Q, K, causal)                    # maske softmax'tan önce
yanlis = attention_weights(Q, K, np.zeros((n, n))) * (j <= i)  # sonradan sıfırlama

dogru.sum(-1)    # [1.    1.    1.    1.   ]  <- her satır bir olasılık dağılımı
yanlis.sum(-1)   # [0.302 0.398 0.941 1.   ]  <- artık dağılım değil
```

In the first line the model has a full attention budget distributed among the
past tokens. In the second, the first position has thrown 70% of its
attention away.

All three masks come out of a single function:

```python
import numpy as np

def additive_mask(n, prefix_len=0):
    """0 = izinli, -inf = yasak.
    prefix_len=0 -> causal (GPT)      prefix_len=n -> fully-visible (BERT)
    0<prefix_len<n -> prefix-LM (T5)"""
    i, j = np.indices((n, n))
    allowed = (j <= i) | (j < prefix_len)
    return np.where(allowed, 0.0, -np.inf)
```

A single integer. `prefix_len=0` gives you GPT, `prefix_len=n` gives you
BERT. What gets called "unidirectional" and "bidirectional" are the two ends
of one parameter.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-2-masks.svg" alt="Three attention masks over the same six-token sequence: fully-visible (all cells open), causal (lower triangle open), prefix-causal (the first two columns open to all rows, causal after that).">

<figcaption><b>Figure 2.</b> Three masks over the same sequence. Row: query position, column: key position. An open cell is 0, a closed cell &minus;&infin;. The only difference among the three is <code>prefix_len</code>.</figcaption>

</figure>

There is a limit here I have to be honest about, because this claim is
usually pushed too far. The sentence "the only difference is the mask" is
true for *a single self-attention operation inside a single stack* — for
encoder-only, decoder-only, prefix-LM. Encoder-decoder does not fall into
that category: there are two separate stacks, and in cross-attention the
query comes from the decoder while the keys and values come from the encoder.
Not a different mask over the same key set; a **different key set**.

So much for the definition. Now for the real question: why choose this mask?

---

## 3. First answer: "because BERT cannot generate"

This is the most common answer. And half of it is right — but the reason
given is wrong.

<aside class="sidenote">

The four sources this section rests on: Wang and Cho, *BERT has a Mouth, and
It Must Speak*, [arXiv:1902.04094](https://arxiv.org/abs/1902.04094) — and
Cho's own
[erratum](https://kyunghyuncho.me/bert-has-a-mouth-and-must-speak-but-it-is-not-an-mrf/)
· Goyal, Dyer and Berg-Kirkpatrick, *Exposing the Implicit Energy Networks
behind Masked Language Models via Metropolis–Hastings*,
[arXiv:2106.02736](https://arxiv.org/abs/2106.02736) · Young et al.,
*Inconsistencies in Masked Language Models*,
[arXiv:2301.00068](https://arxiv.org/abs/2301.00068) · Torroba Hennigen and
Kim, *Deriving Language Models from Masked Language Models*,
[arXiv:2305.15501](https://arxiv.org/abs/2305.15501).

</aside>

First you need to know how BERT is trained. The method is called **masked
language modeling (MLM)**: part of the input sequence — typically 15% — is
hidden behind a `[MASK]` symbol, and the model predicts the hidden tokens
**from context arriving in both directions**. "Today the [MASK] was
beautiful" → fill in the masked word by looking both before and after it.
This is what bidirectional reading looks like at training time.

Decoder-only training, by contrast, is **autoregressive**: predict the next
token from the past alone. The difference in short: **MLM teaches reading,
autoregressive training teaches generating** — because in the second, the
operation during training is identical to the operation during generation.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-3b-mlm-vs-ar.svg" alt="The same sentence under two training objectives. Under MLM roughly fifteen percent of the tokens are masked, context arrives from both directions, and the loss applies only at masked positions. Under autoregressive training every token is predicted from the ones before it and the loss applies at every position.">

<figcaption><b>Figure 3b.</b> Two objectives, one sentence. Under MLM the signal comes only from the masked positions, and the input carries a <code>[MASK]</code> symbol that never appears in real text. Under autoregressive training every position is a target, and training is a literal rehearsal of generation.</figcaption>

</figure>

Now to the common reason given: "BERT cannot generate left to right because
it is bidirectional." That is wrong twice over. Bidirectionality is no
obstacle — prefix-LMs and encoder-decoders read the input bidirectionally and
generate perfectly well.

The real obstacle runs deeper: **MLM's training objective does not define a
valid probability distribution over sequences.**

Causal LMs do not have this problem, because what happens there is an
identity:

$$
\log p(x) = \sum_{i=1}^{n}\log p(x_i \mid x_{<i})
$$

This is not a modeling assumption; it is the chain rule of probability. Once
the model learns these conditionals, the joint distribution emerges *by
construction*. Sampling from it is single-pass and guaranteed.

What MLM learns, on the other hand, is a set of conditionals:
$\{p(x_i \mid x_{\setminus i})\}$. Each one is a valid distribution on its
own. But for a set of conditionals to correspond to a joint distribution is
**an additional condition** — consistency — and MLM training enforces it
nowhere.

Goyal, Dyer and Berg-Kirkpatrick (2021, §2.1) show this is not a theoretical
worry, with a two-variable counterexample:

```python
import numpy as np

# MLM'nin bağımsız eğitilmiş iki "conditional"ı:
p_x1_given_x2 = np.array([[0.99, 0.01],
                           [0.01, 0.99]])
p_x2_given_x1 = np.array([[0.5, 0.5],
                           [0.5, 0.5]])

# Ortak bir dağılım varsa, p(a,a)/p(a,b) oranı
# hangi conditional'dan hesaplanırsa hesaplansın aynı çıkmalı:
odds_via_x1 = p_x1_given_x2[0, 0] / p_x1_given_x2[0, 1]   # 99.0
odds_via_x2 = p_x2_given_x1[0, 0] / p_x2_given_x1[1, 0]   # 1.0

assert not np.isclose(odds_via_x1, odds_via_x2)
# 99.0 != 1.0 -> bu iki conditional hiçbir ortak dağılımdan gelemez.
```

Intuitively: $X_2$ says "I am independent of $X_1$," and $X_1$ says "I am
almost a copy of $X_2$." Independence is symmetric; both cannot be true at
once.

So what happens if you try to sample from these conditionals anyway? You
write a Gibbs sampler: you redraw one variable at a time, conditioned on the
other. The chain converges somewhere — but where it converges depends on the
order in which you sweep:

```python
# Aynı iki conditional, iki farklı tarama sırası.
# Sistematik: her turda önce X1, sonra X2.
# Rastgele:   her turda yazı-tura ile birini seç.
sistematik = stationary(transition_systematic(p_x1_given_x2, p_x2_given_x1))
rastgele   = stationary(transition_random(p_x1_given_x2, p_x2_given_x1))

#             (a,a)   (a,b)   (b,a)   (b,b)
sistematik  # [0.250  0.250  0.250  0.250]
rastgele    # [0.373  0.128  0.128  0.373]
```

Same model, same conditionals, two different answers. The systematic sweep
says "all four states are equally likely"; the random sweep says "the
diagonal is three times more likely." Neither is wrong, because there is no
right one: there is no target distribution that these chains are trying to
approach.

In an autoregressive model the question does not even arise. The chain rule
hands you a joint distribution; sampling from it is single-pass, there is no
such notion as sweep order, and you do not wait for convergence.

<aside class="sidenote">

**How the correction unfolded.** 2019: Wang and Cho say "BERT is a Markov
random field" and derive a Gibbs sampler from it; for years this is the
standard citation. Then Cho announces the error on his own blog — the
factorization of the potentials is wrong, because changing one token also
changes all the other potentials. 2021: Goyal et al. deliver the formal
refutation; these conditionals "need not correspond to any consistent MRF."
2023: Young et al. show empirically that the inconsistency really is there,
from BERT-base all the way to UL2-20B.

</aside>

How this story unfolded in the literature is instructive in itself, because
it went exactly the way it should: a claim is made, one of the authors
announces the error himself, the formal refutation follows, and empirical
confirmation comes last.

To be fair, Wang and Cho's empirical contribution was not retracted —
iterative sampling from BERT really does produce fluent text. What was
retracted is the claim "therefore this is a legitimate sampler."

And the perplexity issue falls out of the same place: the chain converges to
*something*, but nobody can say what. If $\log p(x)$ is undefined, so is
perplexity.

**Scorecard:** This answer holds. Encoder-only architectures really are
unsuited to generative language modeling — but the reason is not "being
bidirectional," it is "not defining a valid joint distribution."

That only solves half the problem, though. Encoder-only is eliminated. What
about encoder-decoder? It generates too, and generates well.

---

## 4. Second answer: "because encoder-decoder is worse"

This is where the experiment from §1 comes back.

The T5 team crossed six structures with two pretraining objectives and held
all of them at the same FLOP cost. At equal budget, with the denoising
objective:

| | GLUE | SQuAD | SuperGLUE |
|---|---|---|---|
| Encoder-decoder | **83.28** | **80.88** | **71.36** |
| Prefix-LM | 81.82 | 78.94 | 68.11 |
| Decoder-only | **74.70** | **61.14** | **55.02** |

This answer is not merely wrong; the **exact opposite** is true. Wherever it
could be measured, encoder-decoder won.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-3-t5-table2.svg" alt="T5 Table 2, at equal compute budget: on GLUE encoder-decoder 83.28, prefix-LM 81.82, decoder-only 74.70; on SQuAD 80.88, 78.94 and 61.14 respectively.">

<figcaption><b>Figure 3.</b> T5 §3.2, Table 2 — equal FLOPs, denoising objective. The 19.7-point gap on SQuAD is the largest split in the table; prefix-LM closes almost all of it with a single change of mask.</figcaption>

</figure>

There is something in the table more interesting than that. Look at
prefix-LM: it has the same number of stacks as decoder-only, yet it closes
almost the entire gap (61.14 → 78.94 on SQuAD). The only difference is the
mask. So the penalty was not "having one stack" — the penalty was **reading
the input through a causal mask too**. The `prefix_len` parameter from §2 is
the explanation for 17.8 points in that table.

### If the experiment was right, how did the reality invert?

This is the crucial part, because waving it away with "so the experiment was
wrong" is as mistaken as waving it away with "so the industry was stupid."
The experiment was right — but it was **measuring the wrong game.**

Look at its two limits. First: the models in this experiment are in the
~110M parameter class. See how small that is, step by step: BERT-base is this
size. GPT-2 is ~14× larger (1.5B). The open 7B models now called "small" are
~64×. GPT-3 is ~1600× (175B). In other words, the field gathered
the evidence for its frontier decision at one-thousandth of frontier scale.

Second and more important: every number in that table was measured **after
task-specific fine-tuning**. But the game that got won in the field was an
entirely different one: making the model do a task with nothing but a prompt,
no fine-tuning at all. That game — zero-shot / few-shot — was **never
measured** in T5's experiment.

The first large controlled study that measured it arrived in 2022 (Wang et
al. 2022, Finding 1) and read: *"Causal decoder-only models pretrained with a
full language modeling objective achieve best zero-shot generalization when
evaluated immediately after unsupervised pretraining."* So decoder-only
**really does win its own game.**

<aside class="sidenote">

This finding should not be overloaded: the paper itself says the result rests
largely on a prompt set designed for causal decoders. And once fine-tuning
enters the picture, the winner flips back to encoder-decoder — more on that
in §8.

</aside>

In short: two experiments, two games, two different winners. The industry
picked which game it would play; after that choice, T5's table did not become
invalid but it did become **irrelevant**.

**Scorecard:** The second answer collapses. Decoder-only is not chosen for
its measured quality — the quality table was measured in the game it lost,
and the game it won was a different one.

---

## 5. Third answer: "because it scales better"

Once the quality argument falls, this is where everyone turns. And the
field's two most influential papers are assumed to stand behind this answer:
the scaling laws.

In 2020 Kaplan et al. showed this: language model loss falls with parameter
count, data size and compute along **smooth power laws** — across six to
eight orders of magnitude, with no sign of deviation. And performance
"depends most strongly on scale" and "very weakly on model shape": depth,
width, head count and other architectural details are nearly irrelevant next
to total scale. Large models are also more sample-efficient. That paper became
the investment map for the next five years: scale up, scale the data, ignore
the rest.

In 2022 Chinchilla corrected the balance (Hoffmann et al. 2022): by training
more than 400 models from 70 million to 16 billion parameters, it showed that
parameters and tokens should be scaled **equally**. The demonstration was
striking: Chinchilla at 70B, trained on the same compute budget,
"uniformly and significantly" outperformed the 280B Gopher, the 175B GPT-3,
and the 530B Megatron-Turing.

Now the question: where in these two papers does it say "decoder-only scales
better"?

<aside class="sidenote">

Kaplan et al., *Scaling Laws for Neural Language Models*,
[arXiv:2001.08361](https://arxiv.org/abs/2001.08361) — the scope statement is
in §2 · Hoffmann et al., *Training Compute-Optimal Large Language Models*
(Chinchilla), [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) · Tay et
al., *Scaling Laws vs Model Architectures*,
[arXiv:2207.10551](https://arxiv.org/abs/2207.10551).

</aside>

Nowhere. Kaplan's own scope statement (Kaplan et al. 2020, §2): *"We
primarily train decoder-only Transformer models."* The comparison arms are
LSTMs and Universal Transformers — no encoder-only, no encoder-decoder.
Chinchilla's 400 models are decoder-only without a single exception. These
are not architecture comparisons; they are scaling laws **inside
decoder-only**. The experiment the sentence "scales better" requires — laying
two families' scaling curves side by side — was never run in these papers.

<aside class="sidenote">

**What are the comparison arms?** The LSTM is the classic recurrent network
that walks the sequence position by position, updating its state. The
Universal Transformer (Dehghani et al., *Universal Transformers*,
[arXiv:1807.03819](https://arxiv.org/abs/1807.03819)) moves recurrence from
time to depth: it applies the same layer over and over with shared weights,
while still processing all positions in parallel — in their own words, a
"parallel-in-time self-attentive recurrent sequence model." On top of that it
adds a "dynamic per-position halting mechanism"; each token decides for itself
how many rounds it gets processed. They also claim it is Turing-complete under
certain assumptions.

</aside>

It is worth looking at what the comparison arms are, because they show which
axis was being measured. The LSTM does its recurrence in time, the Universal
Transformer in depth, the Transformer not at all. So the axis Kaplan actually
varied was "how do you process the sequence." The axis of the question here is
a different one — how many stacks are there and who sees whom. That axis never
appeared in those plots.

There is a nice irony here too. What was Kaplan's headline finding?
Architectural details are negligible next to scale (ibid., §1.1). The paper
that launched the scale-maximalist program is the paper arguing architecture
is *not* the lever. If you believe that paper, decoder-only's victory cannot
by definition be an architectural quality victory.

So did anyone systematically compare architecture against scale? Yes — Tay et
al. (2022), ten architectures, more than 100 models. What they found is the
opposite of "architecture doesn't matter": *"We find that this scaling
coefficient differs greatly from model to model."* Moreover, the best model
changes depending on which compute region you are in, and upstream perplexity
may not predict downstream transfer well.

**Scorecard:** The third answer collapses too. The strongest true sentence
you can say is this: *decoder-only was never shown to scale better;
decoder-only was the family whose scaling got measured, funded and tooled.*

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-4-scaling.svg" alt="Schematic log-log scaling curve: loss falls in a straight line with compute. Only the decoder-only family has been measured in the plot; the comparison curve was never drawn.">

<figcaption><b>Figure 4.</b> Schematic. The only family the scaling laws measured is decoder-only; the dashed line is not a measurement but the place where an experiment was never run.</figcaption>

</figure>

---

## 6. Fourth answer: "because bidirectionality is harmful"

The last candidate. Perhaps bidirectional attention actively hurts a model
that generates?

<aside class="sidenote">

Artetxe et al., *On the Role of Bidirectionality in Language Model
Pre-Training*, [arXiv:2205.11726](https://arxiv.org/abs/2205.11726).

</aside>

First, what each side provides, because only then does this section's
conclusion make sense.

**What does bidirectionality provide?** Every position's representation is
fed by the whole sequence. A word's meaning often becomes clear only after it
— a bidirectional model uses that directly. Hence its natural advantage on
comprehension-heavy work (classification, fine-tuning for question answering,
infilling): the representation is built while seeing all the information
available.

**What does left-to-rightness provide?** Two things. First, training and
generation are the same operation — the model is trained on exactly the job
it will do in deployment, with no training-only artifacts like `[MASK]` in
between. Second, every token is a training target: the whole sequence
produces signal. In return, every position has to decide without seeing the
future.

So the question is not "which one is right"; it is **"which one, for which
job"**. And that is a tested question. Artetxe et al. (2022) did exactly
this — and separated *bidirectional context* from *bidirectional attention*,
controlling each one individually, up to 6.7B parameters. What they found
(ibid., abstract):

> "We find that the optimal configuration is largely application-dependent
> (e.g., bidirectional attention is beneficial for fine-tuning and infilling,
> but harmful for next token prediction and zero-shot priming)."

Exactly the division of labor above: bidirectionality gains on comprehension
work and loses on generation-plus-prompt work. And these differences "remain
consistent at scale" — so the "it closes as you scale up" defense is not
supported by this data either. The authors' own conclusion: focusing on
left-to-right models "comes with some trade-offs, and it might be worthwhile
to develop very large bidirectional models."

**Scorecard:** Partly holds. Bidirectionality really does not help in the
winning scenario — free generation plus prompting. But it was not refuted; it
simply did not fit that scenario. Then that scenario swallowed everything
else.

---

## 7. Fifth answer: "because it is simpler, easier to serve"

After four candidates are eliminated, the fifth explanation that comes to mind
usually comes from engineering: forget quality, decoder-only is just
**easier**. One stack, one code path, one graph. Encoder-decoder means two
stacks, cross-attention, two separate optimization problems. Who would want
that in production?

The intuitive appeal of this answer is high. The trouble is that when I went
looking, I could not find a primary source supporting it — no comparative
serving-cost measurement, no latency benchmark. What is left is a frequently
repeated piece of engineering folklore.

<aside class="sidenote">

Zhang et al., *Encoder-Decoder Gemma: Improving the Quality-Efficiency
Trade-Off via Adaptation* (T5Gemma),
[arXiv:2504.06225](https://arxiv.org/abs/2504.06225).

</aside>

What stands against it is not a frequently repeated thing but a paper that
studies the question directly. T5Gemma (Zhang et al. 2025) says the opposite:
encoder-decoder models are "still widely adopted in real-world applications
for their **inference efficiency** and richer encoder representation" — and
they report that at a similar inference budget these models do better than
their decoder-only counterparts.

So at the level of public evidence at least, the claim that
"encoder-decoder is harder to serve" hangs in the air. It may be true; but it
was not shown, and what has been shown points the other way.

**Scorecard:** Saying it collapses would be too strong — more precisely, it
was **never demonstrated**, and what has been demonstrated runs against it. It
cannot be used as the real reason behind an architectural choice.

---

## 8. The real answer: option value

Three of four candidates collapsed, one was left half-standing. What remains?

The best answer actually sits in the finding of the paper that investigated
this very question — and it is not a quality finding, it is an **economics**
finding.

<aside class="sidenote">

Wang, Roberts et al., *What Language Model Architecture and Pretraining
Objective Work Best for Zero-Shot Generalization?*,
[arXiv:2204.05832](https://arxiv.org/abs/2204.05832) — the third finding is in
§5, the reverse-direction attempt in §E.4 · Brown et al., *Language Models are
Few-Shot Learners* (GPT-3),
[arXiv:2005.14165](https://arxiv.org/abs/2005.14165) · Das et al., *A
decoder-only foundation model for time-series forecasting* (TimesFM),
[arXiv:2310.10688](https://arxiv.org/abs/2310.10688).

</aside>

Wang et al. (2022, BigScience) crossed three architectures with three
objectives and trained them at 5B/11B scale. Their first two findings are the
ones from §4, and they were the inverse of one another: measured
zero-shot right after pretraining, the causal decoder wins; measured after
multitask fine-tuning, encoder-decoder + MLM wins, and by a wide margin. In
their own words: "the best objective and architecture is the opposite in these
two settings."

The real point is their third finding:

> "Decoder-only models can be efficiently adapted from one
> architecture/objective prior to the other."

The concrete number (ibid., §5): converting an MLM-trained non-causal decoder
into a causal LM gives a 1.6× speedup over training from scratch — mostly it
is enough to change the attention mask (that single parameter from §2). The
reverse direction does not work: starting from the decoder half of an
encoder-decoder "performed significantly worse than training from scratch"
(ibid., §E.4).

The asymmetry is this:

**The causal decoder is a cheap option over every other configuration. The
encoder-decoder is a terminal commitment.**

In a world where your budget is uncertain and your need is not known in
advance — which is exactly what the early 2020s were — you buy the option. You
do not need to know which architecture is better; you pick the one where you
can change your mind later.

On top of that comes task unification, and the two feed each other. T5's
text-to-text framing unifies the format but still wants an input/target
boundary for each task: where does the prompt end, where does the answer
begin? A causal LM wants no boundary at all — every token is both target and
context. In GPT-3's own words (Brown et al. 2020), the previous world was
"typically task-agnostic in architecture" yet "still requires task-specific
fine-tuning datasets of thousands or tens of thousands of examples."

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-5-option-value.svg" alt="The conversion asymmetry: a causal decoder converts cheaply into a non-causal decoder and from there into an encoder-decoder; the reverse direction does worse than training from scratch.">

<figcaption><b>Figure 5.</b> The asymmetry. The causal decoder is a cheap option over everything else; the encoder-decoder is a commitment with no way back.</figcaption>

</figure>

And this mechanism is not specific to text. When Google built a foundation
model for time-series forecasting from scratch in 2023, it again chose
decoder-only (Das et al. 2023, TimesFM) — the paper's own phrase is
"pretraining a decoder style attention model with input patching, using a
large time-series corpus," and they say explicitly that they took inspiration
from LLMs in NLP. When the task is genuinely general-purpose and zero-shot,
the same choice gets made again, independently, in another field.

---

## 9. So what happened to the others?

They did not die. And this does not trip up the story — on the contrary, it
confirms the answer here.

<aside class="sidenote">

Warner et al., *Smarter, Better, Faster, Longer* (ModernBERT),
[arXiv:2412.13663](https://arxiv.org/abs/2412.13663) · BehnamGhader et al.,
*LLM2Vec*, [arXiv:2404.05961](https://arxiv.org/abs/2404.05961) · Zhang et
al., *Qwen3 Embedding*,
[arXiv:2506.05176](https://arxiv.org/abs/2506.05176) · T5Gemma's citation is
in §7; its sequel is Zhang et al., *T5Gemma 2: Seeing, Reading, and
Understanding Longer*, [arXiv:2512.14856](https://arxiv.org/abs/2512.14856).

</aside>

**The encoder is not finished; it still has a real job.** In classification
and retrieval the production default is still an encoder. ModernBERT (Warner
et al. 2024) calls them "the workhorse of numerous production pipelines," and
rightly so: for a task with a fixed schema, latency-sensitive, running
billions of times a day, the economics invert. A 150M encoder that is 2×
faster beats a 7B decoder that can also write poetry. Spam filtering, search
ranking, content moderation — none of these jobs wants a generative model;
they want a fast, cheap, bidirectional reader. This does not contradict the
thesis: it is **the same economic logic running in reverse** — when the task
is fixed, you do not pay for the option.

But the limit of that defense has to be stated too: the best embedding and
reranking models of 2025 (Zhang et al. 2025, Qwen3-Embedding) are not BERT
descendants but **decoder-only backbones**. The encoders' survival argument is
about cost and latency, not about quality.

**Encoder-decoder came back too.** Google is deliberately reviving it in 2025:
T5Gemma (Zhang et al. 2025) reports that at a similar inference budget the
encoder-decoder achieves "comparable (often better)" results, and that after
instruction tuning Gemma 2B-2B beats Gemma 2B by ~7%.

But look at how they did it: **they take a decoder-only checkpoint and adapt
it into an encoder-decoder.** They do not train from scratch.

The pattern is the same everywhere:
- decoder-only → encoder: LLM2Vec, Qwen3-Embedding
- decoder-only → encoder-decoder: T5Gemma, T5Gemma 2

LLM2Vec's (BehnamGhader et al. 2024) first step is especially nice: *enable
bidirectional attention.* That is exactly the `prefix_len` parameter from §2,
T5's prefix-LM, Artetxe's bidirectional attention. Bidirectionality was never
wrong — it was simply taken out of pretraining and moved into a cheap
after-the-fact adaptation step. Which is Wang et al.'s third finding from
2022, productized.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-6-conversions-2025.svg" alt="The same conversion graph with 2024-2025 systems: LLM2Vec and Qwen3-Embedding turn a decoder-only checkpoint into an encoder, T5Gemma and T5Gemma 2 into an encoder-decoder.">

<figcaption><b>Figure 6.</b> The same graph, with real names. All of 2024-25's "comebacks" are conversions derived from a single decoder-only checkpoint.</figcaption>

</figure>

---

## 10. The answer

The question at the start was: if it lost that experiment, why is decoder-only
the one in use?

Now the pieces fit together:

**Encoder-only cannot be used for general-purpose generation** — but the reason
is not "being bidirectional." MLM does not define a valid joint distribution
over sequences; there is no $p(x)$ to sample from (§3). Meanwhile the encoder
did not die: for reading work it is still the cheapest and fastest tool (§9).

**Encoder-decoder could have been used** — the quality measurements supported it
(§4). But those measurements were made in a game nobody plays in the field:
small scale, post-fine-tuning. The field's game was zero-shot prompting, and
in that game the causal decoder is ahead (§4). On top of that, an
encoder-decoder is a terminal commitment: it does not convert cheaply into
anything else later (§8).

**And the "scales better" story never happened** — that comparison was never
made; the scaling laws were one family's internal map (§5). Bidirectionality
was not refuted either; it was set aside because it did not fit the winning
scenario (§6).

The real reason that remains is this: **a single decoder-only pretraining run
can be converted cheaply into everything needed later — into an encoder, into
an encoder-decoder. The reverse is not true.** The field did not decide that
decoder-only is the right inductive bias; it decided that the right asset to
amortize is the decoder-only *checkpoint*. Architecture stopped being an
identity and became a cheap decision made at deployment time — and all of
2024-25's "comebacks" are already the product of that conversion (§9).

Finally, where this post might be wrong. The largest controlled architecture
comparisons available are at 5B/11B (Wang et al.) and 6.7B (Artetxe et al.)
scale. At frontier scale — hundreds of billions of parameters — an
equal-budget encoder-decoder vs decoder-only comparison does not exist in the
public literature as far as I know. The field made a permanent decision on
evidence gathered two orders of magnitude below the scale it committed to. If
someone runs that experiment and encoder-decoder wins, the argument here
collapses.

But even that confirms the answer: running that experiment is more expensive
than not running it. Nobody did.
