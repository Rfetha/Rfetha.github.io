---
title: 'The Components of a Modern Transformer'
description: 'Bir LLM mimarisinin parçaları: attention türleri, positional encoding, KV-cache ve MoE — 2017 tasarımından bugüne ne eklendi, ne çıkarıldı.'
pubDate: '2026-08-01'
placeholder: true
---

## Abstract

The 2017 Transformer paper shipped one architecture: an encoder-decoder with sinusoidal positional encoding, Post-LN normalization, and full softmax attention. Almost immediately it split into three distinct types — encoder-only (BERT-style, bidirectional, built for classification), decoder-only (the autoregressive, causally-masked "GPT-type" that generates text left to right), and encoder-decoder (retained for translation and other sequence-to-sequence tasks) — and this post traces how each type evolved on its own path, with decoder-only becoming the dominant paradigm for large-scale LLMs. Component by component, it covers what each type does differently and what changed within decoder-only over time: self-attention and multi-head attention as the shared core, causal masking for autoregressive generation, cross-attention where encoder-decoder models still need it, positional encoding's shift from absolute sinusoidal to RoPE, GQA/MQA trading modeling capacity for KV-cache size, KV-cache itself as the bottleneck that shift addresses, and Mixture-of-Experts routing at trillion-parameter scale. Concrete diagrams and numbers come from open-weight models (DeepSeek V3, Qwen3, Gemma 3, Llama, GPT-OSS) whose full architecture is public — the same models Sebastian Raschka's "Understanding Large Language Models" and "The Big LLM Architecture Comparison" use for exactly this kind of type-level, component-level comparison.

---

## Kapsanacak bileşenler

Aşağıdaki liste yazının iskeleti. Her madde, hangi kaynağın hangi noktayı
desteklediğine dair bir notla birlikte duruyor; metin henüz yazılmadı.

### Tarihsel çerçeve: üç mimari tipi

2017 orijinal Transformer tek bir encoder-decoder tasarımıydı; kısa sürede üçe ayrıştı: encoder-only (BERT-tarzı, çift yönlü), decoder-only / autoregressive (günlük dilde "GPT-tipi", tek yönlü, causal-masked üretim), ve encoder-decoder (çeviri gibi sequence-to-sequence görevler için kaldı). Zamanla decoder-only baskın paradigma haline geldi. Somut mimari örnekler — gerçek sayılar, gerçek diyagramlar — open-weight modellerden (DeepSeek, Qwen3, Gemma, Llama, GPT-OSS) gelecek; bunlar üç tipin doğrulanabilir temsilcileri.

### Self-attention

2017 orijinal makalenin çekirdeği ve günümüze değişmeden ulaşan tek parça.

### Multi-head attention (MHA)

2017 orijinalinden beri var. OLMo 2 gibi bazı modeller MHA'yı bilinçli koruyor — ablasyonda GQA'ya göre modelleme performansı daha iyi çıkıyor. Ama artık azınlıkta.

### Causal (masked) attention — decoder-only

Decoder-only tipin ayırt edici mekanizması: her token yalnızca kendinden önceki token'lara bakabiliyor.

### Cross-attention — encoder-decoder

2017 orijinal tasarımın parçası; decoder-only'nin baskın hale gelmesiyle çoğu güncel LLM'de terk edildi.

### Multi-query / grouped-query attention (MQA/GQA)

Sonradan eklendi: KV-cache baskısı büyüdükçe (MQA 2019, GQA 2023) MHA'nın yerini almaya başladı. GQA'nın bellek kazancı, bazı bağlamlardaki modelleme kaybı, ve DeepSeek'in bunun yerine seçtiği MLA (Multi-Head Latent Attention) alternatifi.

### Positional encoding

Orijinal 2017 tasarımı sinusoidal + absolute; bu büyük ölçüde terk edildi. RoPE baskın hale geldi, QK-Norm yaygınlaştı, SmolLM3 kısmen NoPE deniyor, MiniMax-M2 partial RoPE kullanıyor.

### KV-cache

Orijinal makalede yok; decoder-only otoregresif üretimin yaygınlaşmasıyla ortaya çıkan bir darboğaz. MLA, GQA ve sliding-window tasarımlarının hepsi bunu küçültme motivasyonuyla anlatılıyor.

### Mixture-of-Experts (MoE)

Eski bir fikir, ama trilyon-parametre ölçeğinde yaygınlaşması yeni. DeepSeek V3 (256 expert, 9 aktif, 671B toplam / 37B aktif), Qwen3, GPT-OSS gibi modellerin expert sayısı, aktif expert ve shared-expert tercihleri; DeepSeekMoE'nin "çok sayıda küçük expert daha iyi" bulgusu.

### Transformer encoder bloğu

2017 orijinal tasarım, BERT ile popülerleşti. Pre-LN / Post-LN yerleşimi tartışması buraya ait: orijinal makale Post-LN gösteriyor, pratikte Pre-LN kullanılıyor.

### Transformer decoder bloğu

2017'de encoder-decoder'ın yarısıydı; decoder-only tip yaygınlaştıkça tek başına baskın paradigma oldu. Güncel modellerin blok-seviyesi farkları (Pre-LN, Post-LN, Gemma 3'ün dual-norm'u) burada.

---

## Kaynak durumu

Sebastian Raschka'nın üç yazısı okundu ve kapsam bunlara göre kuruldu:
[Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models),
[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison),
[LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/).

Bir de "Visual AI" YouTube oynatma listesi kapsam haritası olarak kullanıldı; sayfanın kendisi çekilemediği için yalnızca video başlıkları esas alındı, içerik uydurulmadı.

Yazı yazılırken her teknik iddia birincil kaynağa (makaleye) bağlanacak; yukarıdaki blog yazıları kapsam belirlemek için kullanıldı, kanıt olarak değil.
