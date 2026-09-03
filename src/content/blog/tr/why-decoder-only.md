---
title: 'Neden herkes decoder-only kullanıyor?'
description: 'Decoder-only nedir ve alan neden ona yakınsadı — beş aday cevabı birincil kaynaklara karşı test eden bir yazı.'
pubDate: '2026-09-02'
---

## Abstract

Bugün kullandığınız her büyük dil modeli aynı mimariyi paylaşıyor: decoder-only. Bunun sebebi olarak genelde "daha iyi olduğu için" deniyor. Ama bu cevabın bir sorunu var — bu soruyu eşit hesap bütçesiyle test eden tek temiz deney (T5, 2019) tam tersini buldu: encoder-decoder yedi görevin yedisini kazandı, decoder-only ise test edilen en kötü yapıydı. Bu yazı iki soruyu sırayla cevaplıyor. Önce *decoder-only tam olarak nedir* — üç mimari ailesinin farkının aslında tek bir attention maskesine indiğini, on iki satır kodla gösterilebilecek kadar somut biçimde. Sonra *madem daha iyi değil, neden onu kullanıyoruz* — sırayla beş aday cevabı (BERT üretemiyor, encoder-decoder daha kötü, decoder-only daha iyi ölçekleniyor, çift yönlülük zararlı, decoder-only daha basit) birincil kaynaklara karşı test ederek. Biri ayakta kalıyor ama söylenen gerekçeyle değil, ikisi çürüyor, biri kısmen doğru çıkıyor, biri hiç kanıtlanmamış. Geriye asıl cevap kalıyor: kalite değil, opsiyon değeri.

---

## 1. Kaybeden deney

Ekim 2019. Google Research'ten bir ekip, transfer öğrenmeyi baştan sona tarayan devasa bir makale yayımlıyor. Adı T5. Makalenin bilinen tarafı şu: her NLP görevini "metin gir, metin çıkar" biçimine sokabilirsiniz.

Ama makalenin §3.2'sinde, bugün neredeyse hiç konuşulmayan bir şey var. Ekip orada, o güne kadar kimsenin düzgün yapmadığı bir deneyi yapıyor: mimarileri **eşit hesap bütçesinde** yan yana koyuyor. Encoder-decoder, decoder-only, prefix-LM. Aynı FLOP, aynı veri, aynı değerlendirme.

Sonuç net. Encoder-decoder yedi görevin yedisini kazanıyor. Decoder-only masadaki **en kötü** yapı — SQuAD'da 19.7 puan geride.

<aside class="sidenote">

Raffel ve ark., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5), [arXiv:1910.10683](https://arxiv.org/abs/1910.10683). Mimari karşılaştırması §3.2'de, sayılar Tablo 2'de, alıntılanan sonuç §3.2.4'te.

</aside>

Ekip bunu saklamıyor, makaleye yazıyor (Raffel ve ark. 2019, §3.2.4): *"Tüm görevlerde, denoising hedefiyle çalışan encoder-decoder mimarisi en iyi performansı gösterdi."* Ve kendi tavsiyelerini veriyorlar: encoder-decoder kullanın.

Dört yıl sonra, üretimdeki neredeyse her büyük dil modeli o deneyi kaybeden mimariydi.

Bugün bir sohbet modeliyle konuştuğunuzda, bir kod asistanı kullandığınızda, bir API'ye istek attığınızda — karşınızdaki şey decoder-only. GPT, Gemini, Claude, Llama, DeepSeek, Qwen. Hepsi aynı aileden.

Bu yazı basit bir soruyu ciddiye alıyor: **madem o deneyi kaybetti, neden onu kullanıyoruz?**

Cevaba gitmeden önce daha temel bir soruyu halletmek gerekiyor: decoder-only tam olarak ne?

---

## 2. Decoder-only ne demek?

<aside class="sidenote">

Üç ailenin künyesi: Vaswani ve ark., *Attention Is All You Need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) · Devlin ve ark., *BERT*, [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) · Radford ve ark., *Improving Language Understanding by Generative Pre-Training* (GPT-1) — arXiv'de değil, [OpenAI teknik raporu](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

</aside>

2017'de tek bir Transformer vardı — *Attention Is All You Need*'in encoder-decoder'ı, makine çevirisi için tasarlanmış. Sonraki iki yıl bunu üç aileye böldü. Üçünü de gerçekten anlamak gerekiyor, çünkü yazının geri kalanı bu üçünün farkı üzerine.

Önce orijinali görelim, çünkü sonraki her şey bunun bir kesiti:

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-0-transformer-2017.svg" alt="2017 Transformer'ının tam mimarisi: encoder yığınında self-attention ve feed-forward alt katmanları; decoder yığınında masked self-attention, encoder çıktısını okuyan cross-attention ve feed-forward; en üstte linear ve softmax.">

<figcaption><b>Figure 0.</b> Orijinal Transformer (Vaswani ve ark., 2017). İki yığın, her alt katmanın ardından bir artık bağlantı ve normalizasyon. Decoder'ın ortasındaki <b>cross-attention</b> alt katmanı, encoder'ın çıktısını anahtar ve değer olarak okur — yazının geri kalanında en çok bu kutuyu konuşacağız.</figcaption>

</figure>

### Encoder-only (BERT, 2018)

Tek bir self-attention yığını. Ayırt edici özelliği şu: **her token, dizideki her token'ı görebiliyor** — kendinden öncekileri de, sonrakileri de.

Bir cümle veriyorsunuz; model, her kelime için o kelimenin *iki yönden de bağlamıyla zenginleşmiş* bir temsil üretiyor. "Bankaya gittim" cümlesindeki "banka"nın temsili, hem önündeki hem arkasındaki kelimelerden besleniyor — para bankası mı, oturma bankı mı, cümlenin tamamına bakarak karar verilebiliyor.

Çıktısı token üretimi değil, **temsil**: dizinin her pozisyonu için bir vektör. Bu vektörlerin üstüne küçük bir başlık ekleyip sınıflandırma yapıyorsunuz, ya da vektörleri doğrudan arama/retrieval için kullanıyorsunuz. BERT bunun için tasarlandı ve bu işlerde hâlâ kullanılıyor.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1a-encoder-only.svg" alt="Encoder-only mimarisi: girdi token'ları fully-visible bir self-attention yığınından geçip her pozisyon için bir temsil vektörü üretiyor, üstüne sınıflandırma başlığı ekleniyor.">

<figcaption><b>Figure 1a.</b> Encoder-only. Tek yığın, fully-visible maske: her token her token'ı görüyor. Çıktısı token değil, temsil.</figcaption>

</figure>

### Decoder-only (GPT-1, 2018)

Yine tek yığın — ama bir kısıtla: **her token yalnızca kendinden öncekileri görebiliyor.** GPT-1'in kendi tarifiyle (Radford ve ark. 2018, §4.1): "12 katmanlı, maskeli self-attention başlıklı, yalnızca decoder'dan oluşan transformer."

Bu kısıt keyfî değil; modelin işini tanımlıyor. Model, her pozisyonda tek bir soruyu cevaplıyor: *buraya kadarını gördüm, sıradaki token ne?* Eğitim de bundan ibaret — devasa bir metin yığınında, her pozisyon için sonraki token'ı tahmin et. Üretim, aynı işlemin dışarıda tekrarı: bir token üret, girdinin sonuna ekle, tekrar sor.

İki şeye dikkat: birincisi, eğitim sinyali **her token'dan** geliyor — dizideki her pozisyon aynı anda hem bağlam hem hedef. İkincisi, eğitimdeki işlem ile üretimdeki işlem **aynı işlem** — model neyle eğitildiyse onu yaparak çalışıyor.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1b-decoder-only.svg" alt="Decoder-only mimarisi: girdi token'ları causal bir self-attention yığınından geçip sonraki token dağılımını üretiyor; örneklenen token girdinin sonuna eklenip döngü tekrarlanıyor.">

<figcaption><b>Figure 1b.</b> Decoder-only. Tek yığın, causal maske. Eğitimdeki işlem ile üretimdeki işlem aynı: sonraki token'ı tahmin et, ekle, tekrar sor.</figcaption>

</figure>

İçini açtığımızda Figure 0'ın decoder sütunuyla karşılaştırmak kolaylaşıyor:

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1e-decoder-block.svg" alt="Decoder-only bloğunun tam içi: token embedding artı positional encoding, ardından N kez masked self-attention, add and norm, feed-forward, add and norm; en üstte linear ve softmax. 2017 decoder'ındaki cross-attention alt katmanı yok.">

<figcaption><b>Figure 1e.</b> Decoder-only bloğu, tam hâliyle. Figure 0'ın decoder'ıyla aynı — bir alt katman eksik: <b>cross-attention</b> çıkarılmış, çünkü bakacak bir encoder yok.</figcaption>

</figure>

### Encoder-decoder (orijinal Transformer, sonra T5)

İki ayrı yığın, iş bölümü yaparak. **Encoder** girdiyi okuyor — fully-visible, her token her token'ı görüyor, tıpkı BERT gibi. **Decoder** çıktıyı üretiyor — causal, tıpkı GPT gibi. İkisini bağlayan üçüncü bir mekanizma var: **cross-attention**. Decoder her üretim adımında, kendi ürettiklerine bakmanın yanında, encoder'ın çıktısına da bakıyor — sorgu decoder'dan geliyor, anahtar ve değer encoder'dan.

Çeviri için doğal tasarım bu: kaynak cümleyi bir kere, iki yönlü, tam bağlamıyla oku; hedef cümleyi soldan sağa yaz; yazarken her kelimede kaynağa dön ve bak. "Oku" ve "yaz" ayrı modüller.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1c-encoder-decoder.svg" alt="Encoder-decoder mimarisi: fully-visible encoder girdiyi okuyor, causal decoder çıktıyı üretiyor, decoder cross-attention ile encoder temsillerine bakıyor.">

<figcaption><b>Figure 1c.</b> Encoder-decoder. İki yığın ve bir köprü: cross-attention'da sorgu decoder'dan, anahtar ve değer encoder'dan geliyor. "Oku" ve "yaz" ayrı modüller.</figcaption>

</figure>

### Ve bir melez: prefix-LM

Tek yığın, karma maske: dizinin başındaki "prefix" kısmında herkes herkesi görüyor (encoder gibi), sonrasında causal (decoder gibi). Şimdilik bir dipnot gibi duruyor; §4'te tablodaki en önemli satır çıkacak.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1d-prefix-lm.svg" alt="Prefix-LM mimarisi: tek yığın, karma maske — prefix token'ları üzerinde fully-visible, devamında causal.">

<figcaption><b>Figure 1d.</b> Prefix-LM. Tek yığın, karma maske. Tek bir parametre — <code>prefix_len</code> — bu melezi belirliyor.</figcaption>

</figure>

### Üçü de aynı tasarımın kesiti

Dördünü ayrı ayrı anlattım ama aslında ayrı tasarımlar değiller. Aynı 2017 bloğundan alınmış kesitler:

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-1f-three-cuts.svg" alt="Üç aile tek bir tasarımın kesitleri olarak: encoder-only encoder bloğunu aynen tutuyor, decoder-only decoder bloğunu cross-attention alt katmanı çıkarılmış hâlde tutuyor, encoder-decoder iki yığını da cross-attention'la birlikte tutuyor.">

<figcaption><b>Figure 1f.</b> Üç kesit. Hiçbirinde yeni bir mekanizma icat edilmiyor — encoder-only encoder bloğunu alıyor, decoder-only decoder bloğunu alıp cross-attention'ı atıyor, encoder-decoder ikisini de bütün tutuyor.</figcaption>

</figure>

### Hepsi aynı formül

Şimdi işin şaşırtıcı kısmı. Bu dört yapı da **aynı** attention formülünü kullanıyor:

$$
\operatorname{Attention}_M(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V
$$

Bu, literatürde **masked attention** denen şeyin ta kendisi — GPT-1'in kendi tarifindeki "maskeli self-attention" da bu operatör. Fark tek bir yerde: $M$, yani kimin kime bakabildiğini söyleyen maske. İzin verilmeyen pozisyonlara, softmax'tan **önce**, $-\infty$ ekliyorsunuz.

Buradaki ayrıntı önemli: $-\infty$ eklemek, ağırlıkları sonradan sıfırlamakla aynı şey değil. Öncesinde eklediğinizde softmax **izin verilen alt küme üzerinde yeniden normalize ediliyor**; sonradan sıfırlasaydınız satır toplamı 1'in altında kalırdı. İki satır kodla görülüyor:

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

İlk satırda model, geçmişteki token'lar arasında paylaştırılmış tam bir dikkat bütçesine sahip. İkincisinde ilk pozisyon dikkatinin %70'ini çöpe atmış oluyor.

Üç maskenin üçü de tek bir fonksiyondan çıkıyor:

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

Tek bir tamsayı. `prefix_len=0` GPT'yi veriyor, `prefix_len=n` BERT'i. "Tek yönlü" ve "çift yönlü" dediğimiz şey, bir parametrenin iki ucu.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-2-masks.svg" alt="Aynı altı token'lık dizi üzerinde üç attention maskesi: fully-visible (tüm hücreler açık), causal (alt üçgen açık), prefix-causal (ilk iki sütun tüm satırlara açık, sonrası causal).">

<figcaption><b>Figure 2.</b> Aynı dizinin üç maskesi. Satır: sorgu pozisyonu, sütun: anahtar pozisyonu. Açık hücre 0, kapalı hücre &minus;&infin;. Üçü arasındaki tek fark <code>prefix_len</code>.</figcaption>

</figure>

Burada dürüst olmam gereken bir sınır var, çünkü bu iddia genelde fazla ileri götürülüyor. "Fark sadece maske" cümlesi *tek bir yığın içindeki tek bir self-attention işlemi* için doğru — encoder-only, decoder-only, prefix-LM için. Encoder-decoder bu kategoriye girmiyor: orada iki ayrı yığın var ve cross-attention'da sorgu decoder'dan, anahtar/değer encoder'dan geliyor. Aynı anahtar kümesi üzerinde farklı maske değil; **farklı anahtar kümesi**.

Tanım bu kadar. Şimdi asıl soruya dönebiliriz: neden bu maskeyi seçiyoruz?

---

## 3. Birinci cevap: "çünkü BERT üretemez"

En yaygın cevap bu. Ve yarısı doğru — ama söylenen gerekçe yanlış.

<aside class="sidenote">

Bu bölümün dayandığı dört kaynak: Wang ve Cho, *BERT has a Mouth, and It Must Speak*, [arXiv:1902.04094](https://arxiv.org/abs/1902.04094) — ve Cho'nun kendi [erratum'u](https://kyunghyuncho.me/bert-has-a-mouth-and-must-speak-but-it-is-not-an-mrf/) · Goyal, Dyer ve Berg-Kirkpatrick, *Exposing the Implicit Energy Networks behind Masked Language Models via Metropolis–Hastings*, [arXiv:2106.02736](https://arxiv.org/abs/2106.02736) · Young ve ark., *Inconsistencies in Masked Language Models*, [arXiv:2301.00068](https://arxiv.org/abs/2301.00068) · Torroba Hennigen ve Kim, *Deriving Language Models from Masked Language Models*, [arXiv:2305.15501](https://arxiv.org/abs/2305.15501).

</aside>

Önce BERT'in nasıl eğitildiğini bilmek gerekiyor. Yöntemin adı **masked language modeling (MLM)**: girdi dizisinin bir kısmı — tipik olarak %15'i — `[MASK]` sembolüyle gizleniyor, model gizlenen token'ları **iki yönden gelen bağlamla** tahmin ediyor. "Bugün [MASK] çok güzeldi" → maskeli kelimeyi hem öncesine hem sonrasına bakarak doldur. Bu, çift yönlü okumanın eğitimdeki karşılığı.

Decoder-only'nin eğitimi ise **autoregressive**: sonraki token'ı yalnızca geçmişten tahmin et. Fark özetle şu: **MLM okumayı öğretiyor, autoregressive eğitim üretmeyi öğretiyor** — çünkü ikincisinde eğitimdeki işlem, üretimdeki işlemin aynısı.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-3b-mlm-vs-ar.svg" alt="Aynı cümle iki eğitim hedefi altında. MLM'de tokenların yaklaşık yüzde on beşi maskeleniyor, bağlam iki yönden geliyor ve kayıp yalnızca maskeli pozisyonlarda. Otoregresif eğitimde her token kendinden öncekilerden tahmin ediliyor ve kayıp her pozisyonda.">

<figcaption><b>Figure 3b.</b> İki hedef, aynı cümle. MLM'de sinyal yalnızca maskeli pozisyonlardan gelir ve girdide gerçek metinde hiç bulunmayan bir <code>[MASK]</code> sembolü vardır. Otoregresif eğitimde her pozisyon bir hedeftir ve eğitim, üretimin birebir provasıdır.</figcaption>

</figure>

Şimdi yaygın gerekçeye gelelim: "BERT çift yönlü olduğu için soldan sağa üretemez." Bu iki kere yanlış. Çift yönlülük engel değil — prefix-LM'ler ve encoder-decoder'lar girdiyi çift yönlü okuyor ve gayet iyi üretiyorlar.

Asıl engel daha derin: **MLM'nin eğitim hedefi, diziler üzerinde geçerli bir olasılık dağılımı tanımlamıyor.**

Causal LM'de bu sorun yok, çünkü orada olan şey bir kimlik:

$$
\log p(x) = \sum_{i=1}^{n}\log p(x_i \mid x_{<i})
$$

Bu bir modelleme varsayımı değil, olasılığın zincir kuralı. Model bu koşulluları öğrendiğinde ortak dağılım *inşaat gereği* ortaya çıkıyor. Ondan örnekleme yapmak tek geçişte ve garantili.

MLM'de öğrenilen şeyse bir koşullular kümesi: $\{p(x_i \mid x_{\setminus i})\}$. Her biri tek başına geçerli bir dağılım. Ama bir koşullular kümesinin bir ortak dağılıma karşılık gelmesi **ek bir koşul** — tutarlılık — ve MLM eğitimi bunu hiçbir yerde zorlamıyor.

Goyal, Dyer ve Berg-Kirkpatrick (2021, §2.1) bunun teorik bir endişe olmadığını iki değişkenli bir karşı örnekle gösteriyor:

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

Sezgisel hali: $X_2$ "ben $X_1$'den bağımsızım" diyor, $X_1$ "ben neredeyse $X_2$'nin kopyasıyım" diyor. Bağımsızlık simetriktir; ikisi aynı anda doğru olamaz.

Peki yine de bu koşullulardan örnekleme yapmayı denersek ne olur? Gibbs örnekleyicisi yazarsınız: sırayla bir değişkeni diğerine bakarak yeniden çekersiniz. Zincir bir yere yakınsar — ama nereye yakınsadığı, hangi sırayla taradığınıza bağlıdır:

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

Aynı model, aynı koşullular, iki farklı cevap. Sistematik tarama "dört durum da eşit olasılıklı" diyor; rastgele tarama "köşegen üç kat daha olası" diyor. İkisi de yanlış değil, çünkü doğrusu diye bir şey yok: ortada bu zincirlerin yaklaşmaya çalıştığı bir hedef dağılım bulunmuyor.

Otoregresif modelde bu soru sorulmaz bile. Zincir kuralı size zaten bir ortak dağılım verir; ondan örneklemek tek geçiştir, tarama sırası diye bir kavram yoktur, yakınsama beklemezsiniz.

<aside class="sidenote">

**Düzeltmenin seyri.** 2019: Wang ve Cho "BERT bir Markov rastgele alanıdır" der ve buradan bir Gibbs örnekleyici türetir; yıllarca standart atıf bu olur. Ardından Cho hatayı kendi bloğunda ilan eder — potansiyellerin ayrışması yanlıştır, çünkü bir token'ı değiştirmek diğer bütün potansiyelleri de değiştirir. 2021: Goyal ve ark. formel çürütmeyi yapar; bu koşullular "herhangi bir tutarlı MRF'ye karşılık gelmek zorunda değil". 2023: Young ve ark. tutarsızlığın BERT-base'den UL2-20B'ye kadar gerçekten var olduğunu ampirik olarak gösterir.

</aside>

Bu hikâyenin literatürdeki seyri ayrıca öğretici, çünkü tam olarak nasıl yürümesi gerekiyorsa öyle yürümüş: iddia ortaya atılıyor, hatayı yazarlardan biri kendisi ilan ediyor, ardından formel çürütme geliyor, en sonunda da ampirik doğrulama.

Adil olmak gerekirse Wang ve Cho'nun ampirik katkısı geri çekilmedi — BERT'ten yinelemeli örnekleme gerçekten akıcı metin üretiyor. Geri çekilen, "dolayısıyla bu meşru bir örnekleyici" iddiası.

Ve perplexity meselesi de buradan çıkıyor: zincir *bir şeye* yakınsıyor, ama neye yakınsadığını kimse söyleyemiyor. $\log p(x)$ tanımlı değilse perplexity de tanımlı değil.

**Puan durumu:** Bu cevap ayakta kalıyor. Encoder-only mimariler üretken dil modellemesi için gerçekten uygun değil — ama gerekçe "çift yönlü olması" değil, "geçerli bir ortak dağılım tanımlamaması".

Yalnız bu, sorunun sadece yarısını çözüyor. Encoder-only elendi. Peki encoder-decoder? O da üretiyor, hem de gayet iyi.

---

## 4. İkinci cevap: "çünkü encoder-decoder daha kötü"

İşte burada §1'deki deneye geri dönüyoruz.

T5 ekibi altı yapıyı iki pretraining hedefiyle çaprazladı ve hepsini aynı FLOP maliyetinde tuttu. Eşit bütçede, denoising hedefiyle:

| | GLUE | SQuAD | SuperGLUE |
|---|---|---|---|
| Encoder-decoder | **83.28** | **80.88** | **71.36** |
| Prefix-LM | 81.82 | 78.94 | 68.11 |
| Decoder-only | **74.70** | **61.14** | **55.02** |

Bu cevap sadece yanlış değil, **tam tersi** doğru. Ölçülebilen her yerde encoder-decoder kazandı.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-3-t5-table2.svg" alt="T5 Tablo 2, eşit hesap bütçesinde: GLUE'da encoder-decoder 83.28, prefix-LM 81.82, decoder-only 74.70; SQuAD'da sırasıyla 80.88, 78.94 ve 61.14.">

<figcaption><b>Figure 3.</b> T5 §3.2, Tablo 2 — eşit FLOP, denoising hedefi. SQuAD'daki 19.7 puanlık fark tablodaki en büyük ayrım; prefix-LM tek bir maske değişikliğiyle bunun neredeyse tamamını kapatıyor.</figcaption>

</figure>

Tabloda bundan daha ilginç bir şey de var. Prefix-LM'e bakın: decoder-only ile aynı sayıda yığına sahip, ama açığın neredeyse tamamını kapatıyor (SQuAD'da 61.14 → 78.94). Aradaki tek fark maske. Yani ceza "tek yığına sahip olmak" değildi — ceza, **girdiyi de causal maskeyle okumaktı**. §2'deki `prefix_len` parametresi, tablodaki 17.8 puanın açıklaması.

### Peki deney doğruysa, gerçek neden tersine döndü?

Burası kritik, çünkü "deney yanılmış demek ki" diye geçiştirmek de, "sektör aptalmış demek ki" diye geçiştirmek de yanlış. Deney doğruydu — ama **yanlış oyunu ölçüyordu.**

İki sınırına bakın. Birincisi: bu deneydeki modeller ~110M parametre sınıfında. Bunun ne kadar küçük olduğunu kademeli görelim: BERT-base bu boyutta. GPT-2 bunun ~14 katı (1.5B). Bugün "küçük" diye anılan 7B'lik açık modeller ~64 katı. GPT-3, ~1600 katı (175B). Yani alan, frontier'da vereceği kararın kanıtını, frontier'ın binde biri ölçekte topladı.

İkincisi ve daha önemlisi: tablodaki her sayı, **göreve özel fine-tuning sonrası** ölçülmüş. Oysa sahada kazanılan oyun bambaşkaydı: modeli hiç fine-tune etmeden, sadece prompt yazarak görev yaptırmak. Bu oyun — zero-shot / few-shot — T5'in deneyinde **hiç ölçülmedi.**

Onu ölçen ilk büyük kontrollü çalışma 2022'de geldi (Wang ve ark. 2022, Bulgu 1) ve şöyleydi: *"Tam dil modelleme hedefiyle eğitilmiş causal decoder-only modeller, ön eğitimden hemen sonra değerlendirildiğinde en iyi zero-shot genellemeyi sağlıyor."* Yani decoder-only, **kendi oyununda gerçekten kazanıyor.**

<aside class="sidenote">

Bu bulguyu fazla yüklememek gerek: makale, sonucun ağırlıklı olarak causal-decoder'lar için tasarlanmış bir prompt setine dayandığını kendisi söylüyor. Üstelik fine-tuning devreye girince kazanan yine encoder-decoder'a dönüyor — §8'de oraya geleceğiz.

</aside>

Özet: iki deney, iki oyun, iki farklı kazanan. Sektör hangi oyunu oynayacağını seçti; o seçimden sonra T5'in tablosu geçersiz değil ama **alakasız** hale geldi.

**Puan durumu:** İkinci cevap çürüdü. Decoder-only'yi ölçülen kalitesi için seçmiyoruz — kalite tablosu kaybettiği oyunda ölçülmüştü, kazandığı oyun başkaydı.

---

## 5. Üçüncü cevap: "çünkü daha iyi ölçekleniyor"

Kalite argümanı düşünce herkesin döndüğü yer burası. Ve bu cevabın arkasında alanın en etkili iki makalesi duruyor sanılır: scaling law'lar.

2020'de Kaplan ve ark. şunu gösterdi: dil modeli kaybı; parametre sayısı, veri miktarı ve hesapla birlikte **pürüzsüz güç yasalarıyla** düşüyor — altı ilâ sekiz mertebe boyunca, sapma işareti göstermeden. Üstelik "performans ölçeğe güçlü, model şekline zayıf bağlı": derinlik, genişlik, kafa sayısı gibi mimari ayrıntılar, toplam ölçeğin yanında neredeyse önemsiz. Büyük modeller örneklem açısından da daha verimli. Bu makale, sonraki beş yılın yatırım haritası oldu: büyüt, veriyi büyüt, gerisini boş ver.

2022'de Chinchilla dengeyi düzeltti (Hoffmann ve ark. 2022): 70 milyondan 16 milyara, 400'den fazla model eğiterek, parametre ve token'ın **eşit oranda** büyütülmesi gerektiğini gösterdi. Sonucun gösterisi etkileyiciydi: aynı hesap bütçesiyle eğitilen 70B'lik Chinchilla; 280B'lik Gopher'ı, 175B'lik GPT-3'ü, 530B'lik Megatron-Turing'i "tutarlı ve belirgin şekilde" geçti.

Şimdi soru: bu iki makalenin neresinde "decoder-only daha iyi ölçekleniyor" yazıyor?

<aside class="sidenote">

Kaplan ve ark., *Scaling Laws for Neural Language Models*, [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) — kapsam beyanı §2'de · Hoffmann ve ark., *Training Compute-Optimal Large Language Models* (Chinchilla), [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) · Tay ve ark., *Scaling Laws vs Model Architectures*, [arXiv:2207.10551](https://arxiv.org/abs/2207.10551).

</aside>

Hiçbir yerinde. Kaplan'ın kendi kapsam beyanı (Kaplan ve ark. 2020, §2): *"Öncelikli olarak yalnızca decoder'dan oluşan Transformer modelleri eğitiyoruz."* Karşılaştırma kolları LSTM ve Universal Transformer — encoder-only yok, encoder-decoder yok. Chinchilla'nın 400 modeli de tek istisnasız decoder-only. Bunlar mimari karşılaştırmaları değil; **decoder-only'nin kendi içinde** ölçek yasaları. "Daha iyi ölçekleniyor" cümlesinin gerektirdiği deney — iki ailenin ölçek eğrilerini yan yana koymak — bu makalelerde hiç yapılmadı.

<aside class="sidenote">

**Karşılaştırma kolları ne?** LSTM, diziyi pozisyon pozisyon gezerek durumunu güncelleyen klasik tekrarlayan ağ. Universal Transformer (Dehghani ve ark., *Universal Transformers*, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)) ise tekrarlamayı zamana değil derinliğe taşır: aynı katmanı paylaşılan ağırlıklarla defalarca uygular, ama tüm pozisyonları paralel işlemeye devam eder — kendi tarifleriyle "parallel-in-time self-attentive recurrent sequence model". Üstüne bir de "dynamic per-position halting mechanism" ekler; her token kaç tur işleneceğine kendisi karar verir. Belirli varsayımlar altında Turing-complete olduğunu da iddia ederler.

</aside>

Karşılaştırma kollarının ne olduğuna bakmakta fayda var, çünkü hangi eksende ölçüm yapıldığını gösteriyor. LSTM tekrarlamayı zamanda yapar, Universal Transformer derinlikte, Transformer hiç yapmaz. Yani Kaplan'ın gerçekten değiştirdiği eksen "diziyi nasıl işliyorsun" idi. Bizim sorumuzun ekseni ise başka — kaç yığın var ve kim kimi görüyor. O eksen o grafiklerde hiç yer almadı.

Burada güzel bir ironi de var. Kaplan'ın manşet bulgusu neydi? "Mimari ayrıntılar ölçeğin yanında önemsiz" (a.g.e., §1.1). Ölçek-maksimalist programı başlatan makale, mimarinin kaldıraç *olmadığını* savunan makale. O makaleye inanıyorsanız, decoder-only'nin zaferi tanımı gereği mimari bir kalite zaferi olamaz.

Peki mimariyi ölçekle sistematik karşılaştıran biri oldu mu? Evet — Tay ve ark. (2022), on mimari, 100'den fazla model. Buldukları, "mimari önemsiz"in tam tersi: *"Bu ölçek katsayısı modelden modele büyük ölçüde farklılık gösteriyor."* Dahası, en iyi model hangi hesap bölgesinde olduğunuza göre değişiyor, ve upstream perplexity downstream transferi iyi öngörmeyebiliyor.

**Puan durumu:** Üçüncü cevap da çürüdü. Söylenebilecek en güçlü doğru cümle şu: *decoder-only'nin daha iyi ölçeklendiği gösterilmedi; decoder-only, ölçeği ölçülen, finanse edilen ve araçlandırılan aileydi.*

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-4-scaling.svg" alt="Şematik log-log ölçek eğrisi: kayıp, hesapla birlikte düz bir çizgi hâlinde düşüyor. Grafikte yalnızca decoder-only ailesi ölçülmüş; karşılaştırma eğrisi hiç çizilmemiş.">

<figcaption><b>Figure 4.</b> Şematik. Ölçek yasalarının ölçtüğü tek aile decoder-only; kesikli çizgi bir ölçüm değil, hiç yapılmamış deneyin yeri.</figcaption>

</figure>

---

## 6. Dördüncü cevap: "çünkü çift yönlülük zararlı"

Son aday. Belki çift yönlü attention, üretim yapan bir modelde aktif olarak zarar veriyordur?

<aside class="sidenote">

Artetxe ve ark., *On the Role of Bidirectionality in Language Model Pre-Training*, [arXiv:2205.11726](https://arxiv.org/abs/2205.11726).

</aside>

Önce iki tarafın ne sağladığını netleştirelim, çünkü bu bölümün sonucu ancak o zaman anlam kazanıyor.

**Çift yönlülük ne sağlar?** Her pozisyonun temsili, dizinin tamamından beslenir. Bir kelimenin anlamı çoğu zaman sonrasında netleşir — çift yönlü model bunu doğrudan kullanır. Bu yüzden anlama ağırlıklı işlerde (sınıflandırma, soru-cevap için fine-tuning, boşluk doldurma/infilling) doğal avantajı vardır: temsil, eldeki tüm bilgiyi görerek kurulur.

**Soldan-sağalık ne sağlar?** İki şey. Birincisi, eğitim ile üretim aynı işlemdir — model, dağıtımda yapacağı işin bire bir aynısıyla eğitilir, arada `[MASK]` gibi eğitime özgü yapay semboller yoktur. İkincisi, her token bir eğitim hedefidir: dizinin tamamı sinyal üretir. Bunun karşılığında her pozisyon, geleceği görmeden karar vermek zorundadır.

Yani soru "hangisi doğru" değil; **"hangisi, hangi iş için"**. Ve bu, test edilmiş bir soru. Artetxe ve ark. (2022) tam olarak bunu yapmış — üstelik *çift yönlü bağlam* ile *çift yönlü attention*'ı birbirinden ayırıp her birini tek tek kontrol ederek, 6.7B parametreye kadar. Buldukları (a.g.e., özet):

> "En iyi konfigürasyon büyük ölçüde uygulamaya bağlı (örneğin çift yönlü attention fine-tuning ve infilling için faydalı, ama sonraki-token tahmini ve zero-shot priming için zararlı)."

Tam da yukarıdaki iş bölümü: çift yönlülük anlama işlerinde kazandırıyor, üretim-artı-prompt işlerinde kaybettiriyor. Ve bu farklar "ölçekte tutarlı kalıyor" — yani "büyütünce kapanır" savunması da bu veriyle desteklenmiyor. Yazarların kendi sonucu: soldan sağa modellere odaklanmak "bazı ödünleşmelerle geliyor ve çok büyük çift yönlü modeller geliştirmek değerli olabilir."

**Puan durumu:** Kısmen ayakta. Çift yönlülük, kazanan senaryoda — serbest üretim artı prompt'lama — gerçekten işe yaramıyor. Ama çürütülmedi; sadece o senaryoya uymadı. Sonra o senaryo geri kalan her şeyi yuttu.

---

## 7. Beşinci cevap: "çünkü daha basit, servis etmesi kolay"

Dört aday elendikten sonra akla gelen beşinci açıklama genelde mühendislikten gelir: bırakın kaliteyi, decoder-only sadece **daha kolay**. Tek yığın, tek kod yolu, tek grafik. Encoder-decoder iki yığın, cross-attention, iki ayrı optimizasyon problemi demek. Üretimde bunu kim ister?

Bu cevabın sezgisel çekiciliği yüksek. Sorun şu ki, ararken onu destekleyen birincil bir kaynak bulamadım — ne bir karşılaştırmalı servis maliyeti ölçümü, ne bir gecikme kıyaslaması. Elimizde yalnızca sıkça tekrarlanan bir mühendislik folkloru var.

<aside class="sidenote">

Zhang ve ark., *Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation* (T5Gemma), [arXiv:2504.06225](https://arxiv.org/abs/2504.06225).

</aside>

Aleyhine olan ise sıkça tekrarlanan bir şey değil, bizzat konuyu çalışan bir makale. T5Gemma (Zhang ve ark. 2025) tam tersini söylüyor: encoder-decoder modeller "gerçek dünya uygulamalarında **çıkarım verimlilikleri** ve daha zengin encoder temsilleri için hâlâ yaygın olarak kullanılıyor" — ve benzer çıkarım bütçesinde decoder-only muadillerinden daha iyi sonuç verdiklerini rapor ediyorlar.

Yani en azından kamuya açık kanıt düzeyinde, "encoder-decoder'ı serve etmek daha zor" iddiası havada duruyor. Doğru olabilir; ama gösterilmedi ve gösterilenler ters yönü işaret ediyor.

**Puan durumu:** Çürüdü demek fazla iddialı olur — daha doğrusu **hiç kanıtlanmadı**, ve kanıtlanmış olanlar aleyhine. Bir mimari tercihinin arkasındaki gerçek sebep olarak kullanılamaz.

---

## 8. Asıl cevap: opsiyon değeri

Dört adaydan üçü çürüdü, biri yarım kaldı. Geriye ne kalıyor?

En iyi cevap, aslında bu soruyu araştıran makalenin kendi bulgusunda duruyor — ve bir kalite bulgusu değil, bir **ekonomi** bulgusu.

<aside class="sidenote">

Wang, Roberts ve ark., *What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?*, [arXiv:2204.05832](https://arxiv.org/abs/2204.05832) — üçüncü bulgu §5'te, ters yön denemesi §E.4'te · Brown ve ark., *Language Models are Few-Shot Learners* (GPT-3), [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) · Das ve ark., *A decoder-only foundation model for time-series forecasting* (TimesFM), [arXiv:2310.10688](https://arxiv.org/abs/2310.10688).

</aside>

Wang ve ark. (2022, BigScience) üç mimariyi üç hedefle çaprazlayıp 5B/11B ölçeğinde eğitmişler. İlk iki bulgularını §4'ten hatırlıyoruz ve birbirinin tersiydi: pretraining'den hemen sonra zero-shot ölçüldüğünde causal decoder kazanıyor; multitask fine-tuning'den sonra ölçüldüğünde encoder-decoder + MLM kazanıyor, hem de "büyük farkla". Kendi ifadeleriyle: "en iyi hedef ve mimari bu iki ortamda birbirinin tersi."

Asıl mesele üçüncü bulguları:

> "Decoder-only modeller bir mimari/hedef ikilisinden diğerine verimli şekilde uyarlanabiliyor."

Somut sayı (a.g.e., §5): MLM ile eğitilmiş bir non-causal decoder'ı causal LM'e çevirmek, sıfırdan eğitmeye kıyasla 1.6× hızlanma veriyor — çoğunlukla attention maskesini değiştirmek yetiyor (§2'deki o tek parametre). Ters yön ise çalışmıyor: encoder-decoder'ın decoder yarısından başlamak "sıfırdan eğitmekten belirgin şekilde kötü" çıkmış (a.g.e., §E.4).

Asimetri şu:

**Causal decoder, diğer bütün konfigürasyonlar üzerinde ucuz bir opsiyondur. Encoder-decoder ise nihai bir taahhüttür.**

Bütçenizin belirsiz, ihtiyacınızın önceden bilinmediği bir dünyada — 2020'lerin başı tam olarak buydu — opsiyonu satın alırsınız. Hangi mimarinin daha iyi olduğunu bilmenize gerek yok; sonradan fikrinizi değiştirebileceğiniz olanı seçiyorsunuz.

Buna bir de görev birleştirmesi ekleniyor, ve ikisi birbirini besliyor. T5'in metinden-metne çerçevesi biçimi birleştiriyor ama hâlâ her görev için bir girdi/hedef sınırı istiyor: prompt nerede biter, cevap nerede başlar? Causal LM hiçbir sınır istemiyor — her token hem hedef hem bağlam. GPT-3'ün kendi ifadesiyle (Brown ve ark. 2020) önceki dünya "mimari olarak görevden bağımsızdı ama hâlâ binlerce örneklik göreve özgü fine-tuning veri setleri gerektiriyordu."

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-5-option-value.svg" alt="Dönüşüm asimetrisi: causal decoder ucuza non-causal decoder'a, oradan encoder-decoder'a dönüşüyor; ters yön sıfırdan eğitmekten daha kötü sonuç veriyor.">

<figcaption><b>Figure 5.</b> Asimetri. Causal decoder diğer her şeyin üzerinde ucuz bir opsiyon; encoder-decoder ise geri dönüşü olmayan bir taahhüt.</figcaption>

</figure>

Ve bu mekanizma metne özgü değil. Google 2023'te zaman serisi tahmini için sıfırdan bir temel model kurarken yine decoder-only seçti (Das ve ark. 2023, TimesFM) — makalenin kendi cümlesi "büyük bir zaman serisi külliyatı üzerinde yamalı-decoder tarzı bir attention modelini önceden eğitmek", ve NLP'deki LLM'lerden ilham aldıklarını açıkça söylüyorlar. Görev gerçekten genel amaçlı ve zero-shot olduğunda, aynı seçim başka bir alanda bağımsız olarak tekrar yapılıyor.

---

## 9. Peki diğerlerine ne oldu?

Ölmediler. Ve bu, hikâyenin ayağına dolanmıyor — tam tersine, cevabımızı doğruluyor.

<aside class="sidenote">

Warner ve ark., *Smarter, Better, Faster, Longer* (ModernBERT), [arXiv:2412.13663](https://arxiv.org/abs/2412.13663) · BehnamGhader ve ark., *LLM2Vec*, [arXiv:2404.05961](https://arxiv.org/abs/2404.05961) · Zhang ve ark., *Qwen3 Embedding*, [arXiv:2506.05176](https://arxiv.org/abs/2506.05176) · T5Gemma'nın künyesi §7'de; devamı Zhang ve ark., *T5Gemma 2: Seeing, Reading, and Understanding Longer*, [arXiv:2512.14856](https://arxiv.org/abs/2512.14856).

</aside>

**Encoder bitmedi; hâlâ gerçek bir işi var.** Sınıflandırma ve retrieval'da üretim varsayılanı hâlâ encoder. ModernBERT (Warner ve ark. 2024) onları "sayısız üretim hattının iş atı" diye tanımlıyor ve haklı: sabit şemalı, gecikmeye duyarlı, günde milyarlarca kez koşan bir görev için ekonomi tersine dönüyor. Şiir de yazabilen 7B'lik bir decoder yerine, 2× hızlı 150M'lik bir encoder kazanıyor. Spam filtresi, arama sıralaması, içerik denetimi — bu işlerin hiçbiri üretken model istemiyor; hızlı, ucuz, iki yönlü bir okuyucu istiyor. Bu, tezle çelişmiyor: **aynı ekonomik mantığın ters yöne işlemesi** — görev sabitse opsiyona para vermezsiniz.

Ama bu savunmanın sınırını da söylemek lazım: 2025'in en iyi embedding ve reranking modelleri (Zhang ve ark. 2025, Qwen3-Embedding) BERT torunları değil, **decoder-only omurgalar**. Encoder'ların hayatta kalma argümanı maliyet ve gecikme üzerine, kalite üzerine değil.

**Encoder-decoder da geri geldi.** Google 2025'te bilerek diriltiyor: T5Gemma (Zhang ve ark. 2025), benzer çıkarım bütçesinde encoder-decoder'ın "karşılaştırılabilir (çoğu zaman daha iyi)" sonuç verdiğini, talimat ayarından sonra Gemma 2B-2B'nin Gemma 2B'yi ~%7 geçtiğini söylüyor.

Ama nasıl yaptıklarına bakın: **decoder-only bir checkpoint'i alıp encoder-decoder'a uyarlıyorlar.** Sıfırdan eğitmiyorlar.

Desen her yerde aynı:
- decoder-only → encoder: LLM2Vec, Qwen3-Embedding
- decoder-only → encoder-decoder: T5Gemma, T5Gemma 2

LLM2Vec'in (BehnamGhader ve ark. 2024) ilk adımı özellikle güzel: *causal maskeyi sil.* Bu tam olarak §2'deki `prefix_len` parametresi, T5'in prefix-LM'i, Artetxe'nin çift yönlü attention'ı. Çift yönlülük hiç yanlış değildi — sadece pretraining'den çıkarılıp ucuz bir sonradan-uyarlama adımına taşındı. Yani Wang ve ark.'nın 2022'deki üçüncü bulgusu, ürünleşmiş hali.

<figure>

<img loading="lazy" decoding="async" src="/figures/why-decoder-only/fig-6-conversions-2025.svg" alt="Aynı dönüşüm grafiği, 2024-2025 sistemleriyle: LLM2Vec ve Qwen3-Embedding decoder-only checkpoint'i encoder'a, T5Gemma ve T5Gemma 2 encoder-decoder'a dönüştürüyor.">

<figcaption><b>Figure 6.</b> Aynı graf, gerçek isimlerle. 2024-25'in bütün "geri dönüşleri" tek bir decoder-only checkpoint'ten türetilmiş dönüşümler.</figcaption>

</figure>

---

## 10. Cevap

Baştaki soru şuydu: madem o deneyi kaybetti, neden decoder-only kullanıyoruz?

Artık parçaları birleştirebiliriz:

**Encoder-only'yi genel amaçlı üretim için kullanamıyoruz** — ama sebep "çift yönlü olması" değil. MLM, diziler üzerinde geçerli bir ortak dağılım tanımlamıyor; ortada örneklenecek bir $p(x)$ yok (§3). Öte yandan encoder ölmedi: okuma işlerinde hâlâ en ucuz ve en hızlı araç (§9).

**Encoder-decoder'ı kullanabilirdik** — kalite ölçümleri onu destekliyordu (§4). Ama o ölçümler, sahada oynanmayan bir oyunda yapılmıştı: küçük ölçek, fine-tuning sonrası. Sahanın oyunu zero-shot prompting'di ve o oyunda causal decoder önde (§4). Üstelik encoder-decoder nihai bir taahhüt: sonradan başka bir şeye ucuza dönüşmüyor (§8).

**"Daha iyi ölçekleniyor" hikâyesi ise hiç yaşanmadı** — o karşılaştırma yapılmadı; scaling law'lar tek ailenin iç haritasıydı (§5). Çift yönlülük de çürütülmedi; kazanan senaryoya uymadığı için kenara kondu (§6).

Geriye kalan gerçek sebep şu: **tek bir decoder-only pretraining koşusu, sonradan ihtiyaç duyulan her şeye — encoder'a da, encoder-decoder'a da — ucuza dönüştürülebiliyor. Tersi doğru değil.** Alan, decoder-only'nin doğru tümevarımsal önyargı olduğuna karar vermedi; amorti edilecek doğru varlığın decoder-only *checkpoint'i* olduğuna karar verdi. Mimari bir kimlik olmaktan çıkıp dağıtım zamanında verilen ucuz bir karara dönüştü — 2024-25'in bütün "geri dönüşleri" de zaten bu dönüşümün ürünü (§9).

Son olarak, bu yazının nerede yanlış olabileceği. Elimizdeki en büyük kontrollü mimari karşılaştırmaları 5B/11B (Wang ve ark.) ve 6.7B (Artetxe ve ark.) ölçeğinde. Frontier ölçekte — yüz milyarlarca parametrede — eşit bütçeli bir encoder-decoder vs decoder-only karşılaştırması, bildiğim kadarıyla kamuya açık literatürde yok. Alan, taahhüt ettiği ölçeğin iki mertebe altında toplanmış kanıtla kalıcı bir karar verdi. Biri o deneyi yapar ve encoder-decoder kazanırsa, buradaki argüman çöker.

Ama bu bile cevabı doğrular: o deneyi yapmak, yapmamaktan pahalıdır. Kimse yapmadı.
