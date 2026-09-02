---
title: 'The Components of a Modern Transformer'
description: 'Bir LLM mimarisinin parçaları: attention türleri, positional encoding, KV-cache ve MoE — 2017 tasarımından bugüne ne eklendi, ne çıkarıldı.'
pubDate: '2026-08-01'
placeholder: true
---

<!--
İLK TAM SÜRÜM. Metin uçtan uca yazıldı; figürler (Figure 1 hariç) henüz
çizilmedi, yerleri yorum olarak işaretli.
Kanıt: drafts/research-transformer-components-evidence.md (E§n)
       drafts/research-decoder-only-math.md (M§n)
Plan:  drafts/outline-transformer-components.md
BİTİRİRKEN: figürler · placeholder: true sil · drafts kopyasını sil.
-->

## Abstract

Bir dil modelinin mimarisi anlatılırken hâlâ 2017'nin Transformer'ı çiziliyor.
Oysa bugün sevk edilen bir modelin konfigürasyon dosyasını açtığınızda o tarifin
neredeyse hiçbir maddesi yerinde değil: sinüzoidal konum kodlaması, ReLU'lu
feed-forward ağı, alt katmanın çıktısına uygulanan normalizasyon, kendi anahtar
ve değerine sahip attention kafaları — hepsi değişti. Yerinde kalan liste kısa:
attention işleminin kendisi, artık bağlantılar, ve aynı bloğu üst üste yığma
fikri. Bu yazı aradaki sekiz yılı parça parça geziyor ve her değişikliği adı
konmuş bir baskıya bağlıyor — decode belleği, parametre başına hesap, eğitim
kararlılığı, bağlam uzunluğu — çünkü bunları tek bir sebebe indirmek bu konuda
yapılan en yaygın hata. Yakından bakınca bazı gerekçeler ününden zayıf çıkıyor:
GQA'nın kaynak makalesi yöntemi yalnızca encoder-decoder modellerde ölçtüğünü
kendi sınırlar bölümünde yazıyor, 2017'nin kendi ablasyonu öğrenilen konum
gömmelerini sinüzoidalle farksız buluyor, SwiGLU'nun makalesi neden işe
yaradığına dair hiçbir açıklama sunmadığını açıkça söylüyor. Sayılar mimarisi
kamuya açık modellerden geliyor ve türetilen her rakam kaynağın kendi yayımladığı
sayıyı yeniden üretiyor — DeepSeek-V3'te tek bir isteğin 128K bağlamdaki
KV-cache'i 8.6 GiB; aynı model 2017'nin attention'ıyla kurulsaydı 488 GiB
olacaktı. Ortaya bir "modern transformer tarifi" çıkmıyor: beş açık ağırlıklı
model aynı dört baskıya beş farklı cevap veriyor, ve hangisinin doğru olduğunu
söyleyecek eşit bütçeli bir karşılaştırma kamuya açık literatürde yok.

---

## 1. 2017'de tek bir blok vardı

Haziran 2017. *Attention Is All You Need* tek bir mimari yayımlıyor ve o
mimarinin her ayrıntısı tek bir tarifle geliyor: altı katmanlı bir encoder, altı
katmanlı bir decoder, ikisinin arasında cross-attention. Blokların içi de sabit —
sekiz başlı attention, baş başına $d_k = d_v = 64$; girdi gömmesine **eklenen**
sinüzoidal konum kodlaması; içi 2048 genişliğinde, ReLU'lu bir feed-forward ağı.
Ve her alt katmanın etrafında bir artık bağlantı, ardından normalizasyon.
Makale bunu tek satırda yazıyor (Vaswani ve ark. 2017, §3.1): her alt katmanın
çıktısı $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$.

<aside class="sidenote">

Vaswani ve ark., *Attention Is All You Need*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). Blok tarifi §3.1'de, attention §3.2'de, feed-forward §3.3'te, konum kodlaması §3.5'te. Bu yazıda 2017 dendiğinde kastedilen hep bu makale.

</aside>

Sekiz yıl sonra o tarifin maddeleri tek tek yerinden edilmiş durumda:

- **Sinüzoidal kodlama gitti.** Yerine RoPE geldi — konumu toplayarak değil
  döndürerek kodluyor. Qwen3, Gemma 3, OLMo 2, gpt-oss ve DeepSeek'in hepsi
  RoPE kullanıyor.
- **ReLU gitti.** Yerine SwiGLU geldi (Qwen3, gpt-oss, OLMo 2).
- **LayerNorm gitti.** Yerine, ortalamayı hiç hesaplamayan RMSNorm geldi —
  aynı dört modelde.
- **Normalizasyonun yeri değişti.** 2017'de alt katmanın çıktısına
  uygulanıyordu; bugün girdisine uygulanıyor. Gemma 3 ikisini birden yapıyor,
  OLMo 2 ise 2017'nin yerine geri dönmüş durumda.
- **Feed-forward ağı çoğu modelde tek bir ağ değil.** DeepSeek-V3'te her
  katmanda 256 uzman var ve token başına sekizi çalışıyor.
- **Attention'ın kafaları kendi anahtar ve değer matrislerine sahip değil.**
  Ya paylaşıyorlar (Qwen3, Gemma 3, gpt-oss), ya da sıkıştırılmış tek bir
  vektöre indirgeniyorlar (DeepSeek-V3).

Peki yerinde ne kaldı? İskelet: attention işleminin kendisi, artık bağlantılar,
ve aynı bloğu üst üste yığma fikri. Sekiz yılda değişmeyen liste bu kadar kısa.

### Bu yazının anlatmadığı şey

2017'nin tek tasarımı kısa sürede üç aileye ayrıldı — encoder-only,
decoder-only, encoder-decoder — ve
[zamanla decoder-only baskın paradigma haline geldi](/blog/why-decoder-only/).
O ayrışmanın *neden* böyle sonuçlandığı ayrı bir yazının konusu; orada beş aday
cevap birincil kaynaklara karşı tek tek test ediliyor ve çoğu ayakta kalmıyor.

Burada o tartışma yok. Bu yazı ayrışmadan sonrasını anlatıyor: kazanan bloğun
içini, parça parça. Hangi parça 2017'den kalma, hangisi sonradan girdi, ve
girenler neye cevap veriyor.

### Dört baskı

Bir envanterin liste olmaktan çıkması için bir düzen gerekiyor, ve kaynaklarda
düzen var. Bloğa giren her parçanın arkasında adı konmuş bir baskı duruyor, ve
bu baskılar dört tane.

**Decode belleği ve bant genişliği.** Model bir token üretirken, o ana kadarki
her token'ın anahtar ve değerini bellekten okumak zorunda. Üretimin asıl
maliyeti bu okuma — hesap değil. MQA, GQA, MLA ve kayan pencere attention'ın
hepsi bu okumayı küçültme girişimi (§3, §4).

**Parametre başına hesap.** Bir modeli büyütmenin maliyeti toplam parametre
sayısıyla değil, token başına *çalışan* parametre sayısıyla ölçülüyor.
Mixture-of-Experts bu ikisini birbirinden ayırıyor (§6).

**Ölçekte eğitim kararlılığı.** Belli bir boyutun üstünde eğitim kaybı görünür
bir sebep olmadan ıraksıyor. Pre-LN, RMSNorm, QK-Norm ve attention sink'lerinin
hepsi bu ıraksamaya karşı alınmış önlemler (§5, §7).

**Bağlam uzunluğu.** 512 token için tasarlanmış bir konum kodlaması 128.000
token'da çalışmıyor. RoPE'un taban frekansının yeniden ölçeklenmesi ve NoPE
tartışması buradan çıkıyor (§5).

Bu dördü ayrı tutulmalı, çünkü karıştırmak hem kolay hem yanlış: MoE'nin
KV-cache ile hiçbir ilgisi yok, QK-Norm'un bağlam uzunluğuyla hiçbir ilgisi yok.
Bir mimariyi tek bir sebeple açıklamak en sık yapılan hata, ve bu yazının
kaçınmaya çalıştığı asıl şey o.

<figure>

<img src="/figures/transformer-architecture-components/fig-1-timeline.svg" alt="2016'dan 2025'e uzanan zaman çizelgesi. Bileşenler yayımlanma tarihlerine göre yerleştirilmiş ve dört baskıya göre gruplanmış: decode belleği, parametre başına hesap, eğitim kararlılığı, bağlam uzunluğu. Sparsely-gated MoE, Transformer'ın beş ay solunda duruyor.">

<figcaption><b>Figure 1.</b> Bloğun parçaları ve giriş tarihleri (arXiv v1 gönderim tarihleri). Şeritler, her parçanın cevap verdiği baskıyı gösteriyor. Soldaki kırmızı işaret: sparsely-gated <b>MoE</b>, Transformer'dan beş ay önce yayımlandı — içinde yaşadığı bloktan eski.</figcaption>

</figure>

Figürdeki o tarih tesadüf değil, bir uyarı: bu yazıdaki parçaların hepsi 2017'nin
çocuğu değil. Bazıları daha eski fikirlerin, doğru donanım ve doğru ölçek
geldiğinde geri çağrılmış hâlleri.

Envantere değişmeyen parçadan başlayalım.

## 2. Attention mekanizması

"Attention" tek bir işlemin adı. Bu bölümde sayacağım beş çeşidi ayrı
mekanizmalar değil — aynı işlemin farklı argümanlarla çağrılmış hâlleri. Önce
işlemin kendisini kuralım, sonra çeşitleri tek tek gezelim; aralarındaki fark
yalnızca iki şeyde toplanıyor: **sorgunun bakabildiği pozisyonlar**, ve
**anahtar ile değerin nereden geldiği**.

Burada olmayan bir şeyi de baştan söyleyeyim. MQA, GQA ve MLA da "attention
çeşidi" diye anılır, ama farklı bir eksende dururlar: kimin kime bakabildiğini
değiştirmezler, anahtar ve değerin **nasıl saklandığını** değiştirirler. O eksen
2017'nin tasarım uzayında yok — sonradan, üretim maliyeti bir problem hâline
gelince açıldı. Bu yüzden onlar §4'te, ve aralarında §3 duruyor: problemi
tanımlayan bölüm.

### 2.1 Self-attention

Sekiz yılda yerinden oynamayan parça bu. Tanım, 2017'deki hâliyle:

<figure>

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

<figcaption><b>Figure 2.</b> Ölçeklenmiş nokta çarpımı attention'ı (Vaswani ve ark. 2017, §3.2.1). Sorgu, anahtar ve değer aynı diziden geldiğinde buna self-attention deniyor.</figcaption>

</figure>

Her satır bir sorgu pozisyonu. O pozisyonun sorgusu bütün anahtarlarla çarpılıyor,
çıkan skorlar softmax'tan geçirilip bir olasılık dağılımına dönüşüyor, ve o
dağılımla değerlerin ağırlıklı ortalaması alınıyor. Tek cümlede: **her pozisyon,
diğer pozisyonlardan ne kadar bilgi alacağına kendisi karar veriyor.**

#### $\sqrt{d_k}$ neden orada

Paydadaki bölen keyfî bir sabit değil, ve makale gerekçesini bir dipnotta
veriyor: $q$ ile $k$'nin bileşenleri bağımsız, ortalaması 0 ve varyansı 1 olan
değişkenlerse, nokta çarpımları $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ ortalaması
0, **varyansı $d_k$** olan bir değişken. Yani $d_k$ büyüdükçe logitler büyüyor.

Bu neden bir sorun? Softmax'ın girdisi büyüdükçe çıktısı tek bir pozisyona
yığılıyor; dağılım düzleşmek yerine sivrileşiyor ve gradyanlar kayboluyor.
Bölen tam olarak bunu engelliyor. İki satır kodla görülüyor:

```python
import numpy as np
rng = np.random.default_rng(0)

for d_k in (8, 64, 512):
    q = rng.normal(0, 1, (20000, d_k))
    k = rng.normal(0, 1, (20000, d_k))
    logit = (q * k).sum(-1)                    # tek bir q·k çarpımı
    print(f"d_k={d_k:4d}   ham varyans={logit.var():8.1f}   "
          f"√d_k'ye bölünce={(logit/np.sqrt(d_k)).var():5.2f}")
```

```
d_k=   8   ham varyans=     8.2   √d_k'ye bölünce= 1.02
d_k=  64   ham varyans=    64.4   √d_k'ye bölünce= 1.01
d_k= 512   ham varyans=   509.4   √d_k'ye bölünce= 0.99
```

Ham varyans $d_k$'yi birebir takip ediyor; bölen onu boyuttan bağımsız olarak
1'e sabitliyor. Ölçek, softmax'ı doyumdan kurtaran şey.

#### Maliyet: diziye göre karesel

Makalenin Tablo 1'i üç yapıyı üç eksende karşılaştırıyor. Konumuza giren iki
satır:

| | Katman başına karmaşıklık | Sıralı işlem | En uzun yol |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Tekrarlayan (RNN) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |

Takas açık: attention diziye göre **karesel** ama tamamen paralel, ve herhangi
iki pozisyon arasındaki yol sabit uzunlukta. RNN doğrusal ama diziyi sırayla
gezmek zorunda. 2017'de kazanan taraf paralellikti — ve bu yazının geri
kalanının konusu, o karesel terimin sekiz yıl boyunca nasıl bir maliyet olarak
geri döndüğü.

Son bir şerh: "attention 2017'den beri değişmedi" cümlesi neredeyse doğru.
2025'te gpt-oss softmax'ın **paydasına** öğrenilen bir terim ekliyor ve bununla
operatörün kendisini değiştiriyor. Neden yaptıklarını §7.4'te göreceğiz.

### 2.2 Multi-head attention

Tek bir attention işlemi tek bir "bakış açısı" üretiyor. 2017 bunu çoğaltıyor:
aynı işlemi $h$ kez, farklı öğrenilmiş projeksiyonlarla, paralel koşturuyor.

Ama kritik ayrıntı şu — kafalar modeli **genişletmiyor, bölüyor**. Makalenin
kendi konfigürasyonu (Vaswani ve ark. 2017, §3.2.2): *"bu çalışmada $h=8$
paralel attention katmanı, yani kafa kullanıyoruz. Her biri için
$d_k = d_v = d_{\text{model}}/h = 64$ alıyoruz."*

$512 = 8 \times 64$. Sekiz kafa, tek geniş bir kafayla aşağı yukarı aynı
parametre ve aynı hesap maliyetinde çalışıyor; kazanılan şey kapasite değil,
**çeşitlilik** — sekiz farklı alt uzayda sekiz farklı ilişki öğrenilebiliyor.

<!-- FİGÜR: 1 geniş kafa (d=512) vs 8 dar kafa (8×64), aynı toplam genişlik. -->

Buradaki takasın bir sınırı var ve onu da yine 2017 ölçmüş. Makalenin ablasyon
tablosunda kafa sayısı ve kafa genişliği ayrı ayrı oynatılıyor, ve sonuç
(a.g.e., §6.2): *"Tablo 3'ün (B) satırlarında, attention anahtar boyutu $d_k$'yi
küçültmenin model kalitesini düşürdüğünü gözlemliyoruz."*

Bu cümleyi aklınızda tutun, çünkü §4'ün tamamı onun üstüne kuruluyor: kafaları
ucuzlatmanın bariz yolu $d_k$'yi kısmak, ve o yol 2017'de kapatılmış. MQA ve
GQA başka bir düğmeye basacak.

**Bugün hâlâ düz MHA kullanan modeller var.** OLMo 2 bunlardan biri: teknik
raporunun hiperparametre tablosunda tek bir "Attention Heads" satırı var (7B için
32, 13B için 40), ayrı bir anahtar/değer kafa sayısı yok, ve makalede GQA ya da
MQA hiç geçmiyor.

<aside class="sidenote">

OLMo ekibi, *2 OLMo 2 Furious*, [arXiv:2501.00656](https://arxiv.org/abs/2501.00656). Konfigürasyon Tablo 4'te. Bu bir konfigürasyon tespiti: makale MHA'yı GQA'ya karşı ölçen bir ablasyon **içermiyor**, mimari değişikliklerinin tamamını eğitim kararlılığıyla gerekçelendiriyor. "OLMo 2 MHA'yı koruyor çünkü ablasyonlar onu destekliyor" cümlesi bu kaynakta yok.

</aside>

### 2.3 Causal attention ve prefix-invariance

İkinci çeşit, sorgunun bakabildiği pozisyonları kısıtlıyor: her token yalnızca
kendinden öncekileri ve kendisini görebiliyor, sonrasını göremiyor. Bu maskenin
neden seçildiği ve üç mimari ailesini nasıl ayırdığı
[diğer yazının konusu](/blog/why-decoder-only/); burada bizi ilgilendiren tek
şey onun bu yazıya ait sonucu.

O sonucun adı **prefix-invariance**, ve yazının geri kalanının yarısı buradan
çıkıyor.

> **Lemma.** Causal maskeli bir yığında, $\ell$. katmanın $j$. pozisyondaki
> anahtar ve değeri yalnızca $x_{\le j}$'nin fonksiyonudur. Dolayısıyla diziye
> $x_{t+1}$ token'ı eklemek, her $j \le t$ ve her $\ell$ için $k_j^{(\ell)}$ ve
> $v_j^{(\ell)}$'yi **değiştirmez.**

Gerekçe tümevarım: ilk katmanda pozisyon $j$'nin girdisi yalnızca $x_j$'nin
gömmesi. Bir üst katmanda $j$. çıktı yalnızca $j' \le j$ pozisyonlarının bir
önceki katman durumlarına bakıyor — çünkü maske öyle söylüyor — ve feed-forward
pozisyon bazında çalışıyor. Yani hiçbir katmanda, hiçbir pozisyon kendinden
sonrasını okumuyor.

Bunun tersi de doğru ve önemli: maske olmayan bir yığında ($M = \mathbf{0}$)
pozisyon $j$'nin durumu **bütün** pozisyonlara bağlı. Diziye bir token eklemek,
ikinci katmandan itibaren her pozisyonun anahtarını ve değerini değiştiriyor.
Saklanacak kararlı bir şey yok.

İddia sayısal olarak kontrol edilebilir. Üç katmanlı bir yığın, rastgele
ağırlıklar, altı token: önce ilk beş token'la koştur, sonra altısını ekleyip
tekrar koştur, ve ilk beş pozisyonun anahtar/değerlerini karşılaştır.

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
    drift = max(max(np.abs(bk[:5] - ak).max(), np.abs(bv[:5] - av).max())
                for (ak, av), (bk, bv) in zip(a, b))
    print(f"{'causal    ' if causal else 'çift yönlü'}  sapma: {drift:.2e}")
```

```
causal      sapma: 0.00e+00
çift yönlü  sapma: 1.38e+00
```

Sapmanın *yaklaşık* değil **tam** sıfır olması argümanın kendisi: causal yol yeni
token'ı hiç okumuyor, dolayısıyla aritmetik bit düzeyinde aynı kalıyor. Kayan
nokta hatası bile yok, çünkü hesaplanan ifade birebir aynı ifade.

<!-- FİGÜR: iki panel — causal'da ilk beş K/V hücresi değişmemiş, çift yönlüde hepsi değişmiş. -->

Bu, bir dizi optimizasyonun kapısını açıyor: eğer önceki anahtar ve değerler
değişmiyorsa, onları yeniden hesaplamak yerine saklayabilirsiniz. §3 tam olarak
bunun hakkında.

### 2.4 Cross-attention

Üçüncü çeşit, ikinci ekseni oynatıyor: sorgu bir yığından, anahtar ve değer
**başka bir yığından** geliyor. 2017'nin tarifi (a.g.e., §3.2.3):

> "'Encoder-decoder attention' katmanlarında sorgular bir önceki decoder
> katmanından, bellek anahtarları ve değerleri ise encoder'ın çıktısından gelir.
> Bu, decoder'daki her pozisyonun girdi dizisindeki bütün pozisyonlara
> bakabilmesini sağlar."

Yani burada değişen şey maske değil, **anahtar kümesinin kendisi**. Çeviri için
doğal tasarım: kaynak cümleyi bir kez oku, hedef cümleyi yazarken her adımda o
okumaya geri dön.

Bu yazı açısından cross-attention'ın ilginç yanı maliyeti. Encoder'ın çıktısı
bütün üretim boyunca sabit, dolayısıyla cross-attention'ın anahtar ve değerleri
**bir kez** hesaplanıp bütün adımlar boyunca saklanabiliyor. Encoder her token
için değil, her girdi için bir kez koşuyor.

Bu da yaygın bir yanlışı düzeltmemizi gerektiriyor. *"KV-cache'i yalnızca
decoder-only modeller kullanabilir"* cümlesi yanlış. Causal maskeli **her**
decoder cache kullanır — encoder-decoder'ın decoder yarısı dahil; T5 de cache'le
decode eder. Doğru ve dar ifade şu: **büyüyen bir dizi üzerinde çalışan çift
yönlü bir encoder** cache kullanamaz, çünkü §2.3'ün lemması orada geçerli değil.
Encoder-decoder mimarileri bu eksende cezalandırılmıyor.

### 2.5 Kayan pencere

Dördüncü çeşit, causal maskeyi bir kez daha daraltıyor: her token yalnızca son
$W$ token'a bakabiliyor. Fikir Longformer'da (Beltagy ve ark. 2020) sistematik
hâle geliyor — makalenin kendi tarifiyle, *"dizi uzunluğuyla doğrusal ölçeklenen
bir attention mekanizması"* ve *"standart self-attention'ın yerine doğrudan
takılabilen"* bir bileşen.

<aside class="sidenote">

Beltagy, Peters ve Cohan, *Longformer: The Long-Document Transformer*, [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) · Jiang ve ark., *Mistral 7B*, [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) · Gemma Ekibi, *Gemma 3 Technical Report*, [arXiv:2503.19786](https://arxiv.org/abs/2503.19786).

</aside>

Sezgiye ters gelen kısım şu: pencere dar olduğu hâlde modelin görebildiği
menzil dar kalmıyor. Mistral 7B bunu açıkça hesaplıyor (§2): bir attention
katmanında bilgi $W$ token ilerleyebiliyorsa, $k$ katman sonra $k \times W$
token ilerlemiş oluyor. Kendi konfigürasyonlarıyla — $W = 4096$, 32 katman —
*"son katmanda yaklaşık 131K token'lık teorik bir attention menzilimiz oluyor."*

<!-- FİGÜR: katman katman genişleyen etki alanı — k×W yayılımı. -->

Yani "kayan pencere bağlamı kesiyor" sezgisi yanlış: pencere tek bir katmandaki
doğrudan bakışı sınırlıyor, katman yığını ise dolaylı yolu açık tutuyor. Bunun
karşılığında saklanması gereken anahtar/değer sayısı sabitleniyor — Mistral bunu
*rolling buffer cache* ile yapıyor: pozisyon $i$'nin anahtarı cache'te
$i \bmod W$ konumuna yazılıyor, eskisinin üstüne.

Bugün bu, tek başına değil **karışım hâlinde** kullanılıyor:

- **Gemma 3**: beş yerel katmana bir global katman (5:1), yerel pencere 1024
  token. Ve §5.3'e bağlanan bir ayrıntı — yerel katmanlarla global katmanlar
  farklı RoPE taban frekansı kullanıyor.
- **gpt-oss**: bantlı pencere ile tam yoğun attention dönüşümlü, bant genişliği
  128 token. Tasarımı GPT-3'ten devralıyorlar.

Neden yapıldığı — yani cache maliyeti — sıradaki bölümün konusu.

## 3. KV-cache

§2.3'ün lemması bir imkân tanımlıyordu: causal bir yığında geçmiş anahtar ve
değerler değişmiyor, dolayısıyla saklanabilirler. Bu bölüm o imkânın faturasını
çıkarıyor.

Saklamazsanız ne olur? Her yeni token için bütün diziyi baştan işlemeniz
gerekir; $n$ token üretmek $O(n^3 d)$ mertebesinde attention işi demek.
Saklarsanız her adımda yalnızca yeni token'ın projeksiyonlarını hesaplayıp
biriken anahtarlarla çarpıyorsunuz. Bu, üretimi mümkün kılan optimizasyon.

Bedeli bellek. Formül şu:

<figure>

$$
\text{KV bayt} = 2 \times b \times s \times L \times h_{	ext{kv}} \times d_{	ext{head}} \times \text{bayt}_{\text{dtype}}
$$

<figcaption><b>Figure 3.</b> Baştaki 2 anahtar ve değer için; <i>b</i> yığın boyutu, <i>s</i> o ana kadarki token sayısı, <i>L</i> katman sayısı, <i>h<sub>kv</sub></i> <b>anahtar/değer</b> kafa sayısı, <i>d<sub>head</sub></i> kafa başına boyut.</figcaption>

</figure>

Bu formülü kendiniz doğrulayabilirsiniz, çünkü vLLM makalesi çarpımı açık açık
yazmış (Kwon ve ark. 2023, §3):

> "13B parametreli OPT modeli için tek bir token'ın KV-cache'i 800 KB yer
> istiyor; 2 (anahtar ve değer vektörleri) × 5120 (gizli durum boyutu) × 40
> (katman sayısı) × 2 (FP16'da bayt) olarak hesaplanıyor. OPT 2048 token'a kadar
> dizi üretebildiğinden, tek bir isteğin KV-cache'ini saklamak için gereken
> bellek 1.6 GB'a kadar çıkabiliyor."

<aside class="sidenote">

Kwon ve ark., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).

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

Formül makalenin yayımladığı sayıyı birebir üretiyor. Tek bir isteğin,
13 milyarlık bir model için 1.6 GB. Yüzlerce eşzamanlı istekle çalışan bir
sunucuda bunun ne anlama geldiği açık.

### Asıl darboğaz hesap değil, bant genişliği

Buraya kadar cache'i bir kapasite problemi gibi anlattım. Asıl mesele daha
ince, ve bu yazının döndüğü mil o. Shazeer 2019 §2.3'te artımlı üretimin
aritmetik ve bellek erişimi oranını çıkarıyor: bellek erişiminin işlem sayısına
oranı $\Theta\!\left(\frac{n}{d}+\frac{1}{b}\right)$, ve

> "$n \approx d$ ya da $b \approx 1$ olduğunda oran 1'e yaklaşıyor, bu da bellek
> bant genişliğinin modern donanımda büyük bir performans darboğazı olmasına yol
> açıyor."

Karşılaştırma için: aynı analiz **eğitim** için yapıldığında oran
$O(1/k + 1/(bn))$ çıkıyor — yani çok küçük. Cümlenin tamamı şu: **artımlı
decode, eğitimden temelden daha kötü bir donanım rejimi.** Eğitimde çip hesap
yapıyor; decode'da çip çoğunlukla bekliyor.

Pope ve ark. bunu 500B+ ölçekte somutluyor (2022, §2): batch 512 ve 2048 token
bağlamla KV-cache toplam **3TB**'a ulaşıyor — *"modelin parametrelerinin üç
katı"* — ve *"çipin hesap çekirdeği esasen boşta beklerken"* bu cache üretilen
her token için yeniden okunuyor.

<aside class="sidenote">

Pope ve ark., *Efficiently Scaling Transformer Inference*, [arXiv:2211.05102](https://arxiv.org/abs/2211.05102). 3TB rakamı onların bildirdiği değer; makale bu hesabın dayandığı konfigürasyonu satır içinde vermiyor, o yüzden yukarıdaki vLLM örneğini kendiniz doğrulayabileceğiniz aritmetik olarak kullanın.

</aside>

Buradan iki yanlış çıkarım yapmamak lazım. Birincisi §2.4'te düzeltildi:
cache decoder-only'ye özgü değil. İkincisi daha yaygın: **"KV-cache decode'u
ucuzlatır" cümlesi eksik.** Cache bir hesap problemini bir bellek problemine
*çeviriyor*. Bütün MQA/GQA/MLA/PagedAttention literatürü, cache pahalı olduğu
için var.

Sıradaki bölüm o literatür.

## 4. MQA, GQA, MLA

§3'ün problemi tek cümleyle şu: cache çok büyük ve her token için baştan
okunuyor. Formüle bakınca kısılabilecek çarpanlar sınırlı — $L$ ve $d_{	ext{head}}$
modelin kendisi, $s$ kullanıcının isteği, $b$ zaten geliri artıran şey. Geriye
**$h_{	ext{kv}}$** kalıyor: anahtar/değer kafalarının sayısı. Bu bölüm o çarpana
yapılmış üç saldırı, kronolojik sırayla.

### 4.1 MQA — bütün kafalar tek bir anahtar kümesini paylaşsın

Shazeer'in 2019'daki önerisi olabilecek en basit hamle (§3): *"Multi-query
attention, multi-head ile birebir aynı; tek fark, farklı kafaların tek bir
anahtar ve değer kümesini paylaşması."*

Formülde $h_{	ext{kv}} = 1$ demek: cache $h$ kat küçülüyor. Yeni bellek/aritmetik
oranı $\Theta\!\left(\frac{1}{d}+\frac{n}{dh}+\frac{1}{b}\right)$ — kendi
ifadesiyle, *"can sıkıcı $\frac{n}{d}$ terimini $h$ kat küçülttük."*

Ölçülen kazanç çeviri deneyinde şöyle (Tablo 2): temel modelde decoder adımı
token başına **46 µs**, multi-query'de **3.8 µs**. Encoder neredeyse hiç
değişmiyor (1.7 → 1.5 µs) — kazanç tamamen decode tarafında, ki zaten problem
oradaydı.

Peki kalite? Bedeli var, ve makale saklamıyor (Tablo 1):

| Attention | $h$ | $d_k, d_v$ | ln(PPL) | BLEU (dev) | BLEU test (beam 1 / 4) |
|---|---|---|---|---|---|
| multi-head | 8 | 128 | 1.424 | 26.7 | 27.7 / 28.4 |
| **multi-query** | 8 | 128 | 1.439 | 26.5 | 27.5 / **28.5** |
| multi-head | 2 | 64 | 1.480 | 26.2 | 26.8 / 27.9 |
| multi-head | 8 | 16 | 1.513 | 25.8 | — |

Tablonun asıl söylediği, MQA'nın ne kadar kaybettirdiği değil — **alternatiflerin
ne kadar kaybettirdiği.** Shazeer'in kendi okuması (§4.2): multi-query modeli
*"temel modelden biraz daha kötü görünüyor, ama $h$, $d_k$ ve $d_v$'yi düşüren
alternatiflerin hepsinden çok daha yakın."*

Burada §2.2'nin kapanışı devreye giriyor. Cache'i küçültmenin bariz yolları kafa
sayısını ya da kafa genişliğini kısmak, ve 2017 zaten $d_k$'yi kısmanın kaliteyi
düşürdüğünü ölçmüştü. MQA üçüncü bir düğme buluyor: kafa sayısını koru, yalnızca
**anahtar/değer** kafalarını tekle. Kazanma sebebi bu.

### 4.2 GQA — ikisinin arası

MQA agresif bir kesme, ve kalite kaybı bazı bağlamlarda kabul edilemez oluyor.
GQA aradaki bütün noktaları açıyor (Ainslie ve ark. 2023, §2.2):

> "Grouped-query attention sorgu kafalarını $G$ gruba ayırır, her grup tek bir
> anahtar kafasını ve değer kafasını paylaşır. GQA-1, tek grupla ve dolayısıyla
> tek anahtar/değer kafasıyla MQA'ya denktir; GQA-h ise grup sayısı kafa
> sayısına eşit olduğunda MHA'ya denktir."

Tek bir tam sayı, iki ucunda tanıdık iki yapı. Ve seçilen ara değerin gerekçesi
ölçek: *"büyük modeller genelde kafa sayısını da büyütüyor, dolayısıyla
multi-query hem bant genişliğinde hem kapasitede daha agresif bir kesme hâline
geliyor. GQA, model büyüdükçe bant genişliği ve kapasitedeki aynı oransal
azalmayı korumamızı sağlıyor."*

Makalenin ikinci katkısı pratik: mevcut bir MHA checkpoint'ini GQA'ya çevirmek
için sıfırdan eğitmeye gerek yok. Anahtar ve değer projeksiyon matrislerini grup
başına **ortalayıp** orijinal eğitim adımlarının %5'i kadar devam ediyorsunuz.

Bugün her yerde gördüğünüz "8 grup" sayısı da buradan geliyor — ve dayanağı
sanıldığından zayıf (§3.3): *"grup sayısını MQA'dan itibaren artırmak başta
yalnızca mütevazı yavaşlamalara yol açıyor, MHA'ya yaklaştıkça maliyet
artıyor. Uygun bir orta yol olarak 8 grup seçtik."* Tek bir labın kendi eğrisi
üzerinde seçtiği bir orta nokta; türetilmiş bir optimum değil.

**Ve makalenin kendi sınırlar bölümü, bu yazının en dürüst alıntısı:**

> "Sınırlı hesap nedeniyle, XXL GQA modelimizi sıfırdan eğitilmiş karşılaştırmalı
> bir modelle de karşılaştırmıyoruz, dolayısıyla uptraining'in sıfırdan eğitime
> göre göreli performansını bilmiyoruz. Son olarak, uptraining ve GQA'nın
> etkisini yalnızca **encoder-decoder** modellerde değerlendiriyoruz. Son
> zamanlarda decoder-only modeller son derece popüler ve bu modellerde ayrı
> self-attention ve cross-attention olmadığı için, GQA'nın MQA'ya karşı daha
> güçlü bir avantaja sahip olmasını **bekliyoruz**."

Yani Llama 2'den bugüne bütün decoder-only GQA kullanımı, decoder-only üzerinde
hiç ölçülmemiş bir sonuca dayanıyor. Makale bunu açıkça söylüyor; alan sonucu
şerhi olmadan tekrarlıyor.

<aside class="sidenote">

Ainslie ve ark., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) · Touvron ve ark., *Llama 2*, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — §2.1: *"Daha büyük modeller — 34B ve 70B — çıkarım ölçeklenebilirliği için Grouped-Query Attention (GQA) kullanıyor."*

</aside>

### 4.3 MLA — anahtarı ve değeri hiç saklama

DeepSeek üçüncü bir yol seçiyor. MQA ve GQA anahtar/değer **kafalarını**
azaltıyor; MLA anahtar ve değeri hiç saklamıyor, onların yerine düşük ranklı tek
bir **latent vektör** saklıyor (DeepSeek-V2, §2.1.2): *"Çıkarım sırasında MLA
yalnızca $\mathbf{c}_t^{KV}$'yi cache'lemek zorunda, dolayısıyla KV-cache'i
yalnızca $d_c l$ eleman."*

Kendi karşılaştırma tabloları, token başına eleman cinsinden:

| | Token başına KV-cache | Kapasite (kendi etiketleri) |
|---|---|---|
| MHA | $2 n_h d_h l$ | Güçlü |
| GQA | $2 n_g d_h l$ | Orta |
| MQA | $2 d_h l$ | Zayıf |
| MLA | $(d_c + d_h^R)\, l \approx \frac{9}{2} d_h l$ | Daha güçlü |

Kendi ayarlarıyla ($d_c = 4 d_h$, $d_h^R = d_h/2$) bunun anlamı şu:
*"KV-cache'i yalnızca 2.25 gruplu bir GQA'ya eşit, ama performansı MHA'dan daha
güçlü."* DeepSeek-V2'nin özeti rakamları veriyor: DeepSeek 67B'ye kıyasla
*"eğitim maliyetinin %42.5'ini tasarruf ediyor, KV-cache'i %93.3 azaltıyor ve
maksimum üretim verimini 5.76 katına çıkarıyor."*

DeepSeek-V3'ün yayımlanmış konfigürasyonuyla bunu somutlaştırabiliriz: 61 katman,
128 attention kafası, kafa başına 128 boyut, $d_c = 512$, $d_h^R = 64$.

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

Tek bir isteğin 128K bağlamdaki cache'i 488 GiB yerine 8.6 GiB. Bu, mimarinin
tek bir bileşeninden gelen fark.

<!-- FİGÜR: MHA / GQA / MQA / MLA'nın KV ayak izi, token başına eleman formülleriyle. -->

### RoPE ile çarpışma

MLA'nın hikâyesinde, bu yazıdaki en güzel ayrıntı duruyor: yöntem, bağımsız bir
gerekçeyle seçilmiş başka bir bileşenle **matematiksel olarak uyuşmuyor.**

DeepSeek-V2, §2.1.3:

> "Ancak RoPE, düşük ranklı KV sıkıştırmasıyla uyumsuz. Somut olarak, RoPE hem
> anahtarlar hem sorgular için konuma duyarlı. Anahtarlara RoPE uygularsak,
> Denklem 10'daki $W^{UK}$ konuma duyarlı bir RoPE matrisiyle çiftlenmiş olur.
> Bu durumda $W^{UK}$ çıkarım sırasında artık $W^Q$ içine soğurulamaz, çünkü o
> anda üretilen token'a ait bir RoPE matrisi $W^Q$ ile $W^{UK}$ arasında kalır
> ve matris çarpımı değişme özelliğine uymaz. Sonuç olarak, çıkarım sırasında
> bütün önek token'larının anahtarlarını yeniden hesaplamak zorunda kalırız, ki
> bu çıkarım verimliliğini ciddi biçimde engeller."

MLA'nın hızının kaynağı, iki matrisi önceden çarpıp tek matris hâlinde
saklayabilmek. RoPE araya konuma bağlı bir döndürme sokuyor ve matris çarpımı
değişmeli olmadığı için o birleştirme bozuluyor.

Çözümleri **decoupled RoPE**: konum bilgisini taşıyan ayrı, küçük bir boyut
grubu ayırıp yalnızca onu döndürüyorlar. DeepSeek-V3'ün konfigürasyonundaki
$d_h^R = 64$ tam olarak bu — sıkıştırmanın dışında bırakılmış, RoPE'a ayrılmış
bir oyuk.

Bir mimarinin nasıl gerçekten tasarlandığını görmek isterseniz burası iyi bir
yer: iki bileşen ayrı ayrı gerekçelendirilmiş, çarpışmışlar, ve uzlaşma
konfigürasyon dosyasında görünür bir sayı olarak kalmış.

## 5. Positional encoding

Attention işleminde konum diye bir kavram yok. $\operatorname{softmax}(QK^\top)V$
ifadesinde satırları karıştırırsanız çıktı da aynı şekilde karışıyor; işlemin
kendisi diziyi bir küme gibi görüyor. Konum bilgisi dışarıdan enjekte edilmek
zorunda, ve bu bölüm o enjeksiyonun sekiz yılda nasıl değiştiğini anlatıyor.

### 5.1 Sinüzoidal kodlama, ve 2017'nin kendi ablasyonu

Orijinal çözüm, her pozisyon için farklı frekanslarda sinüs ve kosinüslerden
oluşan bir vektör üretip **gömmeye eklemek** (Vaswani ve ark. 2017, §3.5):

<figure>

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

<figcaption><b>Figure 4.</b> Sinüzoidal konum kodlaması. Dalga boyları 2&pi;'den 10000 &middot; 2&pi;'ye uzanan geometrik bir dizi oluşturuyor.</figcaption>

</figure>

Neden bu fonksiyon? Makalenin gerekçesi bağıl konum: *"modelin bağıl konumlara
göre dikkat etmeyi kolayca öğrenebileceğini varsaydığımız için bu fonksiyonu
seçtik, çünkü herhangi bir sabit $k$ kayması için $PE_{pos+k}$, $PE_{pos}$'un
doğrusal bir fonksiyonu olarak yazılabiliyor."*

Bu cümlenin altını çizin: istenen şey **bağıl** konum, elde edilen şey ise
modelin bunu **öğrenebileceği** bir mutlak sinyal. Umut, garanti değil.

Ve hemen ardından gelen cümle, bu bölümün en şaşırtıcı bilgisi:

> "Öğrenilen konum gömmelerini kullanmayı da denedik ve iki sürümün neredeyse
> **birebir aynı** sonuçları ürettiğini gördük (Tablo 3, satır (E)). Sinüzoidal
> sürümü seçtik, çünkü modelin eğitimde karşılaştığından daha uzun dizilere
> genelleme yapmasına imkân verebilir."

Yani seçim kalite gerekçesiyle yapılmadı — ablasyon fark bulamadı. Seçim
**ekstrapolasyon** gerekçesiyle yapıldı: 2017'de ölçülemeyen, tamamen spekülatif
bir eksen. Sekiz yıl sonra tek önemli eksen o.

### 5.2 RoPE — bir gereksinimden türetiliyor

RoPE'un anlatılmaya değer yanı, bir yöntem önerip test etmemesi. Önce bir
**gereksinim** yazıyor, sonra onu sağlayan tek çözümü türetiyor (Su ve ark.
2021, §3.1). Gereksinim şu: sorgu-anahtar iç çarpımı konuma yalnızca **fark**
üzerinden bağlı olsun.

$$
\langle f_q(\boldsymbol{x}_m, m),\, f_k(\boldsymbol{x}_n, n)\rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n)
$$

İki boyutta çözüm bir faz çarpanı, yani gerçel biçimde düpedüz bir **döndürme**
(§3.2.1):

$$
f_{\{q,k\}}(\boldsymbol{x}_m, m) =
\begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix}
\boldsymbol{W}_{\{q,k\}}\,\boldsymbol{x}_m
$$

Kendi ifadeleriyle: *"afin dönüşümden geçmiş kelime gömme vektörünü, konum
indeksinin katları kadar bir açıyla döndürmek yeterli."* Konum bir açıya
dönüşüyor. Fikrin tamamı bu.

$d$ boyuta genelleme, uzayı $d/2$ düzleme bölüp her düzlemi kendi frekansıyla
döndürmek: $\boldsymbol{R}^d_{\Theta,m}$ blok-köşegen bir rotasyon matrisi ve
frekanslar $\Theta = \{\theta_i = 10000^{-2(i-1)/d}\}$.

**Şu sabite dikkat edin: $10000$, Vaswani'nin sabiti.** RoPE sinüsleri
değiştirmedi — onların hesaba *giriş biçimini* değiştirdi. Toplama yerine
çarpma, gömmeye değil sorgu ve anahtara.

Kazanılan şey, §5.1'deki umudun özdeşliğe dönüşmesi (§3.2.2):

$$
\boldsymbol{q}_m^{\top}\boldsymbol{k}_n =
(\boldsymbol{R}^d_{\Theta,m}\boldsymbol{W}_q\boldsymbol{x}_m)^{\top}
(\boldsymbol{R}^d_{\Theta,n}\boldsymbol{W}_k\boldsymbol{x}_n) =
\boldsymbol{x}_m^{\top}\boldsymbol{W}_q\,\boldsymbol{R}^d_{\Theta,\,n-m}\,\boldsymbol{W}_k\boldsymbol{x}_n
$$

Mutlak konumlar giriyor, dışarı yalnızca $n-m$ çıkıyor. Rakamla:

```python
def R(m, d, theta=10000.0):
    """RoPE'un blok-köşegen rotasyon matrisi (Su ve ark. 2021, Denk. 15)."""
    M = np.zeros((d, d))
    for i in range(d // 2):
        a = m * theta ** (-2 * i / d)
        c, s = np.cos(a), np.sin(a)
        M[2*i:2*i+2, 2*i:2*i+2] = [[c, -s], [s, c]]
    return M

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

Son satır ikinci özelliği gösteriyor: döndürme normu değiştirmiyor. Makalenin
vurguladığı gibi, $\boldsymbol{R}^d_\Theta$ ortogonal — *"bu da konum bilgisini
kodlarken kararlılığı garantiliyor."* Toplamalı bir kodlama vektörün boyunu
değiştirir; RoPE değiştirmez.

<!-- FİGÜR: toplamak vs döndürmek — additive'de vektör ucu küreden kayıyor, RoPE'da küre üzerinde. -->

<aside class="sidenote">

Su ve ark., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).

</aside>

### 5.3 Taban frekansı bir ayar düğmesine dönüştü

RoPE'un tasarımında $10000$ sabit bir sayıydı. Bugün hiperparametre — ve
oynatılma sebebi tek bir şey: uzun bağlam.

- **OLMo 2**: $\theta$'yı $10^4$'ten $5 \times 10^5$'e çıkarıyor; gerekçe
  *"konum kodlamasının çözünürlüğünü artırıyor."*
- **Qwen3**: *"RoPE'un taban frekansını ABF tekniğiyle 10.000'den 1.000.000'a
  çıkarıyoruz"*, üstüne çıkarımda YaRN ve Dual Chunk Attention.
- **Gemma 3**: *"global self-attention katmanlarında RoPE taban frekansını
  10k'dan 1M'ye çıkarıyoruz, yerel katmanların frekansını 10k'da bırakıyoruz."*

Son madde §2.5'e bağlanıyor ve tek başına ilginç: **tek model içinde iki farklı
$\theta$.** Yerel katmanlar yalnızca 1024 token görüyor, o yüzden gerilmiş
frekansa ihtiyaçları yok. Konum kodlaması modelin global bir özelliği olmaktan
çıkıp katman tipine göre seçilen bir ayara dönüşmüş durumda.

### 5.4 QK-Norm — ve ölçülmüş bir bedeli

Bu bölümdeki tek bileşen bağlam uzunluğu baskısına değil, **kararlılık**
baskısına ait. Fikir basit: sorgu ve anahtarı nokta çarpımından önce normalize
et.

Kökeni düşük kaynaklı çeviri (Henry ve ark. 2020), ama alanın kullandığı atıf
ViT-22B. Orada neyi çözdüğü çok net (Dehghani ve ark. 2023, §2):

> "ViT'i önceki çalışmaların ötesinde ölçeklerken, birkaç bin adım sonra eğitim
> kaybının ıraksadığını gözlemledik. Bu kararsızlık özellikle 8 milyar parametre
> civarındaki modellerde ortaya çıkıyordu. Sebebi, attention logitlerindeki aşırı
> büyük değerlerdi; bunlar entropisi neredeyse sıfır olan (neredeyse tek-sıcak)
> attention ağırlıklarına yol açıyordu."

Ek B'de rakam da veriyorlar: normalizasyon olmadan attention logitleri
*"50000'i aşan büyüklüklere"* çıkıyor. Softmax'a 50.000 girdiğinde çıkan şey bir
dağılım değil, bir seçim — ve gradyan yok oluyor.

Qwen3, OLMo 2 ve Gemma 3'ün üçü de QK-Norm'u benimsiyor, üçü de gerekçe olarak
kararlılığı gösteriyor. Gemma 3'ün ifadesiyle, Gemma 2'nin soft-capping'inin
yerini alıyor.

**Ama bedava değil**, ve bunu gösteren kontrollü bir çalışma var. Yang ve ark.
(2025) 8 milyar parametrede RoPE, NoPE ve QK-Norm varyantlarını aynı reçeteyle
eğitip karşılaştırıyor. Bulguları (§2.2):

> "RoPE ve QK-Norm varyantları standart kıyaslamalarda karşılaştırılabilir
> performans gösteriyor... Uzun bağlam değerlendirmelerinde ise QK-Norm, diğer
> yeteneklerdeki iyi performansına rağmen üç varyant arasında **en kötüsü.**"

Sebebini de açıklıyorlar: normalizasyon, sorgu ile anahtarın nokta çarpımındaki
büyüklük bilgisini törpülüyor; attention logitleri birbirine yaklaşıyor ve
dağılım düzleşiyor. Ölçtükleri entropi QK-Norm varyantında belirgin biçimde
yüksek — yani dikkat dağınık. Uzun bir bağlamda doğru cümleyi bulması gereken
bir model için bu doğrudan bir kayıp.

<aside class="sidenote">

Yang ve ark., *Rope to Nope and Back Again: A New Hybrid Attention Strategy*, [arXiv:2501.18795](https://arxiv.org/abs/2501.18795) · Dehghani ve ark., *Scaling Vision Transformers to 22 Billion Parameters*, [arXiv:2302.05442](https://arxiv.org/abs/2302.05442) · Henry ve ark., *Query-Key Normalization for Transformers*, [arXiv:2010.04245](https://arxiv.org/abs/2010.04245).

</aside>

Bu, yazının genel desenine iyi bir örnek: bileşenler bedava gelmiyor, ve
gerekçeleri farklı eksenlerde. QK-Norm eğitimi kurtarıyor, uzun bağlamdan bir
şey götürüyor.

### 5.5 NoPE — hiç kodlamamak

Son ihtimal, konum kodlamasını tamamen çıkarmak. Sezgiye aykırı görünüyor ama
bir gerekçesi var: causal maske zaten konum bilgisi taşıyor. İlk pozisyon bir
token görüyor, ikincisi iki, üçüncüsü üç — maskenin kendisi bir sayaç.

Kazemnejad ve ark. (2023) bunu sistematik olarak ölçüyor: sıfırdan eğitilmiş
decoder-only modellerde APE, T5-bağıl, ALiBi, RoPE ve hiçbir kodlama olmayan
NoPE'yi uzunluk genellemesinde karşılaştırıyorlar. Sonuç:

> "Bulgularımız, ALiBi, Rotary ve APE gibi en yaygın kullanılan konum kodlama
> yöntemlerinin aşağı akış görevlerinde uzunluk genellemesi için pek uygun
> olmadığını ortaya koyuyor. Daha da önemlisi, **NoPE, ek bir hesap
> gerektirmeden diğer açık konum kodlama yöntemlerinden daha iyi performans
> gösteriyor.**"

Kapsamı doğru anlamak gerekiyor: bunlar göreve özgü, sıfırdan eğitilmiş modeller.
Bulgu, üretim modellerinin RoPE'u bırakması gerektiği anlamına gelmiyor. Yang ve
ark.'nın 8B'lik karşılaştırmasında da NoPE varyantı standart kıyaslamalarda
RoPE'un gerisinde kalıyor.

Yine de aynı çalışma, iki yaklaşımı **katman katman karıştırmanın** işe
yaradığını gösteriyor. NoPE katmanları ile RoPE katmanlarının davranışları
ölçüldüğünde net biçimde ayrışıyor (§3): NoPE katmanları bilgi getirmede güçlü —
aradıkları token'a yüksek dikkat veriyorlar; RoPE katmanları ise güçlü bir
yakınlık eğilimi gösteriyor, yani son token'lara yaslanıyorlar. Biri arıyor,
öteki yakındakine bakıyor.

Aynı yerde, §5.3'ün ayar düğmesinin bedeli de ölçülmüş: $\theta$ büyüdükçe RoPE
katmanlarının yakınlık eğilimi azalıyor, ama bu kez alıcı alan genişlediği için
gürültü artıyor ve **NoPE katmanlarının arama yeteneği bozuluyor.** $\theta$
10.000'den 4 milyona çıktığında aranan token'a düşen dikkat 0.0765'ten 0.0369'a,
değerlendirme skoru 8.036'dan 6.203'e iniyor. Taban frekansını büyütmek bedava
değil.

## 6. Mixture-of-Experts

Buradan itibaren baskı değişiyor. §3–§5 decode belleği ve bağlam uzunluğu
hakkındaydı; MoE tamamen başka bir şeye cevap veriyor ve ikisini karıştırmamak
gerekiyor. MoE'nin KV-cache ile ilgisi yok.

Cevap verdiği soru şu: **bir modeli, her token için yapılan hesabı aynı oranda
büyütmeden nasıl büyütürsünüz?**

Fikir eski — bu yazıdaki bileşenlerin çoğundan eski. Sparsely-gated MoE Ocak
2017'de yayımlanmış, yani Transformer'dan beş ay önce, ve o zamanki uygulaması
LSTM katmanlarının arasıydı. Kendi özetleriyle (Shazeer ve ark. 2017):

> "Bir sinir ağının bilgi soğurma kapasitesi parametre sayısıyla sınırlıdır.
> Ağın parçalarının örnek bazında etkin olduğu koşullu hesaplama, teoride model
> kapasitesini hesapta orantılı bir artış olmadan dramatik biçimde büyütmenin
> bir yolu olarak önerilmişti... modern GPU kümelerinde, hesap verimliliğinde
> yalnızca küçük kayıplarla model kapasitesinde **1000 kattan fazla** iyileşme
> elde ediyoruz."

Mekanizma basit: feed-forward katmanını $N$ kopyaya ("uzman") çoğaltıyorsunuz,
küçük bir yönlendirici ağ her token için hangi uzmanların çalışacağına karar
veriyor, ve yalnızca seçilenler hesaplanıyor. Parametreler $N$ katına çıkıyor,
token başına hesap sabit kalıyor.

### Kaç uzman, kaçı açık?

Switch Transformer (Fedus ve ark. 2021) en radikal sadeleştirmeyi yapıyor
(§2.1): *"Bu fikirlerin aksine, biz yalnızca tek bir uzmana yönlendiren
sadeleştirilmiş bir strateji kullanıyoruz. Bu sadeleştirmenin model kalitesini
koruduğunu, yönlendirme hesabını azalttığını ve daha iyi performans gösterdiğini
ortaya koyuyoruz."* Aynı hesap bütçesinde T5'e karşı 7 kattan fazla ön eğitim
hızlanması bildiriyorlar.

DeepSeekMoE (Dai ve ark. 2024) ters yöne gidiyor — daha çok, daha küçük uzman —
ve iki fikir öneriyor (§1): *"(1) uzmanları $mN$ tanesine ince taneli biçimde
bölümlemek ve bunlardan $mK$ tanesini aktive etmek, böylece aktive edilen
uzmanların daha esnek biçimde birleşmesine imkân vermek; (2) $K_s$ uzmanı
paylaşılan uzmanlar olarak yalıtmak, böylece ortak bilgiyi yakalamak ve
yönlendirilen uzmanlardaki fazlalığı azaltmak."*

İnce taneliliğin argümanı bir iddia değil, bir sayı. Her uzmanı dörde bölerseniz
seçilebilecek kombinasyon sayısı patlıyor:

```python
from math import comb
print(f"C(16,2)  = {comb(16,2):,}")      # kaba taneli
print(f"C(64,8)  = {comb(64,8):,}")      # her uzman dörde bölünmüş
```

```
C(16,2)  = 120
C(64,8)  = 4,426,165,368
```

Makalenin yayımladığı rakamla birebir aynı. Aynı parametre bütçesi, aynı hesap,
36 milyon kat daha fazla kombinasyon.

### Ne sevk edildi

Gerçek konfigürasyonlar şöyle:

| Model | MoE katmanı başına uzman | Token başına aktif |
|---|---|---|
| Switch Transformer (2021) | 2048'e kadar | **1** |
| DeepSeek-V3 (2024) | 1 paylaşılan + 256 yönlendirilen | 8 + 1 = **9** |
| Qwen3-235B-A22B (2025) | 128, paylaşılan yok | **8** |
| gpt-oss-120b (2025) | 128 | **4** |
| gpt-oss-20b (2025) | 32 | **4** |

DeepSeek-V3'ün sayısını doğru ifade etmek gerekiyor, çünkü genelde yanlış
aktarılıyor. Rapor şöyle diyor (§4.2): *"Her MoE katmanı 1 paylaşılan uzman ve
256 yönlendirilen uzmandan oluşuyor... Yönlendirilen uzmanlar arasından her
token için 8 uzman aktive ediliyor."* Paylaşılan uzman her zaman açık olduğu
için toplam dokuz uzman çalışıyor (a.g.e., §5.2). Yani "256 uzman, 9 aktif"
değil: katman başına **257** uzman var, 8 yönlendirilmiş artı hep açık olan 1
paylaşılan çalışıyor. Toplam 671 milyar parametrenin token başına 37 milyarı,
yani **%5.5**'i.

<aside class="sidenote">

Shazeer ve ark., *Outrageously Large Neural Networks*, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) · Fedus ve ark., *Switch Transformers*, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) · Dai ve ark., *DeepSeekMoE*, [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) · DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) · Qwen Ekibi, *Qwen3 Technical Report*, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) · OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, [arXiv:2508.10925](https://arxiv.org/abs/2508.10925).

</aside>

<!-- FİGÜR: expert konfigürasyonları — toplam vs aktif, paylaşılan uzman farklı biçimle. -->

### İki tartışmalı nokta

Tabloya bakınca "alan ince taneliliğe yakınsadı" demek cazip. Kaynaklar bunu
desteklemiyor.

**Paylaşılan uzman konusunda iki güçlü lab birbirine zıt karar vermiş.**
DeepSeekMoE bunu iki ana fikrinden biri yapıyor. Qwen3 ise tek cümleyle
çıkarıyor (§2): *"Qwen2.5-MoE'nin aksine, Qwen3-MoE tasarımı paylaşılan
uzmanları dışarıda bırakıyor."* Kaldırma kararı için hiçbir ablasyon
yayımlanmamış. İki taraf da kesin diye sunulamaz.

**Uzman sayısının yönü de tek değil.** Switch'in tüm tezi top-1'di; gpt-oss 128
uzmanla top-4 çalışıyor; DeepSeek 256'yla top-8. DeepSeek'in gerekçesi
anlatılmaya değer, ama alanın verdiği ortak bir karar olarak sunulamaz.

Son bir teknik not: yönlendirme dengelenmek zorunda, yoksa birkaç uzman bütün
yükü alıyor. Standart yöntem bir yardımcı kayıp eklemek, ve bunun kaliteden
götürdüğü biliniyor — DeepSeek-V3'ün katkılarından biri tam olarak bunu
aşmak: *"yük dengelemede yardımcı kayıp kullanmayan bir strateji"*, gerekçesi
*"yük dengesini teşvik etme çabasından doğan performans düşüşünü asgariye
indirmek."*

## 7. Normalizasyon

Normalizasyon 2017'de de vardı, hâlâ var — ama hem yeri hem biçimi değişti. Bu
bölüm baştan sona kararlılık baskısının bölümü.

### 7.1 Post-LN ve Pre-LN

İki isim de tek bir soruyu cevaplıyor: **normalizasyon ana yolda mı duruyor,
yoksa daldaki kopyada mı?** İki blok da aynı üç parçayı içeriyor — bir alt
katman (attention ya da FFN), bir artık toplama, bir normalizasyon. Yalnızca
sıra farklı.

**Post-LN** (2017 orijinali): alt katmanı çalıştır, artığı ekle, *sonra*
normalize et. Normalizasyon ana yolun **üstünde**, bloklar arasında duruyor.

$$x_{l+1} = \mathrm{LayerNorm}\big(x_l + \mathrm{Sublayer}(x_l)\big)$$

**Pre-LN** (bugün sevk edilen): önce bir *kopyayı* normalize et, alt katmanı ona
uygula, sonra dokunulmamış artığa ekle. Normalizasyon **dalın içinde**; artık
yolu baştan sona temiz bir kimlik olarak kalıyor.

$$x_{l+1} = x_l + \mathrm{Sublayer}\big(\mathrm{LayerNorm}(x_l)\big)$$

Fark bu kadar, ve ikinci biçimin neden daha iyi davrandığı da burada: ilk
katmandan sonuncuya kadar, hiçbir şeyin yeniden ölçeklemediği kesintisiz bir yol
var.

<!-- FİGÜR: Post-LN vs Pre-LN yan yana; residual yolu kalın, norm kutusunun yeri vurgulu. -->

Xiong ve ark. (2020) bunu teorik olarak gösteriyor:

> "Ortalama alan teorisiyle kanıtlıyoruz ki, başlangıçta, katman
> normalizasyonunu artık blokların arasına yerleştiren orijinal tasarımlı
> Post-LN Transformer'da çıkış katmanına yakın parametrelerin beklenen
> gradyanları büyük. Dolayısıyla bu gradyanlarda büyük bir öğrenme oranı
> kullanmak eğitimi kararsız hâle getiriyor. Isınma aşaması pratikte bu sorunu
> önlemeye yarıyor. Öte yandan teorimiz ayrıca gösteriyor ki, katman
> normalizasyonu artık blokların içine konursa gradyanlar başlangıçta uslu
> davranıyor."

Sonuç pratik: Pre-LN'de öğrenme oranı ısınma aşaması **kaldırılabiliyor**.

Buradaki çerçeveye dikkat: Pre-LN kaliteyle kazanmadı. Model büyüdükçe ayarlama
maliyeti artan bir hiperparametreyi ortadan kaldırdığı için kazandı. Bu yazıdaki
değişikliklerin çoğu böyle — mimari bir zafer değil, operasyonel bir kolaylık.

Pre-LN'in görünür bir bedeli de var: artık akışı yol boyunca hiç normalize
edilmediği için, yığının en tepesine bir son LayerNorm eklemek gerekiyor.

<aside class="sidenote">

Xiong ve ark., *On Layer Normalization in the Transformer Architecture*, [arXiv:2002.04745](https://arxiv.org/abs/2002.04745). İki bloğun adım adım karşılaştırması Tablo 1'de.

</aside>

### 7.2 RMSNorm

İkinci değişiklik normalizasyonun kendi içinde. LayerNorm iki şey yapıyor:
ortalamayı çıkarıyor (yeniden merkezleme) ve standart sapmaya bölüyor (yeniden
ölçekleme). Zhang ve Sennrich (2019) birincisinin gereksiz olduğunu öne sürüyor:

> "LayerNorm'daki yeniden merkezleme değişmezliğinin vazgeçilebilir olduğunu
> öne sürüyoruz ve karesel ortalama karekök katman normalizasyonunu, yani
> RMSNorm'u öneriyoruz... RMSNorm, LayerNorm'a karşı karşılaştırılabilir
> performans elde ederken çalışma süresini farklı modellerde **%7 ilâ %64**
> azaltıyor."

Ortalamayı hesaplamamak bir istatistik geçişini ortadan kaldırıyor; kazanç
tamamen hız. Qwen3, Gemma 3, OLMo 2 ve gpt-oss'un dördü de RMSNorm kullanıyor.

### 7.3 2024–25: yer yeniden tartışmaya açıldı

Pre-LN yerleşik görünüyordu, ama son iki yıl konuyu yeniden açtı.

**OLMo 2 geri dönüyor** (§2.2): *"Her transformer bloğunda attention ve
feed-forward (MLP) katmanlarının girdilerini değil çıktılarını normalize
ediyoruz."* Yani 2017'nin yerleşimine yakın bir noktaya — ama LayerNorm yerine
RMSNorm'la, ve gerekçe yine kararlılık.

**Gemma 3 ikisini birden yapıyor** (§2.1): *"RMSNorm ile hem post-norm hem
pre-norm kullanan bir Grouped-Query Attention (GQA) kullanıyoruz."* Aynı blokta
iki normalizasyon.

Yani "Pre-LN kazandı" cümlesi 2020 için doğruydu, bugün için fazla kesin.

### 7.4 Attention sink — §2.1'de bıraktığımız şerh

Şimdi §2.1'in sonunda bıraktığım şerhe dönebiliriz.

Xiao ve ark. (2023) tuhaf bir gözlemle başlıyor: eğitilmiş modellerde, dizinin
**ilk** token'larına şaşırtıcı derecede yüksek dikkat gidiyor — anlamlı olup
olmadıklarına bakılmaksızın. Bu token'lara *attention sink* diyorlar.

Açıklamaları, bu yazının en zarif parçası (§3.1):

> "Bunun sebebini Softmax işlemine bağlıyoruz; Softmax, attention skorlarının
> bütün bağlam token'ları üzerinde toplamının bire eşit olmasını gerektiriyor.
> Dolayısıyla, mevcut sorgunun önceki token'ların çoğunda güçlü bir eşleşmesi
> olmasa bile, modelin bu gereksiz dikkat değerlerini toplam bir olsun diye bir
> yere yerleştirmesi gerekiyor."

Softmax'ın satır toplamının 1 olması bir tercih değil, tanımın kendisi. Ve bu
kısıt modeli, bakacak bir şey olmadığında bile bir yere bakmaya zorluyor.
Modeller çözümü kendileri buluyor: ilk token'ları çöp kutusu olarak kullanıyorlar.
Öyle ki, kayan pencere kullanırken yalnızca **dört** başlangıç token'ının
anahtar/değerini saklamak performansı geri getirmeye yetiyor.

Ve 2025'te gpt-oss bu kısıttan doğrudan çıkıyor (§2):

> "Her attention kafasının softmax'ın paydasında öğrenilen bir bias'ı var;
> off-by-one attention ve attention sink'lerine benzer şekilde, bu, attention
> mekanizmasının hiçbir token'a dikkat etmemesine imkân veriyor."

Payda artık yalnızca token skorlarının toplamı değil; öğrenilen bir terim de
içeriyor. Sonuç olarak gerçek token'lara giden ağırlıklar **1'e toplanmıyor.**
Model artık "hiçbirine bakmıyorum" diyebiliyor.

Bir kısıt, sekiz yıl arayla iki sonuç: önce modellerin kendi kendine bulduğu bir
telafi, sonra onu gereksiz kılan bir mimari değişiklik.

<aside class="sidenote">

Xiao ve ark., *Efficient Streaming Language Models with Attention Sinks*, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) · Zhang ve Sennrich, *Root Mean Square Layer Normalization*, [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).

</aside>

## 8. FFN ve aktivasyon

Bloğun daha az konuşulan yarısı, ama parametrelerin çoğunu tutan taraf burası.
2017'nin tarifi sade (§3.3): iki doğrusal katman, aralarında bir ReLU, her
pozisyona ayrı ayrı uygulanıyor. Boyutlar $d_{\text{model}} = 512$,
$d_{ff} = 2048$.

Bugünkü karşılığı SwiGLU. Shazeer'in 2020'deki formülasyonu, kapılı doğrusal
birimlerin bir çeşidi:

$$
\mathrm{SwiGLU}(x, W, V) = \mathrm{Swish}_1(xW) \otimes xV, \qquad
\mathrm{Swish}_\beta(x) = x\,\sigma(\beta x)
$$

Üçüncü bir ağırlık matrisi geliyor — girdi hem $W$ hem $V$ ile çarpılıp
sonuçlar eleman bazında çarpılıyor. Ve buradan bir ayrıntı çıkıyor ki, modern
konfigürasyonlardaki tuhaf sayıları açıklıyor (§2):

> "Bu katmanların hepsinde, orijinal FFN'deki ikiye karşılık **üç** ağırlık
> matrisi var. Parametre sayısını ve hesap miktarını sabit tutmak için, bu
> katmanları orijinal iki matrisli sürümle karşılaştırırken gizli birim sayısı
> $d_{ff}$'yi $\frac{2}{3}$ çarpanıyla küçültüyoruz."

2017'nin $4 \times d_{\text{model}}$ oranının bugün neden $\approx 8/3$ gibi
göründüğünün sebebi bu.

Ölçülen fark, parametre ve hesap eşitlenmiş hâlde (Tablo 1, heldout
log-perplexity):

| FFN çeşidi | 65.536 adım | 524.288 adım |
|---|---|---|
| ReLU (temel) | 1.997 | 1.677 |
| GELU | 1.983 | 1.679 |
| Swish | 1.994 | 1.683 |
| GEGLU | **1.942** | **1.633** |
| SwiGLU | 1.944 | 1.636 |

Kazanç gerçek ama mütevazı. Asıl mesele makalenin **sonuç cümlesi**:

> "Bu mimarilerin neden işe yaradığına dair hiçbir açıklama sunmuyoruz;
> başarılarını, her şey gibi, ilahi lütfa bağlıyoruz."

Bu cümle şaka gibi duruyor ama yazının en dürüst yeri. Bu envanterdeki her
bileşenin adı konmuş bir baskısı var — cache, hesap, kararlılık, bağlam. Bunun
bir tablosu ve bir omuz silkmesi var. Ve buna rağmen Llama'dan Qwen3'e,
gpt-oss'tan OLMo 2'ye her yere girdi.

Modern bloğun her parçası açıklanmış değil. Alan da bunu biliyor.

<aside class="sidenote">

Shazeer, *GLU Variants Improve Transformer*, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).

</aside>

## 9. Encoder bloğu

Bileşenleri tek tek gezdik; şimdi bloğu bütün olarak görelim. Önce 2017'nin
hâli, çünkü sonraki her şey bunun bir varyantı (Vaswani ve ark. 2017, §3.1):

> "Encoder, $N=6$ özdeş katmandan oluşan bir yığından meydana geliyor. Her
> katmanın iki alt katmanı var. Birincisi çok başlı bir self-attention
> mekanizması, ikincisi basit, pozisyon bazında tam bağlı bir feed-forward ağı.
> İki alt katmanın her birinin etrafında bir artık bağlantı, ardından katman
> normalizasyonu kullanıyoruz."

<!-- FİGÜR: 2017 encoder bloğu, Post-LN yerleşimi açıkça etiketli. -->

İki alt katman, iki artık bağlantı, iki normalizasyon — ve normalizasyonlar
**alt katmanların çıktısında**, yani Post-LN. Bugün sevk edilen hiçbir modelde
bu yerleşim aynen yok; sebebi §7.1'de.

Encoder bu yazının konusu değil, çünkü §1'de anlattığım ayrışmadan sonra
üretimdeki büyük dil modelleri decoder tarafında toplandı. Ama iskeleti bilmek
gerekiyor, çünkü decoder bloğu bunun üstüne kuruluyor.

## 10. Decoder bloğu

Şimdi hepsini bir araya koyalım. Bugünün bir decoder bloğu, bu yazıda gezdiğimiz
parçalarla:

```
x ──┬─────────────────────────────────────────┐
    │   RMSNorm  →  attention (§2, §4)         │   ← konum: RoPE (§5)
    │              + causal maske (§2.3)       │
    └─────────────────────────────────────── ⊕ ┘
    │
    ├─────────────────────────────────────────┐
    │   RMSNorm  →  FFN: SwiGLU ya da MoE      │      (§6, §8)
    └─────────────────────────────────────── ⊕ ┘
                                              × L katman
```

<!-- FİGÜR: aynı blok, Figure'daki encoder bloğuyla aynı ölçekte çizilmiş hâlde. -->

2017'nin decoder'ıyla karşılaştırınca: cross-attention alt katmanı gitmiş
(bakacak bir encoder yok), normalizasyon yer değiştirmiş ve biçim değiştirmiş,
konum kodlaması gömmeden attention'ın içine taşınmış, feed-forward
seyrekleşmiş.

### Yakınsama yok

Şimdi asıl gözleme gelelim. "Modern transformer" diye tek bir tarif olsaydı, beş
açık ağırlıklı modelin konfigürasyonu birbirine benzerdi. Benzemiyor:

| | DeepSeek-V3 | Qwen3-235B | Gemma 3 | gpt-oss-120b | OLMo 2 13B |
|---|---|---|---|---|---|
| Attention | MLA | GQA 64/4 | GQA + 5:1 yerel/global | GQA 64/8, bantlı ↔ yoğun | **MHA** (40 kafa) |
| Konum | RoPE (decoupled) | RoPE θ=1M + YaRN | RoPE 1M global / 10k yerel | RoPE + YaRN | RoPE θ=5e5 |
| QK-Norm | — | var | var | — | var |
| Normalizasyon | RMSNorm | RMSNorm, pre | RMSNorm, pre **ve** post | RMSNorm, pre | RMSNorm, **çıktıda** |
| FFN | SwiGLU + MoE | SwiGLU + MoE | yoğun | SwiGLU + MoE | SwiGLU, yoğun |
| MoE | 1+256, 9 aktif | 128, 8 aktif, paylaşılan yok | yok | 128, top-4 | yok |
| Katman | 61 | 94 | — | 36 | 40 |

Her hücre modelin kendi raporundan. Beş model, beş farklı cevap — ve
çeliştikleri noktalar rastgele değil: biri MHA'yı koruyor, biri MoE
kullanmıyor, biri paylaşılan uzmanı çıkarıyor, biri normalizasyonu iki yere
birden koyuyor, biri 2017'nin yerleşimine dönüyor.

Yakınsanan şey bileşen listesi değil. Yakınsanan şey **problem listesi**: beşi
de aynı dört baskıyla boğuşuyor, ve birbirinden farklı yerlerde farklı takaslar
yapıyor.

### Bu yazının nerede zayıf olduğu

Kapanışta bir şeyi açıkça söylemek gerekiyor. Yukarıdaki tablonun ve bu yazıdaki
2024–25 sayılarının neredeyse tamamı **teknik raporlardan** geliyor, kontrollü
deneylerden değil. Model kartları neyin sevk edildiğini söylüyor, neyin izole
edildiğini değil.

Ablasyon bulunan yerler var — Gemma 3'ün yerel/global oranı, DeepSeek-V2'nin
MHA/GQA/MQA karşılaştırması, SwiGLU'nun Tablo 1'i, Yang ve ark.'nın 8B'lik
RoPE/NoPE/QK-Norm çalışması — ama bunların her biri tek bir labın kendi reçetesi
üzerinde. Bileşenleri eşit bütçede karşılaştıran, bağımsız, geniş bir çalışma
bu yazının kanıt tabanında **yok**. Çünkü kimse yapmadı: bir frontier modeli iki
kez, tek bir bileşen değiştirilerek eğitmenin maliyeti, sorunun cevabını merak
etmenin bedelinden yüksek.

Dolayısıyla bu yazı şunu iddia edebilir: her değişikliğin adı konmuş bir baskısı
ve o baskıyı belgeleyen birincil bir kaynağı var. Şunu iddia edemez: yapılan
seçimler o baskılara verilebilecek **en iyi** cevaplar. İkincisi ölçülmedi.
