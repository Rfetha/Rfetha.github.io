---
title: "Modern bir Transformer'ın parçaları"
description: 'Bir LLM mimarisinin parçaları: attention türleri, positional encoding, KV-cache ve MoE — 2017 tasarımından bugüne ne eklendi, ne çıkarıldı.'
pubDate: '2026-08-01'
---

## Abstract

DeepSeek-V3'te tek bir isteğin 128K bağlamdaki KV-cache'i 8.6 GiB. Aynı model
2017'nin attention'ıyla kurulsaydı 488 GiB olurdu, ve aradaki fark mimarinin tek
bir bileşeninden geliyor. Buna rağmen bir dil modelinin mimarisi anlatılırken
hâlâ 2017'nin Transformer'ı çiziliyor.

Oysa bugün sevk edilen bir modelin konfigürasyon dosyasını açtığınızda o tarifin
neredeyse hiçbir maddesi yerinde değil: sinüzoidal konum kodlaması, ReLU'lu
feed-forward ağı, alt katmanın çıktısına uygulanan normalizasyon, kendi anahtar
ve değerine sahip attention kafaları — hepsi değişti. Yerinde kalan liste kısa:
attention işleminin kendisi, artık bağlantılar, ve aynı bloğu üst üste yığma
fikri.

Bu yazı aradaki sekiz yılı parça parça geziyor ve her değişikliği adı konmuş bir
baskıya bağlıyor — decode belleği, parametre başına hesap, eğitim kararlılığı,
bağlam uzunluğu — çünkü bunları tek bir sebebe indirmek bu konuda yapılan en
yaygın hata. Sayılar mimarisi kamuya açık modellerden geliyor ve türetilen her
rakam kaynağın kendi yayımladığı sayıyı yeniden üretiyor.

Yakından bakınca bazı gerekçeler ününden zayıf çıkıyor: GQA'nın kaynak makalesi
yöntemi yalnızca encoder-decoder modellerde ölçtüğünü kendi sınırlar bölümünde
yazıyor, 2017'nin kendi ablasyonu öğrenilen konum gömmelerini sinüzoidalle
farksız buluyor, SwiGLU'nun makalesi neden işe yaradığına dair hiçbir açıklama
sunmadığını açıkça söylüyor. Ve ortaya bir "modern transformer tarifi" çıkmıyor:
beş açık ağırlıklı model aynı dört baskıya beş farklı cevap veriyor, hangisinin
doğru olduğunu söyleyecek eşit bütçeli bir karşılaştırma ise kamuya açık
literatürde yok.

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
  Qwen3, Gemma 3, OLMo 2 ve gpt-oss'un dördünde de.
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

### 1.1 Bu yazının anlatmadığı şey

2017'nin tek tasarımı kısa sürede üç aileye ayrıldı — encoder-only,
decoder-only, encoder-decoder — ve
[zamanla decoder-only baskın paradigma haline geldi](/blog/why-decoder-only/).
O ayrışmanın *neden* böyle sonuçlandığı ayrı bir yazının konusu; orada beş aday
cevap birincil kaynaklara karşı tek tek test ediliyor ve çoğu ayakta kalmıyor.

Burada o tartışma yok. Bu yazı ayrışmadan sonrasını anlatıyor: kazanan bloğun
içini, parça parça. Hangi parça 2017'den kalma, hangisi sonradan girdi, ve
girenler neye cevap veriyor.

### 1.2 Dört baskı

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

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-1-timeline.svg" alt="2016'dan 2025'e uzanan zaman çizelgesi. Bileşenler yayımlanma tarihlerine göre yerleştirilmiş ve dört baskıya göre gruplanmış: decode belleği, parametre başına hesap, eğitim kararlılığı, bağlam uzunluğu. Sparsely-gated MoE, Transformer'ın beş ay solunda duruyor.">

<figcaption><b>Figure 1.</b> Bloğun parçaları ve giriş tarihleri (arXiv v1 gönderim tarihleri). Şeritler, her parçanın cevap verdiği baskıyı gösteriyor. Soldaki kırmızı işaret: sparsely-gated <b>MoE</b>, Transformer'dan beş ay önce yayımlandı — içinde yaşadığı bloktan eski.</figcaption>

</figure>

Figürdeki o tarih tesadüf değil, bir uyarı: bu yazıdaki parçaların hepsi 2017'nin
çocuğu değil. Bazıları daha eski fikirlerin, doğru donanım ve doğru ölçek
geldiğinde geri çağrılmış hâlleri.

Envantere değişmeyen parçadan başlayalım.

## 2. Attention: beş çeşit, tek işlem

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
kalanının konusu, o karesel terimin sekiz yıl sonra hangi biçimde geri döndüğü.
Hesap olarak değil: §3'te göreceğimiz gibi, üretim sırasında bağlayıcı olan
kısıt bellek trafiği.

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

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-3-heads.svg" alt="Aynı toplam genişlikte iki kutu dizisi: solda 512 boyutunda tek bir attention kafası, sağda her biri 64 boyutunda sekiz kafa. Altta 512 eşittir 8 çarpı 64.">

<figcaption><b>Figure 3.</b> Kafalar modeli genişletmiyor, <b>bölüyor</b>. Sekiz dar kafa tek geniş kafayla aşağı yukarı aynı parametre ve aynı hesap maliyetinde çalışıyor; kazanılan şey kapasite değil, sekiz ayrı alt uzayda sekiz ayrı ilişki. Bu kutuları daraltmak 2017'nin kendi ablasyonunda kaliteyi düşürmüş — §4'ün üç yöntemi de o yüzden başka bir çarpana basıyor.</figcaption>

</figure>

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

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-4-prefix-invariance.svg" alt="İki panel, her biri üç katman ve altı pozisyonluk bir anahtar-değer ızgarası. Causal maskede ilk beş sütun değişmemiş, sapma sıfır; maske olmayanda bütün sütunlar değişmiş, sapma 1.38.">

<figcaption><b>Figure 4.</b> Aynı yığın, altıncı token eklenmeden önce ve sonra. Causal maskede ilk beş pozisyonun anahtar ve değeri <b>bit düzeyinde</b> aynı kalıyor; maske kalkınca hepsi değişiyor. Soldaki sıfır, KV-cache'i mümkün kılan şeyin kendisi.</figcaption>

</figure>

Bu, bir dizi optimizasyonun kapısını açıyor: eğer önceki anahtar ve değerler
değişmiyorsa, onları yeniden hesaplamak yerine saklayabilirsiniz. §3 tam olarak
bunun hakkında.

### 2.4 Cross-attention

Üçüncü çeşit, ikinci ekseni oynatıyor: sorgu bir yığından, anahtar ve değer
**başka bir yığından** geliyor. 2017'nin tarifi (Vaswani ve ark. 2017, §3.2.3):

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

Beltagy, Peters ve Cohan, *Longformer: The Long-Document Transformer*, [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) · Jiang ve ark., *Mistral 7B*, [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) · Gemma ekibi, *Gemma 3 Technical Report*, [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) · Brown ve ark., *Language Models are Few-Shot Learners* (GPT-3), [arXiv:2005.14165](https://arxiv.org/abs/2005.14165).

</aside>

Sezgiye ters gelen kısım şu: pencere dar olduğu hâlde modelin görebildiği
menzil dar kalmıyor. Mistral 7B bunu açıkça hesaplıyor (Jiang ve ark. 2023, §2): bir attention
katmanında bilgi $W$ token ilerleyebiliyorsa, $k$ katman sonra $k \times W$
token ilerlemiş oluyor. Kendi konfigürasyonlarıyla — $W = 4096$, 32 katman —
*"son katmanda yaklaşık 131K token'lık teorik bir attention menzilimiz oluyor."*

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-5-sliding-window.svg" alt="Sağda sorgulayan token, soluna doğru uzanan altı çubuk. Bir katmanda 4096 token, iki katmanda 8192, dört katmanda 16.384, sekizde 32.768, on altıda 65.536, otuz ikide 131.072.">

<figcaption><b>Figure 5.</b> Tek bir katmanda doğrudan bakış 4096 token, ama menzil katman sayısıyla çarpılıyor. Mistral'in 32 katmanının sonunda 131.072 token'lık teorik bir menzil kalıyor. Ölçek doğrusal — 24 piksel 4096 token.</figcaption>

</figure>

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
  128 token. Desen GPT-3'ten devralınmış — o da mimarisini GPT-2'den aldığını
  söyleyip tek bir istisna sayıyor (Brown ve ark. 2020, §2.1):
  *"transformer'ın katmanlarında dönüşümlü yoğun ve yerel olarak bantlı seyrek
  attention desenleri kullanmamız."* Beş yıl sonra aynı desen geri geliyor.

Neden yapıldığı — yani cache maliyeti — sıradaki bölümün konusu.

## 3. KV-cache: hesabı belleğe çeviren takas

§2.3'ün lemması bir imkân tanımlıyordu: causal bir yığında geçmiş anahtar ve
değerler değişmiyor, dolayısıyla saklanabilirler. Bu bölüm o imkânın faturasını
çıkarıyor.

### 3.1 Bellek faturası

Saklamazsanız ne olur? Her yeni token için bütün diziyi baştan işlemeniz
gerekir; $n$ token üretmek $O(n^3 d)$ mertebesinde attention işi demek.
Saklarsanız her adımda yalnızca yeni token'ın projeksiyonlarını hesaplayıp
biriken anahtarlarla çarpıyorsunuz. Bu, üretimi mümkün kılan optimizasyon.

Bedeli bellek. Formül şu:

<figure>

$$
\text{KV bayt} = 2 \times b \times s \times L \times h_{	ext{kv}} \times d_{	ext{head}} \times \text{bayt}_{\text{dtype}}
$$

<figcaption><b>Figure 6.</b> Baştaki 2 anahtar ve değer için; <i>b</i> yığın boyutu, <i>s</i> o ana kadarki token sayısı, <i>L</i> katman sayısı, <i>h<sub>kv</sub></i> <b>anahtar/değer</b> kafa sayısı, <i>d<sub>head</sub></i> kafa başına boyut.</figcaption>

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

### 3.2 Asıl darboğaz hesap değil, bant genişliği

<aside class="sidenote">

Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150). Artımlı decode'un bellek/aritmetik oranı §2.4.1'de, eğitiminki §2.3.1'de, multi-query önerisi §3'te ve kendi oranı §3.1'de; deneyler §4'te (Tablo 1 kalite, Tablo 2 hız). Aynı yazarın bu yazıda geçen diğer iki makalesi ayrı kaynaklar: sparsely-gated MoE (Shazeer ve ark. 2017) ve GLU çeşitleri (Shazeer 2020); künyeleri bu yazının §6.2 ve §8 bölümlerinde.

</aside>

Buraya kadar cache'i bir kapasite problemi gibi anlattım. Asıl mesele daha
ince, ve bu yazının döndüğü mil o. Shazeer (2019, §2.4.1) artımlı üretimin
aritmetik ve bellek erişimi oranını çıkarıyor: bellek erişiminin işlem sayısına
oranı $\Theta\!\left(\frac{n}{d}+\frac{1}{b}\right)$, ve

> "$n \approx d$ ya da $b \approx 1$ olduğunda oran 1'e yaklaşıyor, bu da bellek
> bant genişliğinin modern donanımda büyük bir performans darboğazı olmasına yol
> açıyor."

Karşılaştırma için: aynı analiz **eğitim** için yapıldığında oran
$O(1/k + 1/(bn))$ çıkıyor (a.g.e., §2.3.1) — yani çok küçük. Cümlenin tamamı şu: **artımlı
decode, eğitimden temelden daha kötü bir donanım rejimi.** Eğitimde çip hesap
yapıyor; decode'da çip çoğunlukla bekliyor.

Oranı gerçek bir çipe oturtunca ne kadar kötü olduğu görünüyor. Bir H100 SXM
yoğun bf16'da 989,5 TFLOPS yapıyor ve HBM'i 3,35 TB/s okuyor; bölünce çipin
doyması için okunan her bayta 295 işlem düşmesi gerektiği çıkıyor. OLMo 2 7B ile:

<aside class="sidenote">

Çip sayıları NVIDIA'nın [H100 ürün sayfasındaki](https://www.nvidia.com/en-us/data-center/h100/) teknik tablodan. Tablodaki 1.979 TFLOPS'un dipnotu *"With sparsity"* diyor; yoğun karşılığı olan 989,5 türetilmiş değer, tabloda ayrıca yazılmıyor. Model sayıları §8'deki `config.json`'dan. Ağırlık trafiği adım başına bir kez okunuyor varsayımıyla; gerçek bir sunucuda parçalama ve önbellek bunu değiştirir, mertebeyi değiştirmez.

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

Tek bir istek servis edilirken çip hesap kapasitesinin **binde üçünü**
kullanıyor. Geri kalan zaman bellek bekliyor. Ve ikinci sütun, akla ilk gelen
çözümün neden yetmediğini söylüyor: batch büyütmek yoğunluğu yükseltiyor ama
64'te bile %2'yi geçmiyor, çünkü **KV trafiği de batch'le birlikte büyüyor** —
ağırlıklar bir kez okunup bütün diziler arasında paylaşılırken cache
paylaşılmıyor. b=64'te okunan her yüz baytın doksanı KV.

Sıradaki bölümün üç yöntemi de bu son cümleye saldırıyor.

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

## 4. MQA, GQA, MLA: tek çarpana üç saldırı

§3'ün problemi tek cümleyle şu: cache çok büyük ve her token için baştan
okunuyor. Formüle bakınca kısılabilecek çarpanlar sınırlı — $L$ ve $d_{	ext{head}}$
modelin kendisi, $s$ kullanıcının isteği, $b$ zaten geliri artıran şey. Geriye
**$h_{	ext{kv}}$** kalıyor: anahtar/değer kafalarının sayısı. Bu bölüm o çarpana
yapılmış üç saldırı, kronolojik sırayla.

### 4.1 MQA — bütün kafalar tek bir anahtar kümesini paylaşsın

Shazeer'in 2019'daki önerisi olabilecek en basit hamle (Shazeer 2019, §3): *"Multi-query
attention, multi-head ile birebir aynı; tek fark, farklı kafaların tek bir
anahtar ve değer kümesini paylaşması."*

Formülde $h_{	ext{kv}} = 1$ demek: cache $h$ kat küçülüyor. Yeni bellek/aritmetik
oranı $\Theta\!\left(\frac{1}{d}+\frac{n}{dh}+\frac{1}{b}\right)$ (a.g.e., §3.1) — kendi
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
ne kadar kaybettirdiği.** Shazeer'in kendi okuması (a.g.e., §4.2): multi-query modeli
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
sanıldığından zayıf (a.g.e., §3.3): *"grup sayısını MQA'dan itibaren artırmak başta
yalnızca mütevazı yavaşlamalara yol açıyor, MHA'ya yaklaştıkça maliyet
artıyor. Uygun bir orta yol olarak 8 grup seçtik."* Tek bir labın kendi eğrisi
üzerinde seçtiği bir orta nokta; türetilmiş bir optimum değil.

**Ve makalenin kendi sınırlar bölümü, bu yazının en dürüst alıntısı** (a.g.e., *Limitations*)**:**

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

<aside class="sidenote">

DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) — yazıda **2024a**. MLA'nın tanımı §2.1.2'de, RoPE uyuşmazlığı ve decoupled çözümü §2.1.3'te. DeepSeek-V3'ün teknik raporu ayrı bir kaynak (2024b, §6.2'deki künye).

</aside>

DeepSeek üçüncü bir yol seçiyor. MQA ve GQA anahtar/değer **kafalarını**
azaltıyor; MLA anahtar ve değeri hiç saklamıyor, onların yerine düşük ranklı tek
bir **latent vektör** saklıyor (DeepSeek-AI 2024a, §2.1.2): *"Çıkarım sırasında MLA
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
güçlü."* Raporun özeti rakamları veriyor: DeepSeek 67B'ye kıyasla
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

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-7-kv-footprint.svg" alt="Dört yatay çubuk: MHA 1.998.848 eleman ve 488 GiB, GQA-8 124.928 eleman ve 30,5 GiB, MQA 15.616 eleman ve 3,8 GiB, MLA 35.136 eleman ve 8,6 GiB. MLA'nın çubuğu MHA'nınkinin elli yedide biri.">

<figcaption><b>Figure 7.</b> Aynı model, dört mekanizmayla. Yalnız <b>MLA</b> sevk edildi; diğer üçü DeepSeek-V3'ün konfigürasyonuna uygulanmış hâlleri. Kapasite sütunu DeepSeek'in kendi etiketleri — bağımsız bir ölçüm değil.</figcaption>

</figure>

### 4.4 RoPE ile çarpışma

MLA'nın hikâyesinde, bu yazıdaki en güzel ayrıntı duruyor: yöntem, bağımsız bir
gerekçeyle seçilmiş başka bir bileşenle **matematiksel olarak uyuşmuyor.**

DeepSeek-AI 2024a, §2.1.3:

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

## 5. Positional encoding: toplamaktan döndürmeye

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

<figcaption><b>Figure 8.</b> Sinüzoidal konum kodlaması. Dalga boyları 2&pi;'den 10000 &middot; 2&pi;'ye uzanan geometrik bir dizi oluşturuyor.</figcaption>

</figure>

Neden bu fonksiyon? Makalenin gerekçesi bağıl konum: *"modelin bağıl konumlara
göre dikkat etmeyi kolayca öğrenebileceğini varsaydığımız için bu fonksiyonu
seçtik, çünkü herhangi bir sabit $k$ kayması için $PE_{pos+k}$, $PE_{pos}$'un
doğrusal bir fonksiyonu olarak yazılabiliyor."*

Bu cümlenin altını çizin: istenen şey **bağıl** konum, elde edilen şey ise
modelin bunu **öğrenebileceği** bir mutlak sinyal. Umut, garanti değil.

Ve hemen ardından gelen cümle, bu bölümün en şaşırtıcı bilgisi (a.g.e., §3.5):

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
(a.g.e., §3.2.1):

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

Kazanılan şey, §5.1'deki umudun özdeşliğe dönüşmesi (a.g.e., §3.2.2):

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

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-9-rotate-vs-add.svg" alt="İki daire. Solda gömmeye konum vektörü eklendiğinde toplamın ucu dairenin dışına çıkıyor. Sağda vektör aynı yarıçapta döndürülüyor ve uç daire üzerinde kalıyor. Her iki normun ölçülen değeri 7.315344.">

<figcaption><b>Figure 9.</b> Toplamak vektörün boyunu değiştiriyor, döndürmek değiştirmiyor. RoPE'un rotasyon matrisi <b>ortogonal</b> olduğu için norm koruması bir gözlem değil, garanti.</figcaption>

</figure>

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

Peki $\theta$'yı büyütmek tam olarak neyi satın alıyor? OLMo 2'nin gerekçesi
*"çözünürlüğü artırıyor"* diyordu; frekansları yazınca bunun böyle olmadığı
görünüyor. RoPE'un en hızlı dönen düzleminin frekansı $\theta^{0} = 1$, yani
dalga boyu $2\pi$ — **$\theta$ ne olursa olsun 6,28 token.** Oynayan yalnızca
öteki uç:

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

Alınan şey çözünürlük değil **menzil**: 10k'dan 1M'ye çıkmak en yavaş düzlemin
dalga boyunu 54 binden 5 milyona taşıyor, en hızlısına dokunmuyor.

Son satır ayrı bir hikâye, ve §4.4'ün devamı. DeepSeek-V3 taban frekansını hiç
oynatmamış — sevk edilen konfigürasyonda $\theta$ hâlâ 10.000, yani Vaswani'nin
sabiti. Üstelik decoupled RoPE yalnızca 64 boyutu döndürdüğü için üs de küçülüyor
ve en yavaş dalga boyu 47 bine iniyor. 163.840 token'lık bağlamda bu **3,48 tam
tur** demek: beşi içinde bağlamı en yavaş dalga boyunu aşan tek model. §4.4'te
"konfigürasyon dosyasında görünür bir sayı olarak kalmış" dediğimiz uzlaşmanın
ikinci yarısı burada — o oyuk, konum çözünürlüğünden ödünç alınmış.

<aside class="sidenote">

Frekans tanımı Su ve ark. 2021, Denk. 15. Konfigürasyonlar modellerin yayımlanmış `config.json` dosyalarından: `deepseek-ai/DeepSeek-V3` (`rope_theta` 10000, `qk_rope_head_dim` 64, `max_position_embeddings` 163840), `google/gemma-3-27b-it` (`rope_theta` 1e6, `rope_local_base_freq` 1e4), `allenai/OLMo-2-1124-7B` (5e5), `Qwen/Qwen3-235B-A22B` (1e6). Oranın 1'in altında olması YaRN'a ihtiyaç olmadığı anlamına gelmiyor — Qwen3 de kullanıyor, sebebi eğitim uzunluğu. DeepSeek-V3 telafiyi çalışma zamanında yapıyor: `rope_scaling` YaRN, `factor` 40, `original_max_position_embeddings` 4096.

</aside>

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
eğitip karşılaştırıyor. Bulguları (a.g.e., §2.2):

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

Bu bir benzetme değil. Kazemnejad ve ark. maskenin konumu *taşıyabildiğini*
ispatlamakla kalmıyor, ağırlıkları açıkça inşa ediyor (a.g.e., Ek C.1): gömmenin
bir boyutu her token'da 1, bir boyutu yalnız ilk token'da 1 olsun; anahtar
birinciyi, değer ikinciyi okusun. O zaman bütün anahtarlar özdeş olur, softmax
causal önek üzerinde düzleşir, ve çıktı tam olarak önekin uzunluğunun tersi
çıkar. Konum kodlaması olmadan:

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

Tek bir attention katmanı, hiçbir konum sinyali verilmeden, her pozisyonun
sırasını okunabilir biçimde taşıyor. Maskenin sayaç olması bir sezgi değil,
kurulabilir bir yapı.

Kazemnejad ve ark. (2023) bunu sistematik olarak ölçüyor: sıfırdan eğitilmiş
decoder-only modellerde APE, T5-bağıl, ALiBi, RoPE ve hiçbir kodlama olmayan
NoPE'yi uzunluk genellemesinde karşılaştırıyorlar. Özetlerinin sonucu:

<aside class="sidenote">

Kazemnejad ve ark., *The Impact of Positional Encoding on Length Generalization in Transformers*, [arXiv:2305.19466](https://arxiv.org/abs/2305.19466). Alıntı özetten; deney kurgusu §3'te, karşılaştırma §4'te.

</aside>


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
ölçüldüğünde net biçimde ayrışıyor (Yang ve ark. 2025, §3): NoPE katmanları bilgi getirmede güçlü —
aradıkları token'a yüksek dikkat veriyorlar; RoPE katmanları ise güçlü bir
yakınlık eğilimi gösteriyor, yani son token'lara yaslanıyorlar. Biri arıyor,
öteki yakındakine bakıyor.

Aynı yerde, §5.3'ün ayar düğmesinin bedeli de ölçülmüş: $\theta$ büyüdükçe RoPE
katmanlarının yakınlık eğilimi azalıyor, ama bu kez alıcı alan genişlediği için
gürültü artıyor ve **NoPE katmanlarının arama yeteneği bozuluyor.** $\theta$
10.000'den 4 milyona çıktığında aranan token'a düşen dikkat 0.0765'ten 0.0369'a,
değerlendirme skoru 8.036'dan 6.203'e iniyor. Taban frekansını büyütmek bedava
değil.

## 6. Mixture-of-Experts: parametreyi hesaptan ayırmak

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

### 6.1 Kaç uzman, kaçı açık?

Switch Transformer (Fedus ve ark. 2021) en radikal sadeleştirmeyi yapıyor
(a.g.e., §2.1): *"Bu fikirlerin aksine, biz yalnızca tek bir uzmana yönlendiren
sadeleştirilmiş bir strateji kullanıyoruz. Bu sadeleştirmenin model kalitesini
koruduğunu, yönlendirme hesabını azalttığını ve daha iyi performans gösterdiğini
ortaya koyuyoruz."* Aynı hesap bütçesinde T5'e karşı 7 kattan fazla ön eğitim
hızlanması bildiriyorlar.

DeepSeekMoE (Dai ve ark. 2024) ters yöne gidiyor — daha çok, daha küçük uzman —
ve iki fikir öneriyor (a.g.e., §1): *"(1) uzmanları $mN$ tanesine ince taneli biçimde
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

### 6.2 Ne sevk edildi

Gerçek konfigürasyonlar şöyle:

| Model | MoE katmanı başına uzman | Token başına aktif |
|---|---|---|
| Switch Transformer (2021) | 2048'e kadar | **1** |
| DeepSeek-V3 (2024) | 1 paylaşılan + 256 yönlendirilen | 8 + 1 = **9** |
| Qwen3-235B-A22B (2025) | 128, paylaşılan yok | **8** |
| gpt-oss-120b (2025) | 128 | **4** |
| gpt-oss-20b (2025) | 32 | **4** |

DeepSeek-V3'ün sayısını doğru ifade etmek gerekiyor, çünkü genelde yanlış
aktarılıyor. Rapor şöyle diyor (DeepSeek-AI 2024b, §4.2): *"Her MoE katmanı 1 paylaşılan uzman ve
256 yönlendirilen uzmandan oluşuyor... Yönlendirilen uzmanlar arasından her
token için 8 uzman aktive ediliyor."* Paylaşılan uzman her zaman açık olduğu
için toplam dokuz uzman çalışıyor (a.g.e., §5.2). Yani "256 uzman, 9 aktif"
değil: katman başına **257** uzman var, 8 yönlendirilmiş artı hep açık olan 1
paylaşılan çalışıyor. Toplam 671 milyar parametrenin token başına 37 milyarı,
yani **%5.5**'i.

<aside class="sidenote">

Shazeer ve ark., *Outrageously Large Neural Networks*, [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) · Fedus ve ark., *Switch Transformers*, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) · Dai ve ark., *DeepSeekMoE*, [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) · DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — yazıda **2024b** · Qwen ekibi, *Qwen3 Technical Report*, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) · OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, [arXiv:2508.10925](https://arxiv.org/abs/2508.10925).

</aside>

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-10-moe-configs.svg" alt="Dağılım grafiği: yatay eksende katman başına toplam uzman logaritmik, dikey eksende token başına aktif uzman. Switch 2048'den 1, gpt-oss-20b 32'den 4, gpt-oss-120b 128'den 4, Qwen3 128'den 8, DeepSeek-V3 257'den 9.">

<figcaption><b>Figure 10.</b> Beş model, beş konfigürasyon; yatay eksen logaritmik. Noktalar dağınık: ne toplam uzman sayısında ne aktif sayıda ortak bir karar var. Switch'in tüm tezi top-1'di, bugün kimse orada değil — ama birbirlerinde de değiller.</figcaption>

</figure>

### 6.3 İki tartışmalı nokta

Tabloya bakınca "alan ince taneliliğe yakınsadı" demek cazip. Kaynaklar bunu
desteklemiyor.

**Paylaşılan uzman konusunda iki güçlü lab birbirine zıt karar vermiş.**
DeepSeekMoE bunu iki ana fikrinden biri yapıyor. Qwen3 ise tek cümleyle
çıkarıyor (Qwen ekibi 2025, §2): *"Qwen2.5-MoE'nin aksine, Qwen3-MoE tasarımı
paylaşılan uzmanları dışarıda bırakıyor."* Kendi kaldırma kararı için hiçbir
ablasyon yayımlamıyorlar.

Ama üçüncü bir lab yayımlıyor, ve bu yazının MoE kanıtları içinde eşit bütçede
yapılmış tek karşılaştırma o. OLMoE iki modeli yan yana eğitiyor: aktif
parametre, toplam parametre ve FLOP aynı; tek fark, birinde 32 yönlendirilen
uzmandan 4'ü açık, ötekinde 1 hep açık paylaşılan uzman artı 31
yönlendirilenden 3'ü. Sonuç (Muennighoff ve ark. 2024, §4.1.3):

> "Her iki ayar da benzer performansa yol açsa da, bir uzmanı paylaşmak biraz
> daha kötü sonuç veriyor."

Asıl ilginç olan gerekçeleri, çünkü §6.1'deki kombinasyon argümanının ta
kendisi — bu kez ters yöne çalışıyor:

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

İki sayı da makalenin yayımladıklarıyla birebir aynı; kendi ifadeleriyle,
yönlendirilen uzmanlardan birini alıp paylaşılan yapmak *"olası kombinasyonların
neredeyse %90'ını ortadan kaldırıyor."*

Desen şu: DeepSeekMoE ince taneliliği **savunmak** için hangi argümanı kurduysa,
OLMoE aynı argümanı paylaşılan uzmana **karşı** kullanıyor. İkisi de kombinasyon
sayısını büyütmek istiyor; paylaşılan uzman o sayıyı küçültüyor. Aynı çalışma
ince tanelilik tarafını da ölçüp DeepSeekMoE'yi doğruluyor (a.g.e., §4.1.2):
uzman boyutu dörde bölünüp sayı 8'den 32'ye çıkarıldığında — yine sabit aktif
parametre ve sabit hesapla — HellaSwag'da *"yaklaşık %10"* iyileşme buluyorlar.
§6.1'deki kombinasyon patlaması bir iddia değil, ölçülmüş bir etki.

<aside class="sidenote">

Muennighoff ve ark., *OLMoE: Open Mixture-of-Experts Language Models*, [arXiv:2409.02060](https://arxiv.org/abs/2409.02060). Paylaşılan uzman ablasyonu §4.1.3'te (Şekil 6), ince tanelilik §4.1.2'de (Şekil 5). Her iki deneyde de aktif ve toplam parametre ile FLOP sabit tutulmuş.

</aside>

Sınırını da söylemek gerekiyor: OLMoE 1B aktif / 7B toplam ölçeğinde, yani
DeepSeek-V3'ün iki mertebe altında, ve yine tek bir labın kendi reçetesi
üzerinde. Soruyu kapatmıyor — ama tarafların birinde artık ölçülmüş bir sayı
var, ötekinde yok.

**Uzman sayısının yönü de tek değil.** Switch'in tüm tezi top-1'di; gpt-oss 128
uzmanla top-4 çalışıyor; DeepSeek 256'yla top-8. DeepSeek'in gerekçesi
anlatılmaya değer, ama alanın verdiği ortak bir karar olarak sunulamaz.

Son bir teknik not: yönlendirme dengelenmek zorunda, yoksa birkaç uzman bütün
yükü alıyor. Standart yöntem bir yardımcı kayıp eklemek, ve bunun kaliteden
götürdüğü biliniyor — DeepSeek-V3'ün katkılarından biri tam olarak bunu
aşmak: *"yük dengelemede yardımcı kayıp kullanmayan bir strateji"*, gerekçesi
*"yük dengesini teşvik etme çabasından doğan performans düşüşünü asgariye
indirmek."*

## 7. Normalizasyon: yeri de biçimi de değişti

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

<figure>

<img loading="lazy" decoding="async" src="/figures/transformer-architecture-components/fig-11-postln-preln.svg" alt="İki blok şeması, her biri iki katman. Post-LN'de normalizasyon kutusu ana yolun üzerinde ve dikey artık çizgisini kesiyor. Pre-LN'de normalizasyon dalın içinde ve dikey çizgi baştan sona kesilmeden geçiyor.">

<figcaption><b>Figure 11.</b> Tek fark normalizasyon kutusunun <b>yeri</b>. Post-LN'de ana yolun üstünde, Pre-LN'de dalın içinde — sağdaki şemada dikey çizgiyi hiçbir kutu kesmiyor. Kazanan biçim kaliteyle değil, ısınma aşamasını kaldırılabilir kıldığı için kazandı.</figcaption>

</figure>

Xiong ve ark. (2020) bunu teorik olarak gösteriyor (a.g.e., özet):

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
ölçekleme). Zhang ve Sennrich (2019) birincisinin gereksiz olduğunu öne sürüyor (a.g.e., özet):

> "LayerNorm'daki yeniden merkezleme değişmezliğinin vazgeçilebilir olduğunu
> öne sürüyoruz ve karesel ortalama karekök katman normalizasyonunu, yani
> RMSNorm'u öneriyoruz... RMSNorm, LayerNorm'a karşı karşılaştırılabilir
> performans elde ederken çalışma süresini farklı modellerde **%7 ilâ %64**
> azaltıyor."

Ortalamayı hesaplamamak bir istatistik geçişini ortadan kaldırıyor; kazanç
tamamen hız. Qwen3, Gemma 3, OLMo 2 ve gpt-oss'un dördü de RMSNorm kullanıyor.

Şu %7–%64 aralığına dikkat: dokuz kat açıklık, ve 2019'un donanımında, o
dönemin modellerinde ölçülmüş. Makalenin kalite iddiası da "daha iyi" değil,
"karşılaştırılabilir". RMSNorm'u sevk eden dört modelin hiçbiri kendi ölçeğinde
LayerNorm'a karşı bir ablasyon yayımlamıyor — yani bugünkü yaygınlığı ölçülmüş
bir üstünlüğe değil, bedava görünen bir hıza dayanıyor.

### 7.3 2024–25: yer yeniden tartışmaya açıldı

Pre-LN yerleşik görünüyordu, ama son iki yıl konuyu yeniden açtı.

**OLMo 2 geri dönüyor** (OLMo ekibi 2025, §2.2): *"Her transformer bloğunda attention ve
feed-forward (MLP) katmanlarının girdilerini değil çıktılarını normalize
ediyoruz."* Yani 2017'nin yerleşimine yakın bir noktaya — ama LayerNorm yerine
RMSNorm'la, ve gerekçe yine kararlılık.

**Gemma 3 ikisini birden yapıyor** (Gemma ekibi 2025, §2.1): *"RMSNorm ile hem post-norm hem
pre-norm kullanan bir Grouped-Query Attention (GQA) kullanıyoruz."* Aynı blokta
iki normalizasyon.

Yani "Pre-LN kazandı" cümlesi 2020 için doğruydu, bugün için fazla kesin.

### 7.4 Attention sink — §2.1'de bıraktığımız şerh

Şimdi §2.1'in sonunda bıraktığım şerhe dönebiliriz.

Xiao ve ark. (2023) tuhaf bir gözlemle başlıyor: eğitilmiş modellerde, dizinin
**ilk** token'larına şaşırtıcı derecede yüksek dikkat gidiyor — anlamlı olup
olmadıklarına bakılmaksızın. Bu token'lara *attention sink* diyorlar.

Açıklamaları, bu yazının en zarif parçası (Xiao ve ark. 2023, §3.1):

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

Ve 2025'te gpt-oss bu kısıttan doğrudan çıkıyor (OpenAI 2025, §2):

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

## 8. FFN: gerekçesi yazılmamış tek değişiklik

Bloğun daha az konuşulan yarısı, ama parametrelerin çoğunu tutan taraf burası.
2017'nin tarifi sade (Vaswani ve ark. 2017, §3.3): iki doğrusal katman, aralarında bir ReLU, her
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
konfigürasyonlardaki tuhaf sayıları açıklıyor (Shazeer 2020, §2):

> "Bu katmanların hepsinde, orijinal FFN'deki ikiye karşılık **üç** ağırlık
> matrisi var. Parametre sayısını ve hesap miktarını sabit tutmak için, bu
> katmanları orijinal iki matrisli sürümle karşılaştırırken gizli birim sayısı
> $d_{ff}$'yi $\frac{2}{3}$ çarpanıyla küçültüyoruz."

2017'nin $4 \times d_{\text{model}}$ oranının bugün neden $\approx 8/3$ gibi
göründüğünün sebebi bu.

Sevk edilmiş bir konfigürasyonda iki iddiayı birden kontrol edebiliriz: FFN
gerçekten parametrelerin çoğunu mu tutuyor, ve $\frac{2}{3}$ düzeltmesi bütçeyi
gerçekten sabit mi tutuyor. OLMo 2 7B'nin `config.json`'ı $d_{\text{model}} =
4096$, $d_{ff} = 11008$ ve 32 katman veriyor — ayrıca `num_key_value_heads`'i
kafa sayısına eşit, yani §2.2'de söylenen MHA burada da görünüyor.

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

İki sonuç çıkıyor. Birincisi bölümün açılış cümlesini hak ediyor: blok
parametrelerinin **üçte ikisi** FFN'de. Attention çok konuşulan taraf, ama
ağırlığı taşıyan taraf değil.

İkincisi daha ilginç. 11008 keyfî bir sayı değil — $\frac{8}{3} \times 4096 =
10922.7$'nin 128'in katına yükseltilmiş hâli. Sonuç, 2017'nin iki matrisli
tarifinden yalnızca **%0.8** uzakta. Üçüncü matris bedava gelmiyor; yeri gizli
birim sayısından kesiliyor. Shazeer'in düzeltmesi kâğıt üstünde kalmış bir
tavsiye değil, sevk edilen dosyada görünen bir sayı.

<aside class="sidenote">

Konfigürasyon iki bağımsız yerden doğrulandı. OLMo ekibi 2025, Tablo 4 ve §2.2: *"karşılık gelen gizli boyutu yaklaşık $\frac{8}{3}d$ alıyoruz, ama verimi artırmak için 128'in en yakın katına yükselterek (7B modelimiz için 11.008)."* Ve modelin yayımlanmış [`config.json`](https://huggingface.co/allenai/OLMo-2-1124-7B/blob/main/config.json)'ı: `hidden_size` 4096, `intermediate_size` 11008, `num_hidden_layers` 32, `num_attention_heads` = `num_key_value_heads` = 32. Üç matrisin varlığı `transformers`'ın [`Olmo2MLP`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py) uygulamasında da açık: `gate_proj`, `up_proj`, `down_proj` — üçü de `bias=False`. Sayım yalnız matrisleri kapsıyor, gömme ve çıkış katmanını değil.

</aside>

Ölçülen fark, parametre ve hesap eşitlenmiş hâlde (Tablo 1, heldout
log-perplexity):

| FFN çeşidi | 65.536 adım | 524.288 adım |
|---|---|---|
| ReLU (temel) | 1.997 | 1.677 |
| GELU | 1.983 | 1.679 |
| Swish | 1.994 | 1.683 |
| GEGLU | **1.942** | **1.633** |
| SwiGLU | 1.944 | 1.636 |

Kazanç gerçek ama mütevazı. Asıl mesele makalenin **sonuç cümlesi** (a.g.e., §4):

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

## 9. Blok bütün hâlde: beş model, beş cevap

Bileşenleri tek tek gezdik. Şimdi ikisini yan yana koyalım: 2017'nin çizdiği
blok, ve bugün sevk edilen blok. Aradaki sekiz yılın tamamı bu iki şemanın
farkında duruyor.

### 9.1 2017: encoder bloğu

Orijinal tarif, makalenin kendi cümleleriyle (Vaswani ve ark. 2017, §3.1):

> "Encoder, $N=6$ özdeş katmandan oluşan bir yığından meydana geliyor. Her
> katmanın iki alt katmanı var. Birincisi çok başlı bir self-attention
> mekanizması, ikincisi basit, pozisyon bazında tam bağlı bir feed-forward ağı.
> İki alt katmanın her birinin etrafında bir artık bağlantı, ardından katman
> normalizasyonu kullanıyoruz."

İki alt katman, iki artık bağlantı, iki normalizasyon — ve normalizasyonlar
**alt katmanların çıktısında**, yani Post-LN. Bugün sevk edilen hiçbir modelde
bu yerleşim aynen yok; sebebi §7.1'de.

### 9.2 Bugün: decoder bloğu

Aynı iskelet, bu yazıda gezdiğimiz parçalarla doldurulmuş hâlde:

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

2017'nin decoder'ıyla karşılaştırınca: cross-attention alt katmanı gitmiş
(bakacak bir encoder yok), normalizasyon yer değiştirmiş ve biçim değiştirmiş,
konum kodlaması gömmeden attention'ın içine taşınmış, feed-forward
seyrekleşmiş. Değişmeyen şey iskeletin kendisi: iki alt katman, iki artık
bağlantı, ve aynı bloğun üst üste yığılması.

### 9.3 Yakınsama yok

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

Her hücre modelin kendi raporundan. Gemma 3 sütunu tek bir boyutu değil aileyi
anlatıyor; rapor bu seçimleri bütün boyutlar için ortak veriyor. Beş model, beş
farklı cevap — ve çeliştikleri noktalar rastgele değil: biri MHA'yı koruyor,
ikisi MoE kullanmıyor, biri paylaşılan uzmanı çıkarıyor, biri normalizasyonu
iki yere birden koyuyor, biri 2017'nin yerleşimine dönüyor.

Yakınsanan şey bileşen listesi değil. Yakınsanan şey **problem listesi**: beşi
de aynı dört baskıyla boğuşuyor, ve birbirinden farklı yerlerde farklı takaslar
yapıyor.

### 9.4 Bu yazının nerede zayıf olduğu

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
