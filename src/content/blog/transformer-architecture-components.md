---
title: 'The Components of a Modern Transformer'
description: 'Bir LLM mimarisinin parçaları: attention türleri, positional encoding, KV-cache ve MoE — 2017 tasarımından bugüne ne eklendi, ne çıkarıldı.'
pubDate: '2026-08-01'
placeholder: true
---

<!--
İSKELET — metin bölüm bölüm buraya yazılacak.
Plan ve kanıt referansları:
  drafts/outline-transformer-components.md          (bölüm planı, uzunluk, figür/kod)
  drafts/research-transformer-components-evidence.md (E§n — iddia/kanıt/atıf)
  drafts/research-decoder-only-math.md               (M§n — türetimler)

TEZ: bugünün bloğundaki her parça ya 2017'den olduğu gibi kalmış, ya da adı
konmuş bir baskının cevabı olarak girmiş.

DÖRT BASKI — ayrı tutulacak (E§0 V7). MoE'yi KV-cache hikâyesine katmak
bu yazıda yapılabilecek en kötü hata.
  1. Decode belleği / bant genişliği  → §6, §7
  2. Parametre başına hesap           → §9
  3. Ölçekte eğitim kararlılığı       → §8.4, §10
  4. Bağlam uzunluğu                  → §8

SINIR: /blog/why-decoder-only/ maskenin SEÇİMİNİ anlatıyor. Bu yazı o seçimin
sonrasını anlatıyor. Maske matematiği, prefix_len, -∞ argümanı, üç aile
figürleri, MLM vs otoregresif — hiçbiri buraya girmeyecek.

BİTİRİRKEN: Türkçe abstract yaz · placeholder: true satırını sil ·
drafts/transformer-architecture-components.md sil.
-->

## Abstract

<!--
Türkçe yazılacak, metin bitince (İngilizce çeviri en son ayrı adım).
Taşıması gerekenler: 2017 tek tasarım · üç aileye ayrışma bir cümle + link ·
dört baskı · bu yazının envanter olduğu · sayıların open-weight modellerden
geldiği ve doğrulanabilir olduğu · yakınsamanın bileşen listesinde değil
baskılar kümesinde olduğu.
-->

---

## 1. 2017'de tek bir blok vardı

<!--
NE: 2017'de tek blok. Üç aileye ayrıldı, decoder-only baskın hale geldi —
ama bu yazının konusu ayrışma değil, kazanan bloğun İÇİ. Ardından dört baskı
isimlendirilir; yazının haritası bu.

ZORUNLU: "zamanla decoder-only baskın paradigma haline geldi" cümlesi METİN İÇİ
link taşıyacak → /blog/why-decoder-only/. Kenar notu değil. İki yazının tek
temas noktası.

KANIT: E§1 zaman çizelgesi (25 satır, arXiv v1) · E§0 V7 · Vaswani §3.1.
~600 kelime · Figür 1: zaman çizelgesi (MoE'nin Transformer'dan 5 ay önce
durması görünmeli) · kod yok.
-->

## 2. Self-attention

<!--
NE: değişmeyen parça. Formül, √d_k'nin neden orada olduğu, karesel maliyet.
ŞERH (E§0 V8): "2017'den beri değişmedi" şerh istiyor — gpt-oss softmax
paydasına öğrenilen bias koyuyor. Tek cümle haber ver, hesabı §10.4'te.

KANIT: M§1.1 tanım · M§1.2 varyans argümanı · M§7 Vaswani Tablo 1.
~550 kelime · figür yok (formül + karmaşıklık tablosu) · Kod 1: √d_k varyansı.
-->

## 3. Multi-head attention

<!--
NE: kafalar d_model'i BÖLÜYOR, çoğaltmıyor. Ve 2017'nin kendi ablasyonu:
d_k'yi küçültmek kaliteyi düşürüyor → §7'yi kuruyor (MQA/GQA d_k'yi değil
KV kafa sayısını kısıyor, farklı düğme).

ŞERH (E§0 V1): OLMo 2 için YALNIZCA konfigürasyon gerçeği. "Ablasyon MHA'yı
destekliyor" YAZILMAYACAK — makalede öyle bir ablasyon yok.

KANIT: Vaswani §3.2.2 (h=8, d_k=d_v=64) · §6.2 satır (B) · M§1.3 · E§3.
~500 kelime · Figür 2: 1 geniş kafa vs 8 dar kafa, aynı parametre · kod yok.
-->

## 4. Causal attention ve prefix-invariance

<!--
NE: maske tanımı İKİ CÜMLE + link. Asıl içerik prefix-invariance: token eklemek
önceki K/V'yi değiştirmiyor, bidirectional'da değiştiriyor. Cache'i mümkün kılan
şey bu; §6'dan §7'ye kadar her şey buradan doğuyor.

SINIR: maske matematiği, prefix_len, -∞ argümanı BURAYA GİRMEYECEK.

KANIT: M§6.1 lemma + corollary · E§4.
~650 kelime · Figür 3: iki panel, causal'da ilk beş K/V değişmemiş ·
Kod 2 (_src/prefix_inv.py, çıktı hazır):
  causal      0.00e+00   ← yaklaşık değil TAM sıfır, argüman bu
  çift yönlü  1.38e+00
-->

## 5. Cross-attention

<!--
NE: tanım yayındaki yazıda. Payı MALİYET: encoder bir kez koşuyor, cross-attn
K/V'si bir kez hesaplanıp bütün üretim boyunca cache'leniyor.

DÜZELTME (M§6.4-1): "yalnızca decoder-only KV-cache kullanabilir" YANLIŞ.
Causal maskeli her decoder cache'ler, T5 dahil. Doğrusu dar: büyüyen dizi
üzerinde çift yönlü encoder cache'leyemez.

KANIT: Vaswani §3.2.3 birebir · M§6.4-1.
~350 kelime — en kısa bölüm · figür yok · kod yok.
-->

## 6. KV-cache

<!--
NE: §4'ün lemması cache'i mümkün kıldı; burada ne kadar tuttuğu ve neden bedava
olmadığı. Bölümün sonu §7'nin kapısı.

ANAHTAR: Shazeer'in oranı — incremental decode eğitimden TEMELDEN daha kötü bir
donanım rejimi. Yazının döndüğü mil.

ŞERH (M§6.4-2): "cache decode'u ucuzlatır" eksik — hesap problemini bellek
kapasitesi/bant genişliği problemine çeviriyor. Bütün MQA/GQA/MLA literatürü
cache pahalı olduğu için var.
ŞERH (E§14-7): Pope 3TB onların rapor ettiği rakam, konfigürasyon satır içinde
yok. Onların figürü olarak sun, yeniden üretilebilir aritmetik olarak değil.

KANIT: M§6.3 formül · Kwon 2023 §3 birebir (800 KB / 1.6 GB) · Shazeer §2.3
birebir · Pope §2 birebir.
~750 kelime · Figür 4: cache bağlamla büyürken model parametresini geçtiği nokta ·
Kod 3: formül vLLM'in sayısını üretiyor (okuyucu alıntıyla karşılaştırabilsin).
-->

## 7. MQA, GQA, MLA

<!--
Omurga bölümü. §6'nın problemine üç cevap, kronolojik. ~1500 kelime.

7.1 MQA (2019) — kafalar tek K/V kümesi paylaşıyor. ASIL ARGÜMAN Shazeer'in
KARŞILAŞTIRMALI tablosu: MQA kaliteden bir şey alıyor ama aynı tasarrufu veren
alternatifler (kafa sayısı ↓, d_k ↓) daha çok alıyor. Kazanma sebebi bu.
  KANIT: §3 tanım birebir · §3.1 oran · Tablo 1 (ln PPL 1.424→1.439, BLEU
  26.7→26.5, beam-4'te MQA 28.5 ile EN YÜKSEK) · Tablo 2 (decoder 46→3.8 µs).

7.2 GQA (2023) — tek parametreli ara değer, G=1 MQA / G=h MHA. Uptraining
(mean-pooling, α=0.05). "8 grup" bir labın eğrisindeki favorable middle ground.
  ZORUNLU ŞERH (E§0 V2): Limitations BİREBİR alıntılanacak — GQA yalnızca
  encoder-decoder'da ölçüldü, sıfırdan eğitimle karşılaştırılmadı, decoder-only
  avantajı BEKLENTİ olarak yazıldı.
  prefix_len paralelliği BİR CÜMLE, uzatma — o numara öbür yazının malı.
  KANIT: Ainslie §2.2 · §2.1 · §3.3 · Llama 2 §2.1.

7.3 MLA (2024) — K/V yerine düşük-ranklı latent. Bölümün en iyi paragrafı:
RoPE ile MATEMATİKSEL UYUMSUZLUK (RoPE matrisi W^Q ile W^UK arasına giriyor,
matris çarpımı değişmeli değil → absorpsiyon bozuluyor) ve decoupled RoPE.
V3'teki d_h^R=64 tam olarak RoPE'a ayrılmış oyuk. §7'yi §8'e bağlıyor.
  ŞERH (E§0 V6): "MLA MHA'dan iyi" DeepSeek'in kendi iddiası, Ek D.1, tek lab.
  KANIT: DS-V2 §2.1.2 · Tablo 1 ("GQA with only 2.25 groups") · abstract
  (%93.3, 5.76×, %42.5) · §2.1.3 BİREBİR · DS-V3 §4.2 konfigürasyon.

Figür 5: dört mekanizmanın KV ayak izi (MHA/GQA/MQA/MLA + eleman formülleri).
Figür 6: 8.6 GiB'e karşı 488 GiB, tek bar çifti, log ölçek YOK.
Kod 4: DeepSeek-V3 aritmetiği — 35.136 vs 1.998.848 eleman/token = 56.9×.
-->

## 8. Positional encoding

<!--
NE: konum bilgisi modele nasıl giriyor — 2017'de toplanarak, bugün döndürerek.
Arkın gerçek şekli "yanlıştı, düzeltildi" DEĞİL. ~1400 kelime.

8.1 Sinüzoidal — Vaswani §3.5 formül BİREBİR. Embedding'e EKLENİR, en altta,
bir kez. Ve kendi ablasyonu: öğrenilen PE ile "nearly identical results";
sinüzoidal seçiminin gerekçesi YALNIZCA ekstrapolasyon.
  ARK (E§0 V5): 2017'de ölçülemeyen eksen bugün her şeyin ayarlandığı eksen.
  Ama "yanlış seçimdi" DENMEYECEK.

8.2 RoPE — beş adım (E§8.2, hepsi birebir):
  1 gereksinim (Denk. 11) → 2 2B rotasyon (Denk. 12-13) → 3 blok-köşegen
  R^d_{Θ,m}, θ_i = 10000^{-2(i-1)/d} (Denk. 14-15) → 4 özdeşlik (Denk. 16) →
  5 ortogonallik, norm korunur.
  VURGU: 10000 sabiti VASWANI'NİN sabiti. RoPE sinüsleri değiştirmedi, hesaba
  giriş biçimini değiştirdi. §8.3'ün düğmesi bu yüzden var.

8.3 Base frekansı ayar düğmesine dönüştü — OLMo 2 (1e4→5e5) · Qwen3 (10k→1M,
ABF+YaRN+DCA) · Gemma 3 (global 1M / yerel 10k). Gözlem: tek model içinde iki
farklı θ; PE modelin global bir özelliği olmaktan çıktı.

8.4 QK-Norm — Henry 2020 köken · ViT-22B §2 BİREBİR (8B'de ıraksama, logitler
50.000'i aşıyor) · Qwen3/OLMo 2/Gemma 3 benimseme.
  BASKI UYARISI: KARARLILIK sütunu, bağlam uzunluğu sütunu DEĞİL.

8.5 NoPE — Kazemnejad 2023 birebir: ALiBi/Rotary/APE uzunluk genellemesi için
uygun değil, NoPE hepsini geçiyor. KAPSAM SINIRI: sıfırdan eğitilmiş göreve özgü
modeller, üretim modelleri RoPE'u bıraksın demek değil. Kanıtladığı: causal maske
zaten konum bilgisi taşıyor. Bölümü bitirecek not bu.

Figür 7: toplamak vs döndürmek — additive'de vektör ucu küreden kayıyor,
RoPE'da küre üzerinde kayıyor, boy sabit. Denklemsiz.
Kod 5: (R_m q)·(R_n k) = q·R_{n-m} k özdeşliği + ‖R_m q‖ = ‖q‖.
-->

## 9. Mixture-of-Experts

<!--
NE: FARKLI BİR BASKI — parametre başına hesap. KV-cache hikâyesine
KARIŞTIRILMAYACAK (E§0 V7). ~900 kelime.

AÇILIŞ: MoE içinde yaşadığı bloktan eski — Ocak 2017, Transformer'dan beş ay
önce, LSTM katmanları arasında.

İFADE TİTİZLİĞİ (E§9.4): "256 expert, 9 aktif" DEĞİL. Doğrusu: MoE katmanında
257 expert (1 shared + 256 routed); token başına 8 routed seçiliyor, shared hep
açık → 9 aktif. 671B toplam / 37B aktif = %5.5.

ŞERH 1 (E§0 V3): shared expert ÇELİŞKİLİ — DeepSeekMoE iki ana fikrinden biri
yapıyor, Qwen3 çıkarıyor ve GEREKÇE VERMİYOR. İki taraf da kesin sunulmayacak.
ŞERH 2 (E§0 V4): "çok sayıda küçük expert kazandı" DENMEYECEK — Switch top-1,
gpt-oss top-4, DeepSeek top-8. Yön tek değil. DeepSeek'in GEREKÇESİ anlatılır,
alanın kararı olarak sunulmaz.

Load balancing bir cümle: routing dengelenmek zorunda, standart yolu kaliteden
alıyor → DeepSeek-V3'ün auxiliary-loss-free katkısı bu yüzden var.

KANIT: Shazeer 2017 abstract · Switch §2.1 birebir + 7.5× · DeepSeekMoE §1, §3.1 ·
DS-V3 §4.2 + §5.2 · Qwen3 Tablo 2 · gpt-oss §2.
Figür 8: expert konfigürasyonları (Switch/DS-V3/Qwen3/gpt-oss), toplam vs aktif,
shared farklı biçimle. Yakınsama olmadığını figür göstermeli.
Kod 6: C(16,2)=120 vs C(64,8)=4.426.165.368 — makalenin rakamını birebir üretiyor.
-->

## 10. Normalizasyon

<!--
NE: önce NE OLDUĞU, sonra hangisinin kazandığı. Okuyucu iki bloğu gözünde
canlandırabilmeli. ~900 kelime.

10.1 Post-LN / Pre-LN — tek fark: normalizasyon ANA YOLDA mı, DALDAKİ KOPYADA mı.
  Post-LN (2017): x_{l+1} = LayerNorm(x_l + Sublayer(x_l))
  Pre-LN (bugün): x_{l+1} = x_l + Sublayer(LayerNorm(x_l))
  İkincisinde ilk katmandan sonuncuya hiçbir şeyin yeniden ölçeklemediği temiz
  bir yol kalıyor. Görünür bedeli: en tepede bir ek LayerNorm.
  KANIT: Vaswani §3.1 birebir · Xiong Tablo 1 altı adım (E§10.1) · Xiong
  abstract birebir (Post-LN'de çıkışa yakın gradyanlar büyük → warm-up şart).
  ÇERÇEVE: Pre-LN KALİTEYLE kazanmadı. Boyutla ayarlama maliyeti büyüyen bir
  hiperparametreyi (warm-up) ortadan kaldırdığı için kazandı.

10.2 RMSNorm — Zhang & Sennrich abstract birebir (re-centering gereksiz,
%7-64 hız). Qwen3/Gemma 3/OLMo 2/gpt-oss.

10.3 2024-25 varyantları — OLMo 2 reordered norm birebir · Gemma 3 "post-norm
and pre-norm with RMSNorm" (aynı blokta ikisi de).
  ŞERH (E§10.3): OLMo 2'nin ikinci denklemi basılı hâliyle MLP(x) diyor, blok
  yapısı MLP(h) gerektiriyor — dizgi hatası. Düzeltilmiş hâli ALINTI diye
  sunulmayacak.

10.4 Attention sinks — V8'in kapandığı yer, YAZININ KAPANIŞ HAMLESİ.
  StreamingLLM §3.1 birebir: sink'ler softmax'ın 1'e toplanma zorunluluğu
  yüzünden var. 4 token yetiyor, 22.2× hızlanma.
  gpt-oss §2 birebir: softmax PAYDASINDA öğrenilen bias, "pay no attention to
  any tokens."
  → Yayındaki yazının maskeleme argümanını taşıyan kısıt ile sink'leri var eden
  kısıt AYNI kısıt. gpt-oss oradan çıkıyor. Tek kısıt, iki sonuç, sekiz yıl
  arayla.

Figür 9 (ZORUNLU): Post-LN vs Pre-LN yan yana. Residual yolu kalın; Post-LN'de
yolun üstünde norm kutusu, Pre-LN'de yol temiz + tepede final LayerNorm.
Kod yok.
-->

## 11. FFN ve aktivasyon

<!--
NE: bloğun en az konuşulan ama parametrelerin çoğunu tutan yarısı. Ve yazının
DÜRÜSTLÜK VALFİ: her şeyin gerekçesi yok. ~450 kelime.

KANIT: Vaswani §3.3 (iki lineer + ReLU, d_ff=2048) · Shazeer 2020 SwiGLU tanımı ·
§2 birebir (ÜÇ matris, d_ff 2/3 ile çarpılıyor — modern d_ff oranlarının tuhaf
görünme sebebi) · Tablo 1 (ReLU 1.677 → SwiGLU 1.636, parametre+hesap eşit) ·
Qwen3/gpt-oss/OLMo 2 benimseme.

BİREBİR ALINTILANACAK: "We offer no explanation as to why these architectures
seem to work; we attribute their success, as all else, to divine benevolence."

NEDEN ÖNEMLİ: bu yazıdaki her bileşenin adı konmuş bir baskısı var. Bunun bir
tablosu ve bir omuz silkmesi var — ve her yere girdi. Bölüm bunu açıkça
söyleyecek: modern bloktaki her şey açıklanmış değil, alan da biliyor.

Figür yok, kod yok. Tablo yeterli.
-->

## 12. Encoder bloğu

<!--
NE: 2017 temeli, bir kez, Post-LN etiketiyle. Kısa. ~350 kelime.
KANIT: Vaswani §3.1 birebir (N=6, iki alt katman, residual + LayerNorm).
İŞ BÖLÜMÜ (E§12.1): bu bölüm Post-LN'i GÖSTERİR, §10 AÇIKLAR. Mekanizma burada
tekrarlanmayacak; ileri referans bir yan cümle.
Figür 10: encoder bloğu 2017, Post-LN açıkça etiketli. Altyazı: bugün shipped
hiçbir modelde bu yerleşim yok, sebebi §10'da.
-->

## 13. Decoder bloğu

<!--
NE: her şey monte. Sonra beş model yan yana, ve asıl argüman: BİLEŞEN
LİSTESİNDE YAKINSAMA YOK; BASKILAR KÜMESİNDE VAR. ~800 kelime.

KANIT: E§12.2 karşılaştırma tablosu — DeepSeek-V3 / Qwen3-235B / Gemma 3 /
gpt-oss-120b / OLMo 2 13B, her hücre kendi raporundan VERIFIED. Boşluklar boşluk
olarak bırakılacak, TAHMİN EDİLMEYECEK.

KAPANIŞ DÜRÜSTLÜK NOTU (E§0 V6): bu yazının 2024-25 sayılarının neredeyse tamamı
TEKNİK RAPOR, kontrollü deney değil. Model kartları ne shipped edildiğini
söylüyor, neyin izole edildiğini değil. Ablasyon olan yerlerde tek labın kendi
reçetesi üzerinde. T5 §3.2 ayarında eşit-bütçeli bir mimari karşılaştırması YOK
— çünkü kimse yapmadı. Bir kez, açıkça.

Figür 11: bugünün decoder bloğu, tam monte (RMSNorm → GQA/MLA + RoPE → ⊕ →
RMSNorm → SwiGLU/MoE → ⊕, ×L), her kutunun yanında § numarası. Figür 10 ile
AYNI ÖLÇEKTE olmalı ki fark görünsün.
Karşılaştırma tablosu markdown olarak — figür değil, mobilde daha iyi davranır.
-->
