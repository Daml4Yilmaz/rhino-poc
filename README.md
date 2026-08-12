# rhino-poc

Telefonla çekilmiş video → gerçek ölçekli 3D yüz modeli + 6 otomatik rinoplasti ölçümü + kumpasla karşılaştırmalı hata raporu.

**Yöntem: fotogrametri.** Video kareye ayrılır ("ekran görüntüsü"), 3D model bu karelerden COLMAP SfM + MVS ile üretilir. Marker (ArUco/ChArUco) **kullanılmaz**.

Fotogrametri şekli verir, **milimetreyi vermez** — aynı kareler 50 mm'lik burunla da 500 mm'lik burunla da tutarlıdır; mutlak boyut görüntülerin içinde yoktur. Marker olmadığı için mm çarpanı telefonun kendi metrik takibinden (ARKit VIO + LiDAR) okunur. Bu veri modeli **üretmez**, sadece "1 COLMAP birimi kaç mm" sayısını verir. Ayrıntı: `poc/pipeline/scale.py`.

Tek komut: `poc process <yakalama> --out vaka_001`

## Çekim protokolü — iki kişi, iki geçiş

Denek sabit oturur, **ikinci bir kişi** telefonu tutup yayı yürür.

**Hazırlık**
- [ ] Denek oturur, sırt dayalı, ayaklar yerde (başlıklı koltuk daha iyi)
- [ ] Saç tamamen bone altında, kulaklar açıkta, gözlük/küpe yok
- [ ] Gözler **3 m ötede işaretli bir noktada** sabit — telefonu takip etmek başı döndürür
- [ ] **İfadesiz yüz, ağız kapalı, konuşma yok** (SfM hareketli sahnede çöker)
- [ ] Mat cilt (parlıyorsa şeffaf pudra) — parlamalar kamerayla birlikte hareket eder ve geometri sanılır

**Ortam (önceki sürümlerden değişti)**
- [ ] Arka plan **sabit ve DOKULU** — düz/boş fon değil. ARKit'in takibi ortam özniteliği ister; denek sabit olduğu için oda ve yüz tek bir katı sahne oluşturur ve arka plan SfM'e de yardım eder. Fon geometrisi sonradan maskelenir (ucuz); bozuk takip geri gelmez.
- [ ] Karede hareket eden hiçbir şey yok (başka kişi, ekran, pencere)
- [ ] Dağınık, eşit ve **sabit** ışık; bantlanma (flicker) olmadığını doğrula

**Telefon**
- [ ] Arka ana kamera, **zoom tam 1.0×** ve hiç dokunulmadan (lens değişimi odak uzaklığını değiştirir, tek-kamera varsayımını sessizce bozar)
- [ ] **Otofokus:** Stray Scanner AF kilidi sunmuyor — gerek de yok. ARKit her karenin `fx, fy, cx, cy` değerini `odometry.csv`'ye yazar; `sfm.py` bunları doğrudan COLMAP veritabanına yazıp bundle adjustment'ın odak uzaklığını değiştirmesini engeller. Bu **kilitten daha iyidir**: kilit sabitliği varsayar, bu ise ölçülmüş gerçek değeri kullanır. Test kaydında odak %3.3 oynamıştı ve model bunu sorunsuz taşıdı.
- [ ] Yine de AF avlanmasını azalt: mesafeyi sabit tut (AF derinlik değişince arar), kaydı yüz zaten çerçevedeyken başlat ve 2 sn bekle, yüksek kontrastlı fon geçişlerinde süpürme.
- [ ] Kayıt uygulaması: **Stray Scanner** veya Record3D. **iPhone'un kendi Kamera uygulaması yetmez** — o yalnızca pikselleri kaydeder, ARKit kamera pozunu ve LiDAR derinliğini dosyaya yazmaz. Ölçek videonun içinde değil, çekim anındaki hareket takibindedir; kayıt bittikten sonra geri gelmez. Stok Kamera 4K (8.3 MP) ile daha iyi doku verir ama **milimetre vermez** → yalnızca açılar ve Goode oranı hesaplanabilir, G1 karşılanamaz.
  - Stray Scanner LiDAR'lı cihaz ister (iPhone 12 Pro ve sonrası; 14 Pro Max uygun). LiDAR yoksa Record3D yalnız ARKit poziyle çalışır, `s_depth` çapraz kontrolü devre dışı kalır.
  - 1920×1440 yeterli mi? 65 cm'den yüz kısa kenarın ~%60'ını kaplar → **~4 piksel/mm**. 2 mm hedefi için fazlasıyla yeterli; 4K'nın üstünlüğü geometride değil dokudadır (WP7).

**Geçiş 1 — göz hizası, kulaktan kulağa**
- [ ] Sağ kulaktan sol kulağa (ya da tersi — `case.json`'a yazılır), 180°
- [ ] Kamera deneğin **göz hizasında**, lens buruna dönük
- [ ] Mesafe **60–70 cm**, sabit
- [ ] **25–35 sn**, topuk-parmak yürüyüş, iki el telefonda, dirsekler gövdeye dayalı
- [ ] Duraklama zararsız; ani hızlanma zararlı (kare bulanır, takip sarsılır)

**Geçiş 2 — alçak açı, yukarı eğik (burun tabanı)**
- [ ] Kamera deneğin göğüs/çene hizasına iner, **~30° yukarı eğik** — iki burun deliği, kolumella ve alar oluk net görünmeli
- [ ] Aynı yay, **15–25 sn**, aynı mesafe
- [ ] 6 ölçümün dördü bu geçişin geometrisine dayanır. Tekrar çekilecek tek geçiş varsa budur.

**Geçişler arası:** deneği gevşet, sonra yeniden yerleştir ve işarete odaklat. İki kısa duruş, tek uzun duruştan iyidir. **Kaydı geçişler arasında durdurma** — tek oturum tek metrik çerçeve demektir, ölçek bunun üstüne kurulur.

## Kurulum

**Yerel (macOS, M-serisi) — GPU gerektirmeyen her şey burada koşar**
```bash
brew install colmap
uv venv && uv pip install -e .
poc process vaka_001_stray/ --out vaka_001 --no-sift-gpu --until sfm
```

Not: macOS'ta COLMAP CUDA'sız derlenir — beklenen. `mvs` adımı yerelde koşmaz, `--until sfm` ile durdur.

**Colab (sadece GPU gereken adımlar: `mvs`, ileride `gsplat`)**
1. Colab'da `File > Open notebook > GitHub` → `Daml4Yilmaz/rhino-poc` → `colab_setup.ipynb`. Notebook kodu repodan çeker; Drive'a repo yüklemek gerekmez, Drive yalnızca veri ve çıktı için.
2. Hücreleri sırayla koş. Hesaplama `/content`'te yapılır (Drive FUSE üzerinde COLMAP MVS on binlerce küçük dosya yazdığı için çok yavaştır); sonuçlar son hücrede `MyDrive/rhino-poc-out/<vaka>/` altına kopyalanır.
3. Colab'da OpenCV **kurma** — hazır geleni kullan. Üstüne `opencv-contrib-python` kurulunca `cv2` yarım yükleniyor (`SIFT_create` çalışır, `CascadeClassifier` kaybolur). ArUco bırakıldığı için contrib gereksiz.

Gerekçe: sadece COLMAP `patch_match_stereo` ve gaussian splatting CUDA ister. Geri kalan her şey M4 Pro'da koşar; Colab'ın ücretsiz kotası bir kez oturum ortasında tükendiği için Colab'a tek adım devredilir. Ayrıntı: `PLAN.md` §4.

## Pipeline durumu

| Adım | Modül | Durum |
|---|---|---|
| (a) Kare çıkarma + bulanıklık eleme | `pipeline/frames.py` | hazır (kaynak kare indeksi korunur) |
| ARKit yakalama okuma (Stray/Record3D) | `pipeline/arkit.py` | hazır |
| (b) COLMAP SfM | `pipeline/sfm.py` | hazır |
| (c1) COLMAP MVS + Poisson | `pipeline/mvs.py` | hazır (CUDA gerekir → Colab) |
| (c2) Gaussian splatting hattı | `pipeline/gsplat.py` | stub |
| (d) Saç/fon maskeleme | `pipeline/masking.py` | stub |
| (e) Markersiz ölçek (ARKit poz + LiDAR) | `pipeline/scale.py` | hazır |
| (f) FLAME kaydı → landmarks.json | `pipeline/flame_fit.py` | stub (FLAME hesabını aç: flame.is.tue.mpg.de) |
| (g) 6 ölçüm | `pipeline/measure.py` | hazır (landmarks.json bekler) |
| (h) Renkli GLB export | `pipeline/export.py` | hazır (vertex rengi; UV doku sonra) |
| Hata raporu | `report/compare.py` | hazır |

Renk bedava gelir: COLMAP `stereo_fusion` her noktaya karelerden okunan RGB'yi yazar, Poisson mesh'e taşır, `export.py` GLB'ye aktarır. Bu **vertex rengi**, UV dokusu değil — çözünürlüğü mesh yoğunluğuyla sınırlı. Ben/kırışıklık gibi ince detay için karelerden doku pişirmek gerekir (WP7); ölçümleri etkilemez.

## Kalite kapıları — kodun reddettiği durumlar

Boşa geçen 40 dakikalık koşuları önlemek için pipeline erken durur:

| Kontrol | Nerede | Eşik |
|---|---|---|
| Açısal kapsama | `arkit.py` | <20° → **hata** (paralaks yok, SfM çözemez); <120° → uyarı |
| VIO sıçraması | `arkit.py` | karelerin %2'sinden fazlasında >15 cm → hata |
| Odak uzaklığı kayması | `arkit.py` | >%1 → uyarı (**AE/AF kilitlenmemiş**) |
| Kare sayısı | `arkit.py` | <120 → hata; <700 → uyarı |
| SfM kayıt oranı | `sfm.py` | <%50 veya <20 kare → **hata**, MVS'e geçilmez |
| Ölçek uyumu | `scale.py` | `agreement_pct` >%1.5 → `scale_verified=false` |

Yol uzunluğu bilerek ölçüt **değil**: bir yüzün etrafında 72°'lik kısa bir yay, iki metrelik düz kaydırmadan çok daha fazla bilgi taşır. Belirleyici olan açısal kapsama.

COLMAP birden fazla alt-model üretirse (`sparse/0`, `sparse/1`, …) `sfm.py` **en çok kare kaydedileni** seçer ve bölünmeyi bildirir.

## Ölçek doğrulaması

`scale.json` iki **bağımsız** tahmin ve aralarındaki uyumu yazar:

- `s_pose_m_per_unit` — COLMAP ve ARKit kamera yörüngeleri arasındaki benzerlik ölçeği (ikili mesafe oranlarının medyanı; VIO dönme kaymasına karşı dayanıklı)
- `s_depth_m_per_unit` — LiDAR derinliği / COLMAP derinliği oranı, yüksek güvenli piksellerde
- `agreement_pct` — ikisi arasındaki fark. **%1.5 üstü → `scale_verified=false`**, vaka incelenmeden G1 tablosuna girmez.

`ipd_mm` (55–70 mm) ve `bbox_mm` (150–320 mm) yalnızca **makullük kontrolüdür**, ölçeği asla belirlemez.

## Kullanım

```bash
poc process vaka_001_stray/ --out vaka_001
poc process vaka_001_stray/ --out vaka_001 --until sfm      # erken dur
poc process vaka_001_stray/ --out vaka_001 --resume         # kesintiden devam
poc scale   vaka_001_stray/ --out vaka_001                  # sadece ölçek
poc measure vaka_001/landmarks.json --out vaka_001/measurements.json
python -m poc.report.compare calipers.csv vaka_001 vaka_002
```

Colab'da `poc` yerine **`python -m poc.cli`** kullan — kurulan komut betiği her zaman PATH'e girmiyor.

Kumpas değerleri: `data/calipers_template.csv`'yi kopyala, doldur.

### Test modu — düz video (ARKit verisi olmadan)

`poc process` argümanı `.mp4/.mov` ise ölçek adımı atlanır ve yalnızca fotogrametri hattı koşar:

```bash
poc process test.mov --out test_out --n-frames 150
```

Çıktı `model_unitless.glb` adıyla yazılır ve koşu başında birimsiz olduğunu bildiren bir uyarı basar. **Yalnızca hattın ayakta olduğunu doğrular** — açı ve Goode oranı anlamlı, mm cinsinden uzunluk/genişlik/sapma üretilemez. G1 için `Stray Scanner` kaydı şart.

### Çekim uygunluk teşhisi

`colab_setup.ipynb` **bölüm 6**: videoyu pipeline'a sokmadan ~2 dakikada ölçer ve `UYGUN` / `RET` der. Ölçtüğü üç şey: kare başına SIFT özelliği (>1500 iyi), ardışık kareler arası iç-eşleşme (>50 iyi), ve **kayma hızı px/sn** (>60 iyi, <25 paralaks yok).

Kayma hızı zamana normalize edilir; ham piksel kayması kare aralığına bağlı olduğu için aynı sahne 37 fps ardışık ile 60 fps'te 34 kare atlayarak ölçüldüğünde 20 kat farklı okunur.

## İlk başarı kontrolü

1. `arkit` çıktısı: açısal kapsama >120°, sıçrama yok, LiDAR "var", odak kayması uyarısı **yok**.
2. `sfm` karelerin >%50'sini kaydediyor, tek parça (alt-model bölünmesi uyarısı yok).
3. `scale.json` → `agreement_pct` < 1.5 ve `scale_verified: true`.
4. `model.glb` renkli, bbox ~200–250 mm; aynı denek 3 kez çekildiğinde ölçek %1 içinde tekrar ediyor.
