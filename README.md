# rhino-poc

Telefonla çekilmiş video → gerçek ölçekli 3D yüz modeli + 6 otomatik rinoplasti ölçümü + kumpasla karşılaştırmalı hata raporu.

Tek komut: `poc process video.mp4 --out vaka_001 --marker-mm 50`

## Kurulum (Google Colab)

1. Bu klasörü Drive'a `MyDrive/rhino-poc` olarak yükle.
2. `colab_setup.ipynb`'yi Colab'da aç (GPU runtime), hücreleri sırayla koş.
3. İlk oturumda COLMAP kurulumu uzun sürebilir; sonucu Drive'a cache'lenir.

Colab notları: oturum diski geçicidir — video ve çıktılar Drive'da tutulur. Docker yok; ortam notebook ile kurulur. Rakamlar ciddiye binince aynı repo değişiklik olmadan kiralık GPU'ya (RunPod vb.) taşınabilir.

## Çekim protokolü (kontrol listesi)

- [ ] 1080p / 60 fps, telefonun kendi kamerası
- [ ] Çekimden önce ekrana basılı tut → **AE/AF kilidi** (sarı kilit simgesi)
- [ ] Denek oturur, saç bone/bant altında, **yüz ifadesiz, ağız kapalı**, sabit noktaya bakar
- [ ] Alına basılı ArUco (mat kağıt, düz yapıştırılmış, kıvrımsız)
- [ ] Kol mesafesinden, 20-30 saniyede kulaktan kulağa **yavaş** tur
- [ ] Burun tabanı için hafif alttan ikinci kısa geçiş
- [ ] Düz fon, dağınık eşit ışık, mat cilt (gerekirse pudra)

ArUco baskısı: `python scripts/make_aruco.py --mm 50` → %100 ölçekle bas → **kenarı cetvelle ölç** → ölçtüğün değeri `--marker-mm`'e ver.

## Pipeline durumu

| Adım | Modül | Durum |
|---|---|---|
| (a) Kare çıkarma + bulanıklık eleme | `pipeline/frames.py` | hazır |
| (b) COLMAP SfM | `pipeline/sfm.py` | hazır |
| (c1) COLMAP MVS + Poisson | `pipeline/mvs.py` | hazır (CUDA'lı COLMAP gerekir) |
| (c2) Gaussian splatting hattı | `pipeline/gsplat.py` | hafta 2 stub |
| (d) Saç/fon maskeleme | `pipeline/masking.py` | hafta 2 stub |
| (e) ArUco ölçek | `pipeline/scale.py` | hazır |
| (f) FLAME kaydı → landmarks.json | `pipeline/flame_fit.py` | hafta 2 stub (FLAME hesabını şimdiden aç: flame.is.tue.mpg.de) |
| (g) 6 ölçüm | `pipeline/measure.py` | hazır (landmarks.json bekler) |
| (h) GLB export | `pipeline/export.py` | hazır (doku hafta 2-3) |
| Hata raporu | `report/compare.py` | hazır |
| Web demo (FastAPI + Three.js) | — | hafta 4 |

## Kullanım

```bash
poc process vaka_001.mp4 --out vaka_001 --marker-mm 49.5
poc process vaka_001.mp4 --out vaka_001 --until sfm     # erken dur
poc measure vaka_001/landmarks.json --out vaka_001/measurements.json
python -m poc.report.compare calipers.csv vaka_001 vaka_002
```

Kumpas değerleri: `data/calipers_template.csv`'yi kopyala, doldur.

## Hafta 1 başarı kontrolü

1. `sfm` ≥150 kare kaydediyor, `mvs` delik(siz)e yakın yüz yüzeyi veriyor.
2. `scale.json` → `side_spread_pct` < 5.
3. `model.glb`'de gözbebeği arası ~60-70 mm (makullük), kafa bbox ~200-250 mm.
