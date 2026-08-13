"""COLMAP'i canli ilerleme ciktisiyla calistiran ortak yardimci.

Sorun: COLMAP ilerlemeyi stdout'a yazar; biz onu log dosyasina yonlendirince
notebook'ta saatlerce hicbir sey gorunmuyordu — hangi adimda, kacinci
goruntude olundugu bilinmiyordu.

Cozum: cikti satir satir okunur, TAMAMI log dosyasina yazilir, icinden
ilerleme satirlari ayiklanip periyodik olarak ozetlenir. Colab'da stdout bir
TTY olmadigi icin `\\r` ile satir uzerine yazmak guvenilir degil; bunun
yerine belirli araliklarla YENI satir basiyoruz — hem okunur hem de
kaydirilabilir bir gecmis birakir.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

# COLMAP'in farkli asamalari ilerlemeyi farkli bicimlerde yazar.
_PATTERNS: list[re.Pattern] = [
    re.compile(r"Processing view (\d+) / (\d+)"),
    re.compile(r"Undistorting image \[(\d+)/(\d+)\]"),
    re.compile(r"Fusing image \[(\d+)/(\d+)\]"),
    re.compile(r"Processed file \[(\d+)/(\d+)\]"),
    re.compile(r"Matching block \[(\d+)/(\d+)"),
]
_REGISTER = re.compile(r"num_reg_frames=(\d+)")
_MATCHED = re.compile(r"feature_matching\.cc:\d+\] in ")
_ITER = re.compile(r"(Sweep|Iteration) (\d+): ([\d.]+)s")


def _fmt(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} sn"
    if sec < 5400:
        return f"{sec/60:.0f} dk"
    return f"{sec/3600:.1f} sa"


def run_logged(cmd: list[str], log_file: Path, label: str,
               every: float = 30.0,
               watch: tuple[Path, str, int] | None = None) -> None:
    """Komutu kos; ciktiyi log'a yaz, ilerlemeyi ekrana ozetle.

    `every` saniyede bir durum satiri basar — her goruntude basmak Colab
    ciktisini bogar, hic basmamak da simdiki durumu yaratir.

    `watch=(klasor, uzanti, toplam)` verilirse ilerleme LOG BICIMINDEN DEGIL
    diskte olusan dosya sayisindan okunur. patch_match icin bunu tercih
    ediyoruz: COLMAP surumleri ilerleme satirini farkli yazar (ya da hic
    yazmaz), ama derinlik haritasi dosyalari her surumde ayni yere duser.
    """
    t0 = time.time()
    last = 0.0
    cur = tot = 0
    n_reg = 0
    iter_times: list[float] = []
    print(f"[{label}] basladi", flush=True)

    with open(log_file, "a") as lf:
        lf.write("\n$ " + " ".join(cmd) + "\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)

            for pat in _PATTERNS:
                m = pat.search(line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    break
            else:
                m = _REGISTER.search(line)
                if m:
                    n_reg = int(m.group(1))
                elif _MATCHED.search(line):
                    n_reg = n_reg + 1 if n_reg else 1
                m = _ITER.search(line)
                if m:
                    iter_times.append(float(m.group(3)))

            now = time.time()
            if now - last < every:
                continue
            last = now
            el = now - t0

            if watch is not None:
                wdir, suf, wtot = watch
                cur = len(list(wdir.glob(f"*{suf}"))) if wdir.is_dir() else 0
                tot = wtot

            if tot:
                pct = 100.0 * cur / tot
                eta = el / cur * (tot - cur) if cur else 0.0
                extra = ""
                if iter_times:
                    extra = f" · {sum(iter_times[-20:])/len(iter_times[-20:]):.1f} sn/iter"
                print(f"  [{label}] {cur}/{tot}  %{pct:.0f}  "
                      f"gecen {_fmt(el)}  kalan ~{_fmt(eta)}{extra}", flush=True)
            elif n_reg:
                unit = "cift" if label == "eslestirme" else "goruntu"
                print(f"  [{label}] {n_reg} {unit} islendi  "
                      f"gecen {_fmt(el)}", flush=True)
            else:
                print(f"  [{label}] calisiyor…  gecen {_fmt(el)}", flush=True)

        p.wait()
        lf.flush()

    dt = time.time() - t0
    if p.returncode != 0:
        print(f"[{label}] HATA (kod {p.returncode}) — {_fmt(dt)} sonra. "
              f"Ayrinti: {log_file}", flush=True)
        raise subprocess.CalledProcessError(p.returncode, cmd)
    print(f"[{label}] bitti — {_fmt(dt)}", flush=True)


class Ticker:
    """Python tarafindaki uzun donguler icin basit ilerleme yazici.

    Colab'da `\\r` guvenilir olmadigi icin araliklarla yeni satir basar.
    """

    def __init__(self, label: str, total: int, every: float = 20.0):
        self.label, self.total, self.every = label, total, every
        self.t0 = time.time()
        self.last = self.t0

    def step(self, i: int) -> None:
        now = time.time()
        if now - self.last < self.every:
            return
        self.last = now
        el = now - self.t0
        eta = el / max(i, 1) * (self.total - i)
        print(f"  [{self.label}] {i}/{self.total}  %{100.0*i/max(self.total,1):.0f}  "
              f"gecen {_fmt(el)}  kalan ~{_fmt(eta)}", flush=True)

    def done(self) -> None:
        print(f"  [{self.label}] {self.total}/{self.total} — "
              f"{_fmt(time.time() - self.t0)}", flush=True)
