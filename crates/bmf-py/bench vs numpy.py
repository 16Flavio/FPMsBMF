from __future__ import annotations

import gc
import statistics
import time
from typing import Callable, List, Tuple

import numpy as np

import FPMsBMF as bmf

SHAPES: List[Tuple[int, int]] = [(302, 917), (1608, 517), (3000, 3000)]
DENSITY = 0.3
REPEATS = 7


def timeit(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    fn()
    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    return statistics.median(times)


def fmt_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:7.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.2f} ms"
    return f"{seconds:7.3f} s "


def fmt_bytes(n: int) -> str:
    for unit in ("o", "Ko", "Mo", "Go"):
        if n < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} To"


def rand(rows: int, cols: int, density: float = DENSITY, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols)) < density


def row(label: str, t_np: float, t_bmf: float) -> None:
    ratio = t_np / t_bmf if t_bmf > 0 else float("inf")
    marker = "x" if ratio >= 1 else "/"
    value = ratio if ratio >= 1 else 1 / ratio
    print(f"  {label:<22} {fmt_time(t_np)}  {fmt_time(t_bmf)}   {marker}{value:6.1f}")


def bench_memory() -> None:
    print("\n" + "=" * 72)
    print("EMPREINTE MEMOIRE")
    print("=" * 72)
    print(f"  {'dimensions':<22} {'numpy':>12} {'FPMsBMF':>12}   {'rapport':>8}")

    for rows, cols in SHAPES:
        n_np = rows * cols
        words_per_row = (cols + 63) // 64
        n_bmf = rows * words_per_row * 8
        print(
            f"  {f'{rows}x{cols}':<22} {fmt_bytes(n_np):>12} {fmt_bytes(n_bmf):>12}"
            f"   x{n_np / n_bmf:6.2f}"
        )

    print(
        "\n  Le rapport est inferieur a 8 quand cols n'est pas multiple de 64 :\n"
        "  le remplissage de fin de ligne est stocke mais inutilise."
    )

def bench_conversion() -> None:
    print("\n" + "=" * 72)
    print("COUT DE CONVERSION")
    print("=" * 72)
    print("  Ce cout est paye une fois ; tout l'interet du type est de le")
    print("  mutualiser sur de nombreuses operations.\n")
    print(f"  {'dimensions':<22} {'from_numpy':>12} {'to_numpy':>12}   {'ns/element':>10}")

    for rows, cols in SHAPES:
        a = rand(rows, cols, seed=1)
        t_pack = timeit(lambda: bmf.BitMatrix.from_numpy(a), repeats=3)
        m = bmf.BitMatrix.from_numpy(a)
        t_unpack = timeit(lambda: m.to_numpy(), repeats=3)
        per_elem = t_pack / (rows * cols) * 1e9
        print(
            f"  {f'{rows}x{cols}':<22} {fmt_time(t_pack):>12} {fmt_time(t_unpack):>12}"
            f"   {per_elem:9.2f}"
        )



def bench_elementwise() -> None:
    print("\n" + "=" * 72)
    print("OPERATIONS ELEMENT PAR ELEMENT")
    print("=" * 72)
    print("  numpy est vectorise (AVX2 sur des octets) : l'ecart vient du")
    print("  trafic memoire, huit fois moindre, pas du nombre d'instructions.")

    for rows, cols in SHAPES:
        a, b = rand(rows, cols, seed=2), rand(rows, cols, seed=3)
        ma, mb = bmf.BitMatrix.from_numpy(a), bmf.BitMatrix.from_numpy(b)

        print(f"\n  --- {rows}x{cols} ---")
        print(f"  {'operation':<22} {'numpy':>10} {'FPMsBMF':>10}   {'rapport':>8}")

        # Correction verifiee avant toute mesure.
        assert np.array_equal((ma | mb).to_numpy(), a | b)
        assert np.array_equal((ma & mb).to_numpy(), a & b)
        assert np.array_equal((ma ^ mb).to_numpy(), a ^ b)
        assert np.array_equal((~ma).to_numpy(), ~a)

        row("or  (A | B)", timeit(lambda: a | b), timeit(lambda: ma | mb))
        row("and (A & B)", timeit(lambda: a & b), timeit(lambda: ma & mb))
        row("xor (A ^ B)", timeit(lambda: a ^ b), timeit(lambda: ma ^ mb))
        row("not (~A)", timeit(lambda: ~a), timeit(lambda: ~ma))

        out = np.empty_like(a)
        acc = ma.copy()
        row(
            "or en place (|=)",
            timeit(lambda: np.logical_or(a, b, out=out)),
            timeit(lambda: acc.__ior__(mb)),
        )



def bench_reductions() -> None:
    print("\n" + "=" * 72)
    print("REDUCTIONS")
    print("=" * 72)
    print("  count_ones exploite l'instruction POPCNT : un mot de 64 bits")
    print("  compte en un cycle, la ou numpy additionne 64 octets.")

    for rows, cols in SHAPES:
        a, b = rand(rows, cols, seed=4), rand(rows, cols, seed=5)
        ma, mb = bmf.BitMatrix.from_numpy(a), bmf.BitMatrix.from_numpy(b)

        assert ma.count_ones() == int(np.count_nonzero(a))
        assert ma.hamming(mb) == int(np.count_nonzero(a != b))
        assert ma.count_andnot(mb) == int(np.count_nonzero(a & ~b))

        print(f"\n  --- {rows}x{cols} ---")
        print(f"  {'operation':<22} {'numpy':>10} {'FPMsBMF':>10}   {'rapport':>8}")

        row(
            "count_ones",
            timeit(lambda: np.count_nonzero(a)),
            timeit(lambda: ma.count_ones()),
        )
        row(
            "hamming",
            timeit(lambda: np.count_nonzero(a != b)),
            timeit(lambda: ma.hamming(mb)),
        )
        row(
            "count_andnot",
            timeit(lambda: np.count_nonzero(a & ~b)),
            timeit(lambda: ma.count_andnot(mb)),
        )


def bench_product() -> None:
    print("\n" + "=" * 72)
    print("PRODUIT BOOLEEN  (forme BMF : W de m x r, H de r x n)")
    print("=" * 72)
    print("  Ici l'ecart n'est pas du a la representation mais a l'ALGORITHME.")
    print("  FPMsBMF accumule les lignes de H designees par les 1 de W, en")
    print("  O(nnz(W) * n/64). numpy passe par un produit matriciel dense en")
    print("  O(m*n*r), quelle que soit la voie choisie.\n")
    print("  Trois voies numpy sont mesurees : le produit BLAS en float32 est")
    print("  le plus rapide malgre la conversion, ce qui est contre-intuitif")
    print("  et merite d'etre mesure plutot que suppose.")

    m, n = 1608, 517

    for r in (5, 10, 20):
        w_np, h_np = rand(m, r, seed=6), rand(r, n, seed=7)
        w, h = bmf.BitMatrix.from_numpy(w_np), bmf.BitMatrix.from_numpy(h_np)

        expected = (w_np.astype(np.uint8) @ h_np.astype(np.uint8)) > 0
        assert np.array_equal((w @ h).to_numpy(), expected)

        w_f = w_np.astype(np.float32)
        h_f = h_np.astype(np.float32)
        assert np.array_equal((w_f @ h_f) > 0, expected)

        print(f"\n  --- m={m}, r={r}, n={n} ---")
        print(f"  {'methode':<34} {'temps':>10}")

        t_u8 = timeit(
            lambda: (w_np.astype(np.uint8) @ h_np.astype(np.uint8)) > 0, repeats=3
        )
        t_f32 = timeit(lambda: (w_f @ h_f) > 0, repeats=3)
        t_f32_pre = timeit(
            lambda: (w_np.astype(np.float32) @ h_np.astype(np.float32)) > 0, repeats=3
        )
        t_bmf = timeit(lambda: w @ h)

        print(f"  {'numpy uint8 @ uint8':<34} {fmt_time(t_u8):>10}")
        print(f"  {'numpy float32 (deja converti, BLAS)':<34} {fmt_time(t_f32):>10}")
        print(f"  {'numpy float32 (conversion incluse)':<34} {fmt_time(t_f32_pre):>10}")
        print(f"  {'FPMsBMF  W @ H':<34} {fmt_time(t_bmf):>10}")
        print(f"  {'':34} rapport contre le meilleur numpy : "
              f"x{min(t_u8, t_f32, t_f32_pre) / t_bmf:.1f}")



def bench_breakeven() -> None:
    print("\n" + "=" * 72)
    print("SEUIL DE RENTABILITE")
    print("=" * 72)
    print("  A partir de combien d'operations la conversion est-elle amortie ?")
    print("  C'est la seule question qui compte pour decider d'utiliser le type.\n")

    for rows, cols in SHAPES:
        a, b = rand(rows, cols, seed=8), rand(rows, cols, seed=9)

        t_conv = timeit(lambda: bmf.BitMatrix.from_numpy(a), repeats=3) * 2
        ma, mb = bmf.BitMatrix.from_numpy(a), bmf.BitMatrix.from_numpy(b)

        t_np = timeit(lambda: np.count_nonzero(a != b))
        t_bmf = timeit(lambda: ma.hamming(mb))

        gain = t_np - t_bmf
        if gain <= 0:
            print(f"  {rows}x{cols} : aucun gain sur cette operation.")
            continue

        n_ops = t_conv / gain
        print(
            f"  {f'{rows}x{cols}':<14} conversion {fmt_time(t_conv)}"
            f"  gain/appel {fmt_time(gain)}"
            f"  seuil : {n_ops:6.0f} appels"
        )

    print(
        "\n  En dessous de ce seuil, rester en numpy. Au-dessus — le regime de\n"
        "  toute campagne experimentale — le packing unique s'impose."
    )


def bench_solvers() -> None:
    print("\n" + "=" * 72)
    print("SOLVEURS  (aucun equivalent numpy)")
    print("=" * 72)
    print("  numpy ne fournit pas de solveur : ces chiffres situent le cout")
    print("  d'une resolution, pas un rapport.")

    m, n = 1608, 517
    x_np = rand(m, n, seed=10)
    x = bmf.BitMatrix.from_numpy(x_np)

    print(f"\n  --- BoolLS, une colonne, m={m} ---")
    print(f"  {'methode':<16} {'temps':>10}")
    w = bmf.BitMatrix.from_numpy(rand(m, 10, seed=11))
    solver = bmf.BoolLs(w)
    col = x_np[:, 0]
    for method in ("naive", "zeta", "greedy", "greedy-ls"):
        t = timeit(lambda mth=method: solver.solve(col, method=mth), repeats=3)
        print(f"  {method:<16} {fmt_time(t):>10}")

    print(f"\n  --- Mise a jour complete de H ({n} colonnes) ---")
    print(f"  {'rang':<16} {'temps':>10} {'us/colonne':>12}")
    for r in (5, 10, 15, 20):
        wr = bmf.BitMatrix.from_numpy(rand(m, r, seed=12))
        sr = bmf.BoolLs(wr)
        t = timeit(lambda s=sr: s.solve_all(x, method="zeta"), repeats=3)
        print(f"  r = {r:<12} {fmt_time(t):>10} {t / n * 1e6:11.2f}")

    print(f"\n  --- Factorisation complete (50 iterations max) ---")
    print(f"  {'rang':<16} {'temps':>10} {'erreur':>10} {'iterations':>11}")
    for r in (5, 10):
        res = bmf.ao_bmf(x, r, method="zeta", seed=0)
        t = timeit(lambda rr=r: bmf.ao_bmf(x, rr, method="zeta", seed=0), repeats=3)
        print(f"  r = {r:<12} {fmt_time(t):>10} {res.error:>10} {res.iterations:>11}")


def main() -> None:
    print("=" * 72)
    print(f"FPMsBMF {bmf.__version__}  vs  numpy {np.__version__}")
    print("=" * 72)
    print(f"  densite des matrices : {DENSITY}")
    print(f"  mesures : mediane de {REPEATS} executions, apres echauffement")
    print("\n  Fermez les autres applications : les mesures precedentes ont")
    print("  montre des ecarts d'un facteur deux selon l'etat thermique.")

    bench_memory()
    bench_conversion()
    bench_elementwise()
    bench_reductions()
    bench_product()
    bench_breakeven()
    bench_solvers()


if __name__ == "__main__":
    main()