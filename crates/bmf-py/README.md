# FPMsBMF

Factorisation matricielle booléenne : structures bit-packées et solveurs exacts.

Étant donné une matrice binaire `X` de taille `m × n` et un rang `r`, on cherche
`W` (`m × r`) et `H` (`r × n`) binaires minimisant

```
||X - W ∘ H||²   où   (W ∘ H)[i,j] = OR_k (W[i,k] ET H[k,j])
```

Le problème est NP-difficile et inapproximable à tout facteur multiplicatif.
Cette bibliothèque fournit des solveurs **exacts** pour le sous-problème à un
facteur fixé, ce qui permet une optimisation alternée dont chaque demi-itération
est un minimum global sur son bloc.

## Installation

```bash
pip install FPMsBMF
```

## Utilisation

```python
import numpy as np
import FPMsBMF as bmf

X_np = np.random.rand(500, 300) < 0.3
X = bmf.BitMatrix.from_numpy(X_np)      # packing : une seule fois

res = bmf.ao_bmf(X, r=8, method="zeta", seed=0)
print(res.error, res.iterations)
print(res.w.shape, res.h.shape)

# res.w @ res.h est le produit booléen
assert res.error == X.hamming(res.w @ res.h)
```

L'optimisation alternée converge vers un point stationnaire par blocs, pas vers
l'optimum global. Des graines différentes explorent des bassins différents :

```python
best = min(bmf.ao_bmf(X, r=8, seed=s) for s in range(20), key=lambda r: r.error)
```

## Le type `BitMatrix`

Un booléen occupe un bit et non un octet : huit fois moins de mémoire qu'un
tableau numpy, et les opérations logiques traitent 64 entrées par instruction.

L'intérêt principal face à numpy n'est pas la vitesse brute des opérations
élémentaires, numpy est déjà vectorisé, mais la **persistance de la
représentation packée**. La conversion coûte `O(m·n)` ; garder l'objet entre
deux appels évite de repayer ce coût à chaque expérience.

```python
A = bmf.BitMatrix.from_numpy(a)
B = bmf.BitMatrix.from_numpy(b)

A | B      # ou logique         (nouvelle matrice)
A |= B     # en place, sans allocation
A & B      # et logique
A ^ B      # ou exclusif
A - B      # retrait ensembliste A & ~B
~A         # complément
A @ B      # produit BOOLÉEN, pas arithmétique

A.count_ones()          # nombre d'entrées à 1
A.hamming(B)            # ||A - B||², soit le nombre d'entrées différentes
A.count_andnot(B)       # |A & ~B|
A.transpose()
A[i, j]                 # lecture / écriture d'une entrée
```

## Le sous-problème BoolLS

À `W` fixé, le problème se décompose en `n` sous-problèmes indépendants, un par
colonne. Résoudre `min_h ||x - W ∘ h||²` est le cœur de l'algorithme.

```python
solver = bmf.BoolLs(W)          # prétraitement, amorti sur les appels suivants
h, cost = solver.solve(x_np, method="zeta")
H, total = solver.solve_all(X, method="zeta")
```

Quatre méthodes sont disponibles :

| `method`     | Nature      | Coût par colonne         |
|--------------|-------------|--------------------------|
| `"naive"`    | exact       | `O(2^r · m)`             |
| `"zeta"`     | exact       | `O(nnz(x) + 2^r · r)`    |
| `"greedy"`   | heuristique | `O(r² · m/64)`           |
| `"greedy-ls"`| heuristique | `O(r³ log r · m/64)`     |

`"zeta"` regroupe les lignes par motif, il n'y en a que `min(m, 2^r)` distincts,
puis évalue tous les candidats en une transformée zêta sur le treillis des
sous-ensembles. Après le prétraitement, `m` a disparu du coût par colonne.

Les deux méthodes exactes donnent toujours le même résultat ; `"zeta"` est plus
rapide dès que `m > r · 64`.

## Limites

- Le rang est plafonné à 26 : le tableau interne de la transformée zêta compte
  `2^r` entrées.
- La conversion depuis numpy est une copie obligatoire : un booléen numpy occupe
  un octet, un bit ici.
- L'optimisation alternée n'offre aucune garantie d'optimalité globale.

## Licence

MIT
