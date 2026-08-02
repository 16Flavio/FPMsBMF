"""FPMsBMF : factorisation matricielle booleenne.

Ce fichier ne contient que des signatures : Python ne l'execute jamais.
Il sert aux editeurs et aux verificateurs de types, qui ne peuvent pas
introspecter un module natif.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

__version__: str

def methods() -> List[str]:
    """Noms de methode acceptes par l'argument `method`."""
    ...

class BitMatrix:
    """Matrice booleenne bit-packee.

    Un booleen occupe un bit et non un octet : huit fois moins de memoire
    qu'un tableau numpy, et les operations logiques traitent 64 entrees par
    instruction machine.

    L'interet principal face a numpy est la persistance : convertir coute
    O(m*n), garder l'objet entre deux appels evite de repayer ce cout.
    """

    def __init__(self, rows: int, cols: int, fill: bool = False) -> None: ...
    @staticmethod
    def zeros(rows: int, cols: int) -> "BitMatrix": ...
    @staticmethod
    def ones(rows: int, cols: int) -> "BitMatrix": ...
    @staticmethod
    def from_list(data: List[List[int]]) -> "BitMatrix":
        """Construit depuis une liste de listes de 0 et de 1.

        Les booleens Python sont acceptes : True et False valent 1 et 0.

        >>> BitMatrix.from_list([[1, 0, 1], [0, 1, 1]])
        <BitMatrix 2x3, 4 bits a 1>
        """
        ...

    def to_list(self) -> List[List[int]]:
        """Renvoie une liste de listes de 0 et de 1.

        Format symetrique de `from_list`. Pour un tableau de booleens,
        utiliser `to_numpy`.
        """
        ...

    @staticmethod
    def from_numpy(a: npt.NDArray[np.bool_]) -> "BitMatrix":
        """Packe un tableau numpy 2D de booleens. Cout O(m*n)."""
        ...

    def to_numpy(self) -> npt.NDArray[np.bool_]:
        """Depacke vers un tableau numpy 2D de booleens."""
        ...

    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    def count_ones(self) -> int:
        """Nombre d'entrees a 1."""
        ...

    def hamming(self, other: "BitMatrix") -> int:
        """Nombre d'entrees differentes, soit ||A - B||_F^2."""
        ...

    def count_andnot(self, other: "BitMatrix") -> int:
        """|A & ~B| : entrees a 1 dans self et a 0 dans other."""
        ...

    def row_count_ones(self, i: int) -> int: ...
    def transpose(self) -> "BitMatrix": ...
    def copy(self) -> "BitMatrix": ...
    def __getitem__(self, idx: Tuple[int, int]) -> bool: ...
    def __setitem__(self, idx: Tuple[int, int], value: bool) -> None: ...
    def __or__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __ior__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __and__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __iand__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __xor__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __ixor__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __sub__(self, other: "BitMatrix") -> "BitMatrix":
        """Retrait ensembliste A & ~B, pas une soustraction arithmetique."""
        ...

    def __invert__(self) -> "BitMatrix": ...
    def __matmul__(self, other: "BitMatrix") -> "BitMatrix":
        """Produit **booleen** : (W o H)_ij = OR_k (W_ik and H_kj)."""
        ...

    def __eq__(self, other: object) -> bool:
        """Egalite structurelle : un booleen, pas un tableau comme numpy."""
        ...

    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...

class BoolLs:
    """Solveur de min_h ||x - W o h||^2 pour h dans {0,1}^r.

    Le pretraitement ne depend que de W. Construire l'objet une fois et
    l'appeler sur plusieurs vecteurs amortit ce cout.
    """

    def __init__(self, w: BitMatrix) -> None: ...
    @property
    def m(self) -> int: ...
    @property
    def r(self) -> int: ...
    def solve(
        self,
        x: npt.NDArray[np.bool_],
        method: str = "zeta",
        seed: int = 0,
    ) -> Tuple[int, int]:
        """Renvoie (h, cost) ; le bit k de h designe la colonne k de W."""
        ...

    def solve_all(
        self,
        x: BitMatrix,
        method: str = "zeta",
        seed: int = 0,
    ) -> Tuple[BitMatrix, int]:
        """Resout toutes les colonnes de X et renvoie (H, erreur totale)."""
        ...

class BmfResult:
    """X est approche par `w @ h`."""

    w: BitMatrix
    h: BitMatrix
    error: int
    iterations: int
    def reconstruct(self) -> BitMatrix: ...

def ao_bmf(
    x: BitMatrix,
    r: int,
    method: str = "zeta",
    max_iter: int = 50,
    seed: int = 0,
) -> BmfResult:
    """Factorisation par optimisation alternee.

    Avec une methode exacte, l'erreur ne peut pas augmenter d'une iteration a
    l'autre — mais l'optimalite globale n'est pas garantie : l'algorithme
    converge vers un point stationnaire par blocs. Des graines differentes
    explorent des bassins differents.
    """
    ...

def boolls(
    w: BitMatrix,
    x: npt.NDArray[np.bool_],
    method: str = "zeta",
    seed: int = 0,
) -> Tuple[int, int]:
    """Resout un seul probleme BoolLS.

    Pour plusieurs vecteurs partageant le meme W, utiliser la classe `BoolLs` :
    le pretraitement n'est alors paye qu'une fois.
    """
    ...
