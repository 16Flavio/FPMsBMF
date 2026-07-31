#![allow(clippy::needless_range_loop)]

use crate::matrix::BitMatrix;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RefMatrix {
    data: Vec<Vec<bool>>,
    rows: usize,
    cols: usize,
}

impl RefMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(cols != 0, "cols égale à 0");
        assert!(rows != 0, "rows égale à 0");
        Self {
            data: vec![vec![false; cols]; rows],
            rows,
            cols,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        assert!(cols != 0, "cols égale à 0");
        assert!(rows != 0, "rows égale à 0");
        Self {
            data: vec![vec![true; cols]; rows],
            rows,
            cols,
        }
    }

    pub fn from_bools(bools: &[&[bool]]) -> Self {
        assert!(!bools.is_empty(), "from_bools: aucune ligne fournie");

        let rows = bools.len();
        let cols = bools[0].len();
        assert!(cols != 0, "from_bools: les lignes sont vides");

        for (i, row) in bools.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "from_bools: la ligne {i} n'a pas la même longueur que la ligne 0"
            );
        }

        let mut data = vec![vec![false; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                data[i][j] = bools[i][j];
            }
        }

        Self { data, rows, cols }
    }

    pub fn from_bit(bm: &BitMatrix) -> Self {
        let rows = bm.rows();
        let cols = bm.cols();
        let mut data = vec![vec![false; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                data[i][j] = bm.get(i, j);
            }
        }

        Self { data, rows, cols }
    }

    pub fn to_bit(&self) -> BitMatrix {
        let mut bm = BitMatrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                bm.set(i, j, self.data[i][j]);
            }
        }
        bm
    }

    pub fn get(&self, i: usize, j: usize) -> bool {
        assert!(
            i < self.rows,
            "index {i} hors limites (rows = {})",
            self.rows
        );
        assert!(
            j < self.cols,
            "index {j} hors limites (cols = {})",
            self.cols
        );

        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: bool) {
        assert!(
            i < self.rows,
            "index {i} hors limites (rows = {})",
            self.rows
        );
        assert!(
            j < self.cols,
            "index {j} hors limites (cols = {})",
            self.cols
        );

        self.data[i][j] = v;
    }

    pub fn or(&mut self, other: &RefMatrix) {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "les deux matrices sont de dimensions différentes"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] |= other.data[i][j];
            }
        }
    }

    pub fn and(&mut self, other: &RefMatrix) {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "les deux matrices sont de dimensions différentes"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] &= other.data[i][j];
            }
        }
    }

    pub fn xor(&mut self, other: &RefMatrix) {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "les deux matrices sont de dimensions différentes"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] ^= other.data[i][j];
            }
        }
    }

    pub fn andnot(&mut self, other: &RefMatrix) {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "les deux matrices sont de dimensions différentes"
        );

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] &= !other.data[i][j];
            }
        }
    }

    pub fn invert(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = !self.data[i][j];
            }
        }
    }

    pub fn count_ones(&self) -> usize {
        let mut sum: usize = 0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] {
                    sum += 1;
                }
            }
        }
        sum
    }

    pub fn hamming(&self, other: &RefMatrix) -> usize {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "les deux matrices sont de dimensions différentes"
        );

        let mut sum: usize = 0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] != other.data[i][j] {
                    sum += 1;
                }
            }
        }
        sum
    }

    pub fn count_andnot(&self, other: &RefMatrix) -> usize {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "les deux matrices sont de dimensions différentes"
        );

        let mut sum: usize = 0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] & !other.data[i][j] {
                    sum += 1;
                }
            }
        }
        sum
    }

    pub fn or_rows(&mut self, dst: usize, src: usize) {
        assert!(
            dst < self.rows,
            "ligne {dst} hors limites (rows = {})",
            self.rows
        );
        assert!(
            src < self.rows,
            "ligne {src} hors limites (rows = {})",
            self.rows
        );

        if dst == src {
            return;
        }

        for j in 0..self.cols {
            self.data[dst][j] |= self.data[src][j];
        }
    }

    pub fn and_rows(&mut self, dst: usize, src: usize) {
        assert!(
            dst < self.rows,
            "ligne {dst} hors limites (rows = {})",
            self.rows
        );
        assert!(
            src < self.rows,
            "ligne {src} hors limites (rows = {})",
            self.rows
        );

        if dst == src {
            return;
        }

        for j in 0..self.cols {
            self.data[dst][j] &= self.data[src][j];
        }
    }

    pub fn xor_rows(&mut self, dst: usize, src: usize) {
        assert!(
            dst < self.rows,
            "ligne {dst} hors limites (rows = {})",
            self.rows
        );
        assert!(
            src < self.rows,
            "ligne {src} hors limites (rows = {})",
            self.rows
        );

        for j in 0..self.cols {
            if dst == src {
                self.data[dst][j] = false;
            } else {
                self.data[dst][j] ^= self.data[src][j];
            }
        }
    }

    pub fn andnot_rows(&mut self, dst: usize, src: usize) {
        assert!(
            dst < self.rows,
            "ligne {dst} hors limites (rows = {})",
            self.rows
        );
        assert!(
            src < self.rows,
            "ligne {src} hors limites (rows = {})",
            self.rows
        );

        for j in 0..self.cols {
            if dst == src {
                self.data[dst][j] = false;
            } else {
                self.data[dst][j] &= !self.data[src][j];
            }
        }
    }

    pub fn product(&self, other: &RefMatrix) -> Self {
        assert!(
            self.cols == other.rows,
            "les deux matrices sont de dimensions incompatibles (A = {}x{} et B = {}x{})",
            self.rows,
            self.cols,
            other.rows,
            other.cols
        );

        let mut data = vec![vec![false; other.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut temp = false;
                for k in 0..self.cols {
                    if self.data[i][k] && other.data[k][j] {
                        temp = true;
                        break;
                    }
                }
                data[i][j] = temp;
            }
        }

        Self {
            data,
            rows: self.rows,
            cols: other.cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn hamming_masked(&self, other: &RefMatrix, mask: &RefMatrix) -> usize {
        assert!(
            self.rows == other.rows
                && self.cols == other.cols
                && self.rows == mask.rows
                && self.cols == mask.cols,
            "les trois matrices sont de dimensions différentes"
        );

        let mut sum: usize = 0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.data[i][j] != other.data[i][j] && mask.data[i][j] {
                    sum += 1;
                }
            }
        }
        sum
    }

    pub fn row_count_ones(&self, i: usize) -> usize {
        assert!(
            i < self.rows,
            "index {i} hors limites (rows = {})",
            self.rows
        );

        let mut sum: usize = 0;
        for j in 0..self.cols {
            if self.data[i][j] {
                sum += 1;
            }
        }

        sum
    }

    pub fn row_hamming(&self, dst: usize, src: usize) -> usize {
        assert!(
            dst < self.rows,
            "ligne {dst} hors limites (rows = {})",
            self.rows
        );
        assert!(
            src < self.rows,
            "ligne {src} hors limites (rows = {})",
            self.rows
        );

        let mut sum: usize = 0;
        for j in 0..self.cols {
            if self.data[dst][j] != self.data[src][j] {
                sum += 1;
            }
        }
        sum
    }

    pub fn transpose(&self) -> Self {
        let mut data = vec![vec![false; self.rows]; self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self.data[i][j];
            }
        }

        Self {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }
}
