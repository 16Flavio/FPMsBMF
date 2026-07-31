use crate::word::*;
use std::fmt;

fn transpose_block(a: &mut [Word; NUMBER_OF_BITS]) {
    let mut j = NUMBER_OF_BITS / 2;
    let mut m: Word = !0 >> (NUMBER_OF_BITS / 2);

    while j != 0 {
        let mut base = 0;
        while base < NUMBER_OF_BITS {
            for k in base..base + j {
                let t = ((a[k] >> j) ^ a[k + j]) & m;
                a[k] ^= t << j;
                a[k + j] ^= t;
            }
            base += 2 * j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct BitMatrix {
    data: Vec<Word>,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl BitMatrix {
    const DISPLAY_MAX_ROWS: usize = 20;
    const DISPLAY_MAX_COLS: usize = 128;
    const DISPLAY_EDGE: usize = 64;

    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(cols != 0, "cols égale à 0");
        assert!(rows != 0, "rows égale à 0");
        let stride = words_for(cols);
        Self {
            data: vec![0; rows * stride],
            rows,
            cols,
            stride,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        assert!(cols != 0, "cols égale à 0");
        assert!(rows != 0, "rows égale à 0");
        let stride = words_for(cols);
        let mut data = vec![Word::MAX; rows * stride];

        for chunk in data.chunks_exact_mut(stride) {
            mask_tail(chunk, cols);
        }

        let bm = Self {
            data,
            rows,
            cols,
            stride,
        };

        debug_assert!(bm.is_canonical());

        bm
    }

    pub fn from_bools(input: &[&[bool]]) -> Self {
        assert!(!input.is_empty(), "from_bools: aucune ligne fournie");

        let rows = input.len();
        let cols = input[0].len();
        assert!(cols != 0, "from_bools: les lignes sont vides");

        for (i, row) in input.iter().enumerate() {
            assert_eq!(
                row.len(),
                cols,
                "from_bools: la ligne {i} n'a pas la même longueur que la ligne 0"
            );
        }

        let mut bm = Self::zeros(rows, cols);
        for (i, row) in input.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v {
                    bm.set(i, j, true);
                }
            }
        }
        bm
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn as_words(&self) -> &[Word] {
        &self.data
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

        let mask = bit_mask(j);

        (self.data[i * self.stride + word_index(j)] & mask) != 0
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

        let mask = bit_mask(j);

        if v {
            self.data[i * self.stride + word_index(j)] |= mask;
        } else {
            self.data[i * self.stride + word_index(j)] &= !mask;
        }

        debug_assert!(self.row_is_canonical(i));
    }

    pub fn row(&self, i: usize) -> &[Word] {
        assert!(
            i < self.rows,
            "index {i} hors limites (rows = {})",
            self.rows
        );
        &self.data[i * self.stride..(i + 1) * self.stride]
    }

    pub fn or_rows(&mut self, dst: usize, src: usize) {
        if dst == src {
            return;
        }
        let (d, s) = self.two_rows_mut(dst, src);
        for (a, b) in d.iter_mut().zip(s.iter()) {
            *a |= *b;
        }
        debug_assert!(self.row_is_canonical(dst));
    }

    pub fn and_rows(&mut self, dst: usize, src: usize) {
        if dst == src {
            return;
        }
        let (d, s) = self.two_rows_mut(dst, src);
        for (a, b) in d.iter_mut().zip(s.iter()) {
            *a &= *b;
        }
        debug_assert!(self.row_is_canonical(dst));
    }

    pub fn xor_rows(&mut self, dst: usize, src: usize) {
        if dst == src {
            self.row_mut(dst).fill(0);
            return;
        }
        let (d, s) = self.two_rows_mut(dst, src);
        for (a, b) in d.iter_mut().zip(s.iter()) {
            *a ^= *b;
        }
        debug_assert!(self.row_is_canonical(dst));
    }

    pub fn andnot_rows(&mut self, dst: usize, src: usize) {
        if dst == src {
            self.row_mut(dst).fill(0);
            return;
        }
        let (d, s) = self.two_rows_mut(dst, src);
        for (a, b) in d.iter_mut().zip(s.iter()) {
            *a &= !*b;
        }
        debug_assert!(self.row_is_canonical(dst));
    }

    pub fn or(&mut self, other: &BitMatrix) {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "les deux matrices sont de dimensions différentes"
        );

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a |= *b;
        }
    }

    pub fn and(&mut self, other: &BitMatrix) {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "les deux matrices sont de dimensions différentes"
        );

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a &= *b;
        }
    }

    pub fn xor(&mut self, other: &BitMatrix) {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "les deux matrices sont de dimensions différentes"
        );

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a ^= *b;
        }
    }

    pub fn andnot(&mut self, other: &BitMatrix) {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "les deux matrices sont de dimensions différentes"
        );

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a &= !*b;
        }
    }

    pub fn count_ones(&self) -> usize {
        self.data.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn hamming(&self, other: &BitMatrix) -> usize {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "les deux matrices sont de dimensions différentes"
        );

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (*a ^ *b).count_ones() as usize)
            .sum()
    }

    pub fn invert(&mut self) {
        for a in self.data.iter_mut() {
            *a = !*a;
        }

        for chunk in self.data.chunks_exact_mut(self.stride) {
            mask_tail(chunk, self.cols);
        }

        debug_assert!(self.is_canonical());
    }

    pub fn count_andnot(&self, other: &BitMatrix) -> usize {
        assert!(
            self.rows() == other.rows() && self.cols() == other.cols(),
            "les deux matrices sont de dimensions différentes"
        );

        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (*a & !*b).count_ones() as usize)
            .sum()
    }

    pub fn hamming_masked(&self, other: &BitMatrix, mask: &BitMatrix) -> usize {
        assert!(
            self.rows() == other.rows()
                && self.cols() == other.cols()
                && self.rows() == mask.rows()
                && self.cols() == mask.cols(),
            "une des trois matrices a des dimensions différentes"
        );

        self.data
            .iter()
            .zip(other.data.iter().zip(mask.data.iter()))
            .map(|(a, (b, m))| ((*a ^ *b) & *m).count_ones() as usize)
            .sum()
    }

    pub fn row_count_ones(&self, i: usize) -> usize {
        self.row(i).iter().map(|a| a.count_ones() as usize).sum()
    }

    pub fn row_hamming(&self, i: usize, k: usize) -> usize {
        assert!(
            i < self.rows,
            "ligne {i} hors limites (rows = {})",
            self.rows
        );
        assert!(
            k < self.rows,
            "ligne {k} hors limites (rows = {})",
            self.rows
        );

        if i == k {
            return 0;
        }

        let row_dst = self.row(i);
        let row_src = self.row(k);

        row_dst
            .iter()
            .zip(row_src.iter())
            .map(|(a, b)| (*a ^ *b).count_ones() as usize)
            .sum()
    }

    fn accumulate_into(&self, other: &BitMatrix, out: &mut BitMatrix) {
        debug_assert_eq!(out.rows, self.rows);
        debug_assert_eq!(out.cols, other.cols);

        let stride = other.stride;

        for i in 0..self.rows {
            let base = i * stride;
            let dst = &mut out.data[base..base + stride];

            for (w, &word) in self.row(i).iter().enumerate() {
                let mut m = word;
                while m != 0 {
                    let l = w * NUMBER_OF_BITS + m.trailing_zeros() as usize;
                    debug_assert!(l < other.rows);
                    let src = &other.data[l * stride..l * stride + stride];
                    for (a, b) in dst.iter_mut().zip(src.iter()) {
                        *a |= *b;
                    }
                    m &= m - 1;
                }
            }
        }

        debug_assert!(out.is_canonical());
    }

    pub fn product_into(&self, other: &BitMatrix, out: &mut BitMatrix) {
        assert_eq!(
            self.cols, other.rows,
            "dimensions incompatibles : A est {}x{}, B est {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        assert_eq!(out.rows, self.rows, "out.rows incorrect");
        assert_eq!(out.cols, other.cols, "out.cols incorrect");

        out.data.fill(0);
        self.accumulate_into(other, out);
    }

    pub fn product(&self, other: &BitMatrix) -> BitMatrix {
        assert_eq!(
            self.cols, other.rows,
            "dimensions incompatibles : A est {}x{}, B est {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );
        let mut out = BitMatrix::zeros(self.rows, other.cols);
        self.accumulate_into(other, &mut out);
        out
    }

    pub fn transpose(&self) -> BitMatrix {
        let mut t = BitMatrix::zeros(self.cols, self.rows);
        let src_stride = self.stride;
        let dst_stride = t.stride;

        let mut i0 = 0;
        while i0 < self.rows {
            let rows_here = (self.rows - i0).min(NUMBER_OF_BITS);
            let bi = i0 / NUMBER_OF_BITS;

            let mut j0 = 0;
            while j0 < self.cols {
                let cols_here = (self.cols - j0).min(NUMBER_OF_BITS);
                let bj = j0 / NUMBER_OF_BITS;

                let mut block = [0 as Word; NUMBER_OF_BITS];
                for r in 0..rows_here {
                    block[r] = self.data[(i0 + r) * src_stride + bj];
                }

                transpose_block(&mut block);

                for c in 0..cols_here {
                    t.data[(j0 + c) * dst_stride + bi] = block[c];
                }

                j0 += NUMBER_OF_BITS;
            }
            i0 += NUMBER_OF_BITS;
        }

        debug_assert!(t.is_canonical());
        t
    }

    pub(crate) fn row_mut(&mut self, i: usize) -> &mut [Word] {
        assert!(
            i < self.rows,
            "index {i} hors limites (rows = {})",
            self.rows
        );
        &mut self.data[i * self.stride..(i + 1) * self.stride]
    }

    fn is_canonical(&self) -> bool {
        (0..self.rows).all(|i| self.row_is_canonical(i))
    }

    fn row_is_canonical(&self, i: usize) -> bool {
        debug_assert!(i < self.rows);
        self.data[(i + 1) * self.stride - 1] & !tail_mask(self.cols) == 0
    }

    fn two_rows_mut(&mut self, dst: usize, src: usize) -> (&mut [Word], &[Word]) {
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
        debug_assert_ne!(dst, src, "two_rows_mut exige deux lignes distinctes");

        let stride = self.stride;
        let (left, right) = self.data.split_at_mut(dst.max(src) * stride);

        if dst < src {
            (
                &mut left[dst * stride..dst * stride + stride],
                &right[..stride],
            )
        } else {
            (
                &mut right[..stride],
                &left[src * stride..src * stride + stride],
            )
        }
    }

    fn fmt_row_range(
        &self,
        i: usize,
        start: usize,
        end: usize,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        for j in start..end {
            write!(f, "{}", if self.get(i, j) { '1' } else { '0' })?;
            if j + 1 != end {
                if (j + 1) % NUMBER_OF_BITS == 0 {
                    write!(f, "|")?;
                } else if (j + 1) % 8 == 0 {
                    write!(f, "_")?;
                }
            }
        }
        Ok(())
    }

    fn fmt_row(&self, i: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.cols <= Self::DISPLAY_MAX_COLS {
            self.fmt_row_range(i, 0, self.cols, f)
        } else {
            self.fmt_row_range(i, 0, Self::DISPLAY_EDGE, f)?;
            write!(f, " ... ")?;
            self.fmt_row_range(i, self.cols - Self::DISPLAY_EDGE, self.cols, f)
        }
    }
}

impl fmt::Display for BitMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}x{} ({} bits a 1)",
            self.rows,
            self.cols,
            self.count_ones()
        )?;

        if self.rows <= Self::DISPLAY_MAX_ROWS {
            for i in 0..self.rows {
                self.fmt_row(i, f)?;
                if i + 1 != self.rows {
                    writeln!(f)?;
                }
            }
        } else {
            let edge = Self::DISPLAY_MAX_ROWS / 2;
            for i in 0..edge {
                self.fmt_row(i, f)?;
                writeln!(f)?;
            }
            writeln!(f, "... ({} lignes omises) ...", self.rows - 2 * edge)?;
            for i in self.rows - edge..self.rows {
                self.fmt_row(i, f)?;
                if i + 1 != self.rows {
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::RefMatrix;

    const TRIPLES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (3, 65, 70),
        (4, 128, 3),
        (2, 129, 127),
        (100, 3, 64),
        (5, 64, 65),
        (7, 70, 130),
    ];

    const DIMS_T: &[(usize, usize)] = &[
        (1, 1),
        (1, 200),
        (200, 1),
        (63, 63),
        (64, 64),
        (65, 65),
        (64, 65),
        (65, 64),
        (127, 129),
        (129, 127),
        (3, 65),
        (100, 3),
        (70, 70),
        (128, 128),
        (130, 200),
    ];

    const DIMS_SMALL: &[(usize, usize)] = &[(1, 1), (3, 65), (4, 70), (2, 127), (5, 129)];

    fn pattern(rows: usize, cols: usize, seed: usize) -> BitMatrix {
        let mut m = BitMatrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if (i * 7 + j * 13 + seed * 5) % 3 == 0 {
                    m.set(i, j, true);
                }
            }
        }
        m
    }

    #[test]
    fn zeros_dims_and_popcount() {
        for &(rows, cols) in DIMS_T {
            let m = BitMatrix::zeros(rows, cols);
            assert_eq!(m.rows(), rows, "rows pour {rows}x{cols}");
            assert_eq!(m.cols(), cols, "cols pour {rows}x{cols}");
            assert_eq!(m.count_ones(), 0, "popcount pour {rows}x{cols}");
            assert!(m.is_canonical(), "canonicite pour {rows}x{cols}");
        }
    }

    #[test]
    fn ones_popcount_is_rows_times_cols() {
        for &(rows, cols) in DIMS_T {
            let m = BitMatrix::ones(rows, cols);
            assert_eq!(m.count_ones(), rows * cols, "pour {rows}x{cols}");
            assert!(m.is_canonical(), "pour {rows}x{cols}");
            for i in 0..rows {
                for j in 0..cols {
                    assert!(m.get(i, j), "ones({rows},{cols}) : bit ({i},{j}) eteint");
                }
            }
        }
    }

    #[test]
    fn single_set_lands_exactly_there() {
        for &(rows, cols) in DIMS_T {
            for i in 0..rows {
                for j in 0..cols {
                    let mut m = BitMatrix::zeros(rows, cols);
                    m.set(i, j, true);

                    assert_eq!(
                        m.count_ones(),
                        1,
                        "{rows}x{cols} : set({i},{j}) a ecrit {} bits",
                        m.count_ones()
                    );

                    for a in 0..rows {
                        for b in 0..cols {
                            assert_eq!(
                                m.get(a, b),
                                (a, b) == (i, j),
                                "{rows}x{cols} : set({i},{j}) puis get({a},{b})"
                            );
                        }
                    }

                    m.set(i, j, false);
                    assert_eq!(m.count_ones(), 0, "{rows}x{cols} : effacement ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn from_bools_roundtrip() {
        for &(rows, cols) in DIMS_T {
            let grid: Vec<Vec<bool>> = (0..rows)
                .map(|i| (0..cols).map(|j| (i + j) % 3 == 0).collect())
                .collect();
            let refs: Vec<&[bool]> = grid.iter().map(|r| r.as_slice()).collect();

            let m = BitMatrix::from_bools(&refs);
            assert_eq!(m.rows(), rows);
            assert_eq!(m.cols(), cols);
            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(m.get(i, j), grid[i][j], "{rows}x{cols} en ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn is_canonical_detects_a_dirty_row() {
        for dirty in 0..3 {
            let mut m = BitMatrix::ones(3, 65);
            assert!(m.is_canonical());
            m.data[dirty * 2 + 1] = Word::MAX;
            assert!(
                !m.is_canonical(),
                "queue sale sur la ligne {dirty} non detectee"
            );
        }
    }

    fn check_row_op<F>(
        before: &BitMatrix,
        after: &BitMatrix,
        dst: usize,
        src: usize,
        op: F,
        name: &str,
    ) where
        F: Fn(bool, bool) -> bool,
    {
        let (rows, cols) = (before.rows(), before.cols());

        for j in 0..cols {
            let expected = op(before.get(dst, j), before.get(src, j));
            assert_eq!(
                after.get(dst, j),
                expected,
                "{name}({dst},{src}) : bit {j} de la ligne cible"
            );
        }

        for i in 0..rows {
            if i == dst {
                continue;
            }
            for j in 0..cols {
                assert_eq!(
                    after.get(i, j),
                    before.get(i, j),
                    "{name}({dst},{src}) : la ligne {i} a ete modifiee (bit {j})"
                );
            }
        }

        assert!(after.is_canonical(), "{name}({dst},{src}) : queue sale");
    }

    #[test]
    fn row_ops_are_correct_and_local() {
        for &(rows, cols) in DIMS_SMALL {
            let base = pattern(rows, cols, 1);

            for dst in 0..rows {
                for src in 0..rows {
                    let mut m = base.clone();
                    m.or_rows(dst, src);
                    check_row_op(&base, &m, dst, src, |a, b| a || b, "or_rows");

                    let mut m = base.clone();
                    m.and_rows(dst, src);
                    check_row_op(&base, &m, dst, src, |a, b| a && b, "and_rows");

                    let mut m = base.clone();
                    m.xor_rows(dst, src);
                    check_row_op(&base, &m, dst, src, |a, b| a != b, "xor_rows");

                    let mut m = base.clone();
                    m.andnot_rows(dst, src);
                    check_row_op(&base, &m, dst, src, |a, b| a && !b, "andnot_rows");
                }
            }
        }
    }

    #[test]
    fn row_count_ones_and_row_hamming() {
        for &(rows, cols) in DIMS_SMALL {
            let m = pattern(rows, cols, 2);

            for i in 0..rows {
                let expected = (0..cols).filter(|&j| m.get(i, j)).count();
                assert_eq!(m.row_count_ones(i), expected, "row_count_ones({i})");
            }

            for i in 0..rows {
                for k in 0..rows {
                    let expected = (0..cols).filter(|&j| m.get(i, j) != m.get(k, j)).count();
                    assert_eq!(m.row_hamming(i, k), expected, "row_hamming({i},{k})");
                }
            }
        }
    }

    #[test]
    fn matrix_ops_are_elementwise() {
        for &(rows, cols) in DIMS_T {
            let a = pattern(rows, cols, 1);
            let b = pattern(rows, cols, 2);

            let mut m = a.clone();
            m.or(&b);
            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(m.get(i, j), a.get(i, j) || b.get(i, j), "or ({i},{j})");
                }
            }
            assert!(m.is_canonical());

            let mut m = a.clone();
            m.and(&b);
            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(m.get(i, j), a.get(i, j) && b.get(i, j), "and ({i},{j})");
                }
            }

            let mut m = a.clone();
            m.xor(&b);
            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(m.get(i, j), a.get(i, j) != b.get(i, j), "xor ({i},{j})");
                }
            }

            let mut m = a.clone();
            m.andnot(&b);
            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(m.get(i, j), a.get(i, j) && !b.get(i, j), "andnot ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn invert_complements_and_is_involutive() {
        for &(rows, cols) in DIMS_T {
            let a = pattern(rows, cols, 3);
            let before = a.count_ones();

            let mut m = a.clone();
            m.invert();
            assert_eq!(
                m.count_ones(),
                rows * cols - before,
                "invert sur {rows}x{cols}"
            );
            assert!(m.is_canonical(), "invert sur {rows}x{cols}");

            m.invert();
            assert_eq!(m, a, "double invert sur {rows}x{cols}");
        }
    }

    #[test]
    fn hamming_matches_xor_then_count() {
        for &(rows, cols) in DIMS_T {
            let a = pattern(rows, cols, 1);
            let b = pattern(rows, cols, 2);

            let mut t = a.clone();
            t.xor(&b);
            assert_eq!(a.hamming(&b), t.count_ones(), "hamming sur {rows}x{cols}");
            assert_eq!(a.hamming(&a), 0, "hamming reflexif sur {rows}x{cols}");
            assert_eq!(a.hamming(&b), b.hamming(&a), "symetrie sur {rows}x{cols}");
        }
    }

    #[test]
    fn count_andnot_matches_andnot_then_count() {
        for &(rows, cols) in DIMS_T {
            let a = pattern(rows, cols, 1);
            let b = pattern(rows, cols, 2);

            let mut t = a.clone();
            t.andnot(&b);
            assert_eq!(a.count_andnot(&b), t.count_ones(), "sur {rows}x{cols}");

            let mut u = a.clone();
            u.and(&b);
            assert_eq!(
                a.count_andnot(&b) + u.count_ones(),
                a.count_ones(),
                "partition sur {rows}x{cols}"
            );
        }
    }

    #[test]
    fn hamming_masked_degenerates_correctly() {
        for &(rows, cols) in DIMS_T {
            let a = pattern(rows, cols, 1);
            let b = pattern(rows, cols, 2);

            let full = BitMatrix::ones(rows, cols);
            assert_eq!(
                a.hamming_masked(&b, &full),
                a.hamming(&b),
                "masque plein sur {rows}x{cols}"
            );

            let empty = BitMatrix::zeros(rows, cols);
            assert_eq!(
                a.hamming_masked(&b, &empty),
                0,
                "masque vide sur {rows}x{cols}"
            );
        }
    }

    #[test]
    fn display_shows_bits_and_dims() {
        let m = BitMatrix::from_bools(&[&[true, false, true], &[false, true, false]]);
        let s = format!("{m}");
        assert!(s.contains("2x3"), "en-tete absent : {s}");
        assert!(s.contains("101"), "ligne 0 absente : {s}");
        assert!(s.contains("010"), "ligne 1 absente : {s}");

        let wide = BitMatrix::ones(1, 300);
        let s = format!("{wide}");
        assert!(s.contains("..."), "troncature en largeur absente");

        let tall = BitMatrix::ones(50, 4);
        let s = format!("{tall}");
        assert!(s.contains("lignes omises"), "troncature en hauteur absente");
    }

    #[test]
    fn conversion_roundtrip() {
        for &(rows, cols) in DIMS_T {
            let m = pattern(rows, cols, 1);
            let r = RefMatrix::from_bit(&m);
            assert_eq!(r.rows(), rows, "rows apres conversion");
            assert_eq!(r.cols(), cols, "cols apres conversion");
            assert_eq!(r.to_bit(), m, "aller-retour sur {rows}x{cols}");
        }
    }

    #[test]
    fn product_matches_reference() {
        for &(m, k, n) in TRIPLES {
            let a = pattern(m, k, 1);
            let b = pattern(k, n, 2);

            let got = a.product(&b);
            let want = RefMatrix::from_bit(&a)
                .product(&RefMatrix::from_bit(&b))
                .to_bit();

            assert_eq!(got.rows(), m, "rows du produit {m}x{k} * {k}x{n}");
            assert_eq!(got.cols(), n, "cols du produit {m}x{k} * {k}x{n}");
            assert_eq!(
                got, want,
                "produit {m}x{k} * {k}x{n}\nobtenu:\n{got}\nattendu:\n{want}"
            );
        }
    }

    #[test]
    fn product_identities() {
        for &(rows, cols) in DIMS_SMALL {
            let a = pattern(rows, cols, 1);

            let mut id_right = BitMatrix::zeros(cols, cols);
            for j in 0..cols {
                id_right.set(j, j, true);
            }
            assert_eq!(a.product(&id_right), a, "A o I sur {rows}x{cols}");

            let mut id_left = BitMatrix::zeros(rows, rows);
            for i in 0..rows {
                id_left.set(i, i, true);
            }
            assert_eq!(id_left.product(&a), a, "I o A sur {rows}x{cols}");

            let zero = BitMatrix::zeros(cols, 5);
            assert_eq!(a.product(&zero).count_ones(), 0, "A o 0 sur {rows}x{cols}");
        }
    }

    #[test]
    fn transpose_matches_reference() {
        for &(rows, cols) in DIMS_T {
            let m = pattern(rows, cols, 1);
            let got = m.transpose();
            let want = RefMatrix::from_bit(&m).transpose().to_bit();

            assert_eq!(got.rows(), cols, "rows de la transposee de {rows}x{cols}");
            assert_eq!(got.cols(), rows, "cols de la transposee de {rows}x{cols}");
            assert_eq!(got, want, "transposee de {rows}x{cols}");
        }
    }

    #[test]
    fn transpose_is_involutive_and_preserves_popcount() {
        for &(rows, cols) in DIMS_T {
            let m = pattern(rows, cols, 2);
            let t = m.transpose();

            assert_eq!(t.count_ones(), m.count_ones(), "popcount sur {rows}x{cols}");
            assert!(t.is_canonical(), "queue sale sur {rows}x{cols}");
            assert_eq!(t.transpose(), m, "double transposee sur {rows}x{cols}");

            for i in 0..rows {
                for j in 0..cols {
                    assert_eq!(t.get(j, i), m.get(i, j), "({i},{j}) sur {rows}x{cols}");
                }
            }
        }
    }

    #[test]
    fn transpose_single_bit() {
        for &(rows, cols) in &[(65, 65), (129, 127), (64, 128), (200, 1)] {
            for i in [0, 1, 63, 64, 65, rows - 1] {
                for j in [0, 1, 63, 64, 65, cols - 1] {
                    if i >= rows || j >= cols {
                        continue;
                    }
                    let mut src = BitMatrix::zeros(rows, cols);
                    src.set(i, j, true);
                    let t = src.transpose();

                    assert_eq!(t.count_ones(), 1, "{rows}x{cols} : bit ({i},{j})");
                    assert!(t.get(j, i), "{rows}x{cols} : bit ({i},{j}) mal place");
                }
            }
        }
    }

    #[test]
    fn transpose_of_product_reverses_factors() {
        for &(m, k, n) in TRIPLES {
            let a = pattern(m, k, 1);
            let b = pattern(k, n, 2);

            let left = a.product(&b).transpose();
            let right = b.transpose().product(&a.transpose());

            assert_eq!(left, right, "(A o B)^T sur {m}x{k} * {k}x{n}");
        }
    }

    #[test]
    fn product_into_matches_product_and_clears_buffer() {
        for &(m, k, n) in TRIPLES {
            let a = pattern(m, k, 1);
            let b = pattern(k, n, 2);
            let want = a.product(&b);

            let mut out = BitMatrix::ones(m, n);
            a.product_into(&b, &mut out);
            assert_eq!(out, want, "tampon sale, {m}x{k} * {k}x{n}");

            let a2 = pattern(m, k, 7);
            a2.product_into(&b, &mut out);
            assert_eq!(out, a2.product(&b), "reutilisation, {m}x{k} * {k}x{n}");
        }
    }

    #[test]
    fn product_into_handles_empty_rows() {
        let mut a = BitMatrix::zeros(4, 65);
        a.set(1, 0, true);
        a.set(3, 64, true);
        let b = pattern(65, 130, 3);

        let mut out = BitMatrix::ones(4, 130);
        a.product_into(&b, &mut out);

        assert_eq!(out.row_count_ones(0), 0, "ligne 0 de A vide");
        assert_eq!(out.row_count_ones(2), 0, "ligne 2 de A vide");
        assert_eq!(out, a.product(&b));
    }
}
