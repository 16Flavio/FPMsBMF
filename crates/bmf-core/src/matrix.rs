use crate::word::*;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct BitMatrix {
    data: Vec<Word>,
    rows: usize,
    cols: usize,
    stride: usize,
}

impl BitMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(cols != 0, "cols égale à 0");
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

        debug_assert!(self.is_canonical());
    }

    pub fn row(&self, i: usize) -> &[Word] {
        assert!(
            i < self.rows,
            "index {i} hors limites (rows = {})",
            self.rows
        );
        &self.data[i * self.stride..(i + 1) * self.stride]
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
        if self.data.is_empty() {
            return true;
        }
        let keep = tail_mask(self.cols);
        self.data
            .chunks_exact(self.stride)
            .all(|row| (*row.last().unwrap() & !keep) == 0)
    }
}
