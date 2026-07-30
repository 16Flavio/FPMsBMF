use crate::word::*;
use std::fmt;

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct BitVec {
    data: Vec<Word>,
    n: usize,
}

impl BitVec {
    pub fn zeros(n: usize) -> Self {
        Self {
            data: vec![0; words_for(n)],
            n,
        }
    }

    pub fn ones(n: usize) -> Self {
        let mut data = vec![Word::MAX; words_for(n)];

        mask_tail(&mut data, n);

        let v = Self { data, n };

        debug_assert!(v.is_canonical());

        v
    }

    pub fn from_bools(bools: &[bool]) -> Self {
        let n = bools.len();

        let mut bitvec = Self::zeros(n);

        for (i, item) in bools.iter().enumerate() {
            bitvec.set(i, *item);
        }

        debug_assert!(bitvec.is_canonical());

        bitvec
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn as_words(&self) -> &[Word] {
        &self.data
    }

    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.n, "index {i} hors limites (len = {})", self.n);
        let word_idx = word_index(i);
        let mask = bit_mask(i);
        (self.data[word_idx] & mask) != 0
    }

    pub fn set(&mut self, i: usize, value: bool) {
        assert!(i < self.n, "index {i} hors limites (len = {})", self.n);
        let word_idx = word_index(i);
        let mask = bit_mask(i);
        if value {
            self.data[word_idx] |= mask;
        } else {
            self.data[word_idx] &= !mask;
        }
    }

    pub fn flip(&mut self, i: usize) {
        assert!(i < self.n, "index {i} hors limites (len = {})", self.n);
        let word_idx = word_index(i);
        let mask = bit_mask(i);
        self.data[word_idx] ^= mask;
    }

    pub fn or(&mut self, other: &BitVec) {
        assert_eq!(self.len(), other.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a |= *b;
        }
        debug_assert!(self.is_canonical());
    }

    pub fn and(&mut self, other: &BitVec) {
        assert_eq!(self.len(), other.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a &= *b;
        }
        debug_assert!(self.is_canonical());
    }

    pub fn xor(&mut self, other: &BitVec) {
        assert_eq!(self.len(), other.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a ^= *b;
        }
        debug_assert!(self.is_canonical());
    }

    pub fn andnot(&mut self, other: &BitVec) {
        assert_eq!(self.len(), other.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a &= !*b;
        }
        debug_assert!(self.is_canonical());
    }

    pub fn invert(&mut self) {
        for a in self.data.iter_mut() {
            *a = !*a;
        }
        mask_tail(&mut self.data, self.n);
        debug_assert!(self.is_canonical());
    }

    pub fn count_ones(&self) -> usize {
        self.data
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>()
    }

    pub fn hamming(&self, other: &BitVec) -> usize {
        assert_eq!(self.len(), other.len());
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (*a ^ *b).count_ones() as usize)
            .sum::<usize>()
    }

    fn is_canonical(&self) -> bool {
        if self.data.is_empty() {
            return true;
        }
        let last_word: Word = *self.data.last().unwrap();
        (last_word & (!tail_mask(self.n))) == 0
    }

    fn display_seq_bit(&self, start: usize, end: usize, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in start..end {
            if self.get(i) {
                write!(f, "1")?;
            } else {
                write!(f, "0")?;
            }
            if (i + 1) % NUMBER_OF_BITS == 0 && i + 1 != end {
                write!(f, "|")?;
            } else if (i + 1) % 8 == 0 && i + 1 != end {
                write!(f, "_")?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for BitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.n;

        if n != 0 {
            if n <= 256 {
                self.display_seq_bit(0, n, f)?;
            } else {
                self.display_seq_bit(0, 128, f)?;
                write!(f, " ... ({}/{}) ", self.count_ones(), n)?;
                self.display_seq_bit(n - 128, n, f)?;
            }
        } else {
            write!(f, "<empty>")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn back_and_forth_test() {
        for n in [1, 63, 64, 65, 127, 128, 129, 137] {
            for i in 0..n {
                let mut bv = BitVec::zeros(n);
                bv.set(i, true);
                assert!(bv.get(i));
                for j in 0..n {
                    if j != i {
                        assert!(!bv.get(j));
                    }
                }
            }
        }
    }

    #[test]
    fn is_canonical_test() {
        let bv = BitVec {
            data: vec![Word::MAX, Word::MAX],
            n: 65,
        };
        assert!(!bv.is_canonical());
    }

    #[test]
    fn ones_test() {
        for n in 0..2000 {
            let bv = BitVec::ones(n);
            assert_eq!(bv.len(), n);
            let mut sum: usize = 0;
            for word in bv.as_words() {
                sum += word.count_ones() as usize;
            }
            assert_eq!(sum, n);
        }
    }

    #[test]
    fn zeros_test() {
        for n in 0..2000 {
            let bv = BitVec::zeros(n);
            let mut sum: usize = 0;
            for word in bv.as_words() {
                sum += word.count_ones() as usize;
            }
            assert_eq!(sum, 0);
        }
    }

    #[test]
    fn from_bools_test() {
        for n in 1..2000 {
            let mut bools = Vec::new();
            for i in 0..n {
                bools.push(i % 3 == 0);
            }

            let bv = BitVec::from_bools(&bools);

            for (i, item) in bools.iter().enumerate() {
                assert_eq!(bv.get(i), *item);
            }
        }
    }

    #[test]
    fn flip_test() {
        for n in [1, 63, 64, 65, 127, 128, 129, 137] {
            let mut bv = BitVec::ones(n);
            for i in 0..n {
                assert_eq!(
                    bv.as_words()
                        .iter()
                        .map(|&x| x.count_ones() as usize)
                        .sum::<usize>(),
                    n
                );
                assert!(bv.get(i));
                bv.flip(i);
                assert_eq!(
                    bv.as_words()
                        .iter()
                        .map(|&x| x.count_ones() as usize)
                        .sum::<usize>(),
                    n - 1
                );
                assert!(!bv.get(i));
                bv.flip(i);
                assert!(bv.get(i));
                assert_eq!(
                    bv.as_words()
                        .iter()
                        .map(|&x| x.count_ones() as usize)
                        .sum::<usize>(),
                    n
                );
            }
        }
    }

    #[test]
    fn bit_properties_test() {
        for n in 1..2000 {
            let mut a = Vec::new();
            for i in 0..n {
                a.push(i % 3 == 0);
            }
            let mut a = BitVec::from_bools(&a);

            let mut b = Vec::new();
            for i in 0..n {
                b.push(i % 5 == 0);
            }
            let b = BitVec::from_bools(&b);

            let a_copy = a.clone();

            a.or(&b);
            for i in 0..n {
                assert!(a.get(i) == (a_copy.get(i) || b.get(i)));
            }

            let mut a = a_copy.clone();

            a.and(&b);
            for i in 0..n {
                assert!(a.get(i) == (a_copy.get(i) && b.get(i)));
            }

            let mut a = a_copy.clone();

            a.xor(&b);
            for i in 0..n {
                assert!(a.get(i) == (a_copy.get(i) != b.get(i)));
            }

            let mut a = a_copy.clone();

            a.andnot(&b);
            for i in 0..n {
                assert!(a.get(i) == (a_copy.get(i) && (!b.get(i))));
            }
        }
    }

    #[test]
    fn hamming_test() {
        for n in 1..2000 {
            let mut a = Vec::new();
            for i in 0..n {
                a.push(i % 3 == 0);
            }
            let a = BitVec::from_bools(&a);

            let mut b = Vec::new();
            for i in 0..n {
                b.push(i % 5 == 0);
            }
            let b = BitVec::from_bools(&b);

            let mut a_copy = a.clone();

            a_copy.xor(&b);

            assert!(a.hamming(&b) == a_copy.count_ones())
        }
    }

    #[test]
    fn invert_test() {
        for n in 1..2000 {
            let mut a = Vec::new();
            for i in 0..n {
                a.push(i % 3 == 0);
            }
            let mut a = BitVec::from_bools(&a);

            let a_copy = a.clone();

            a.invert();
            assert_eq!(a.count_ones(), n - a_copy.count_ones());
            a.invert();
            assert_eq!(a, a_copy);
        }
    }

    #[test]
    fn display_test() {
        let bv = BitVec::from_bools(&[true, false, true]);
        let s = format!("{bv}");
        assert_eq!(s, "101");

        let bv = BitVec::from_bools(&[
            true, false, true, false, true, false, true, false, false, true,
        ]);
        let s = format!("{bv}");
        assert_eq!(s, "10101010_01");

        let bv = BitVec::from_bools(&[false; 512]);
        let s = format!("{bv}");
        assert_eq!(
            s,
            "00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000|00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000 ... (0/512) 00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000|00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000"
        );
    }
}
