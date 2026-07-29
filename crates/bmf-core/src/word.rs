pub type Word = u64;
pub const NUMBER_OF_BITS: usize = Word::BITS as usize;

// Fonction qui pour un nombre de bit n, renvoie le nombre de mot nécessaire
pub(crate) const fn words_for(n: usize) -> usize {
    n.div_ceil(NUMBER_OF_BITS)
}

// Fonction qui pour un indice vers le bit, renvoie l'indice vers le mot qui le contient
pub(crate) const fn word_index(i: usize) -> usize {
    i / NUMBER_OF_BITS
}

// Fonction qui pour un indice vers le bit, renvoie l'indice vers le bit à l'intérieur du mot
pub(crate) const fn bit_index(i: usize) -> usize {
    i % NUMBER_OF_BITS
}

// Fonction qui pour un indice vers le bit, renvoie un mask Word
pub(crate) const fn bit_mask(i: usize) -> Word {
    (1 as Word) << bit_index(i)
}

// Fonction qui pour un nombre de bit, renvoie un mask Word
pub(crate) const fn tail_mask(n: usize) -> Word {
    if n == 0 {
        0
    } else if bit_index(n) != 0 {
        ((1 as Word) << bit_index(n)) - 1
    } else {
        Word::MAX
    }
}

// Fonction qui applique le mask trouvé par tail_mask
pub(crate) fn mask_tail(words: &mut [Word], n: usize) {
    debug_assert_eq!(words.len(), words_for(n));
    if words.is_empty() {
        return;
    }
    let len = words.len();
    let mask = tail_mask(n);
    words[len - 1] &= mask;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_for_static_test() {
        assert_eq!(words_for(0), 0);
        assert_eq!(words_for(1), 1);
        assert_eq!(words_for(63), 1);
        assert_eq!(words_for(64), 1);
        assert_eq!(words_for(65), 2);
        assert_eq!(words_for(128), 2);
        assert_eq!(words_for(129), 3);
        assert_eq!(words_for(137), 3);
    }

    #[test]
    fn words_for_dynamic_test() {
        for n in 0..500 {
            // Vérifie que le nombre de mot est suffisant pour couvrir le nombre de bit
            let is_big_enough: bool = words_for(n) * NUMBER_OF_BITS >= n;
            assert!(
                is_big_enough,
                "n = {n}: words_for renvoie {} mots",
                words_for(n)
            );

            // Vérifie s'il y a un mot en trop
            if n > 0 {
                let word_sup: bool = (words_for(n) - 1) * NUMBER_OF_BITS < n;
                assert!(word_sup, "n = {n}: words_for renvoie {} mots", words_for(n));
            }
        }
    }

    #[test]
    fn word_index_and_bit_index_test() {
        for i in 0..5000 {
            assert_eq!(word_index(i) * NUMBER_OF_BITS + bit_index(i), i);
        }
    }

    #[test]
    fn bit_mask_test() {
        for i in 0..5000 {
            assert_eq!(bit_mask(i).count_ones(), 1);
            assert_eq!(bit_mask(i).trailing_zeros() as usize, bit_index(i));
        }
    }

    #[test]
    fn tail_mask_test() {
        assert_eq!(tail_mask(0), 0);
        for n in 1..200 {
            let expected = if bit_index(n) == 0 {
                NUMBER_OF_BITS
            } else {
                bit_index(n)
            };
            assert_eq!(tail_mask(n).count_ones() as usize, expected, "n = {n}");

            assert_eq!(tail_mask(n).trailing_ones(), tail_mask(n).count_ones());
        }
    }

    #[test]
    fn mask_tail_test() {
        for n in [0, 1, 63, 64, 65, 127, 128, 129, 137] {
            let mut words = vec![Word::MAX; words_for(n)];
            mask_tail(&mut words, n);

            let mut sum = 0;
            for word in words {
                sum += word.count_ones() as usize;
            }

            assert_eq!(sum, n);
        }
    }
}
