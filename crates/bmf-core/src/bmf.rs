use crate::boolls::{BoolLs, Method, Rng};
use boolmat::{BitMatrix, BitVec};

pub struct BmfResult {
    pub w: BitMatrix,
    pub h: BitMatrix,
    pub error: usize,
    pub iterations: usize,
}

fn init_w(x_t: &BitMatrix, m: usize, r: usize, rng: &mut Rng) -> BitMatrix {
    let mut candidates: Vec<usize> = (0..x_t.rows())
        .filter(|&j| x_t.row(j).iter().any(|&w| w != 0))
        .collect();

    let take = r.min(candidates.len());
    for k in 0..take {
        let j = k + rng.below(candidates.len() - k);
        candidates.swap(k, j);
    }

    let mut w_t = BitMatrix::zeros(r, m);
    for k in 0..take {
        w_t.set_row(k, x_t.row(candidates[k]));
    }

    w_t.transpose()
}

pub fn ao_bmf(x: &BitMatrix, r: usize, method: Method, max_iter: usize, seed: u64) -> BmfResult {
    assert!(r >= 1, "r doit valoir au moins 1");

    let (m, n) = (x.rows(), x.cols());
    let mut rng = Rng::new(seed);

    let x_t = x.transpose();
    let mut w = init_w(&x_t, m, r, &mut rng);

    debug_assert_eq!(w.rows(), m);
    debug_assert_eq!(w.cols(), r);

    let mut iterations = 0;
    let mut error = x.count_ones();
    let mut new_error: usize = 0;
    let mut bl;
    let mut h = BitMatrix::zeros(r, n);
    let mut h_t = h.transpose();

    let mut best_w = w.clone();
    let mut best_h = h.clone();

    loop {
        // Update H
        bl = BoolLs::new(&w);
        new_error = 0;
        for j in 0..n {
            let x_col_j = BitVec::from_words(x_t.row(j), m);
            let (h_col_j, error_col_j) = bl.solve(&x_col_j, method);
            h_t.set_row(j, &[h_col_j]);
            new_error += error_col_j;
        }
        h = h_t.transpose();

        //update W
        bl = BoolLs::new(&h_t);
        new_error = 0;
        for i in 0..m {
            let x_row_i = BitVec::from_words(x.row(i), n);
            let (w_row_i, error_row_j) = bl.solve(&x_row_i, method);
            w.set_row(i, &[w_row_i]);
            new_error += error_row_j;
        }

        iterations += 1;

        if new_error >= error {
            break;
        }

        error = new_error;

        best_h = h.clone();
        best_w = w.clone();

        if iterations >= max_iter {
            break;
        }
    }

    BmfResult {
        w: best_w,
        h: best_h,
        error,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_matrix(rows: usize, cols: usize, density: f64, seed: u64) -> BitMatrix {
        let mut rng = Rng::new(seed);
        let threshold = (density * (1u64 << 31) as f64) as u64;
        let mut m = BitMatrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if rng.next() < threshold {
                    m.set(i, j, true);
                }
            }
        }
        m
    }

    const CASES: &[(usize, usize, usize)] = &[
        (10, 8, 2),
        (30, 20, 3),
        (65, 40, 5),
        (100, 129, 4),
        (70, 70, 8),
        (200, 50, 6),
    ];

    #[test]
    fn error_matches_actual_reconstruction() {
        for &(m, n, r) in CASES {
            for &method in &[Method::Zeta, Method::Greedy, Method::GreedyLs] {
                let x = rand_matrix(m, n, 0.3, 1);
                let res = ao_bmf(&x, r, method, 50, 42);

                assert_eq!(res.w.rows(), m, "{method:?} w.rows sur {m}x{n}, r={r}");
                assert_eq!(res.w.cols(), r, "{method:?} w.cols sur {m}x{n}, r={r}");
                assert_eq!(res.h.rows(), r, "{method:?} h.rows sur {m}x{n}, r={r}");
                assert_eq!(res.h.cols(), n, "{method:?} h.cols sur {m}x{n}, r={r}");

                assert_eq!(
                    res.error,
                    x.hamming(&res.w.product(&res.h)),
                    "{method:?} : erreur annoncee incoherente sur {m}x{n}, r={r}"
                );
            }
        }
    }

    #[test]
    fn error_never_exceeds_trivial_bound() {
        for &(m, n, r) in CASES {
            let x = rand_matrix(m, n, 0.3, 2);
            let res = ao_bmf(&x, r, Method::Zeta, 50, 7);
            assert!(
                res.error <= x.count_ones(),
                "{m}x{n}, r={r} : erreur {} > nnz(X) = {}",
                res.error,
                x.count_ones()
            );
            assert!(res.iterations >= 1, "aucune iteration effectuee");
            assert!(res.iterations <= 50, "max_iter depasse");
        }
    }

    #[test]
    fn higher_rank_does_not_hurt() {
        let (m, n) = (60usize, 45usize);
        let x = rand_matrix(m, n, 0.3, 3);

        let best_at = |r: usize| {
            (0..3)
                .map(|s| ao_bmf(&x, r, Method::Zeta, 50, s).error)
                .min()
                .unwrap()
        };

        let mut prev = x.count_ones();
        for r in 1..=6 {
            let e = best_at(r);
            assert!(
                e <= prev,
                "r={r} donne {e}, pire que r={} qui donne {prev}",
                r - 1
            );
            prev = e;
        }
    }

    #[test]
    fn rank_above_column_count_reaches_zero() {
        for &(m, n, r) in &[(30usize, 4usize, 5usize), (65, 6, 8), (20, 3, 3)] {
            let x = rand_matrix(m, n, 0.4, 4);
            let res = ao_bmf(&x, r, Method::Zeta, 50, 0);
            assert_eq!(
                res.error, 0,
                "{m}x{n}, r={r} : toutes les colonnes de X sont dans W"
            );
        }
    }

    #[test]
    fn planted_instance_beats_trivial_but_not_optimal() {
        for &(m, n, r) in &[(40usize, 30usize, 3usize), (60, 50, 4)] {
            let w0 = rand_matrix(m, r, 0.4, 10);
            let h0 = rand_matrix(r, n, 0.4, 11);
            let x = w0.product(&h0);

            assert_eq!(x.hamming(&w0.product(&h0)), 0, "instance mal construite");

            let best = (0..10)
                .map(|s| ao_bmf(&x, r, Method::Zeta, 100, s).error)
                .min()
                .unwrap();

            assert!(
                best < x.count_ones() / 4,
                "{m}x{n}, r={r} : erreur {best}, pas mieux que trivial ({})",
                x.count_ones()
            );
        }
    }

    #[test]
    fn same_seed_gives_same_result() {
        let x = rand_matrix(50, 40, 0.3, 5);
        for &method in &[Method::Zeta, Method::Greedy, Method::GreedyLs] {
            for seed in [0u64, 1, 99] {
                let a = ao_bmf(&x, 4, method, 50, seed);
                let b = ao_bmf(&x, 4, method, 50, seed);
                assert_eq!(a.error, b.error, "{method:?}, graine {seed}");
                assert_eq!(a.w, b.w, "{method:?}, graine {seed} : W differe");
                assert_eq!(a.h, b.h, "{method:?}, graine {seed} : H differe");
            }
        }
    }

    #[test]
    fn seeds_explore_different_optima() {
        let x = rand_matrix(80, 60, 0.3, 6);
        let errors: Vec<usize> = (0..8)
            .map(|s| ao_bmf(&x, 5, Method::Zeta, 50, s).error)
            .collect();
        let distinct: std::collections::HashSet<_> = errors.iter().collect();
        assert!(
            distinct.len() > 1,
            "les 8 graines donnent toutes {} : l'initialisation ne varie pas",
            errors[0]
        );
    }

    #[test]
    fn exact_blocks_beat_greedy_blocks() {
        let x = rand_matrix(100, 80, 0.3, 8);
        let mut wins = 0;
        let mut total = 0;

        for seed in 0..6 {
            let exact = ao_bmf(&x, 5, Method::Zeta, 50, seed).error;
            let greedy = ao_bmf(&x, 5, Method::Greedy, 50, seed).error;
            total += 1;
            if exact <= greedy {
                wins += 1;
            }
        }

        assert!(
            wins * 2 >= total,
            "l'exact ne bat le glouton que {wins} fois sur {total}"
        );
    }

    #[test]
    fn zero_matrix_gives_zero_error() {
        let x = BitMatrix::zeros(20, 15);
        let res = ao_bmf(&x, 3, Method::Zeta, 50, 0);
        assert_eq!(res.error, 0);
        assert_eq!(res.w.count_ones(), 0, "W devrait rester nul");
    }

    #[test]
    fn all_ones_matrix_is_rank_one() {
        let x = BitMatrix::ones(25, 18);
        let res = ao_bmf(&x, 1, Method::Zeta, 50, 0);
        assert_eq!(res.error, 0, "la matrice pleine est de rang booleen 1");
    }

    #[test]
    fn fewer_nonzero_columns_than_rank() {
        let mut x = BitMatrix::zeros(30, 20);
        for i in 0..10 {
            x.set(i, 3, true);
            x.set(i + 5, 7, true);
        }
        let res = ao_bmf(&x, 8, Method::Zeta, 50, 0);
        assert_eq!(
            res.error,
            x.hamming(&res.w.product(&res.h)),
            "erreur incoherente sur l'instance degeneree"
        );
        assert_eq!(res.error, 0, "deux colonnes distinctes, rang booleen <= 2");
    }

    #[test]
    fn single_column_matrix() {
        let x = rand_matrix(40, 1, 0.5, 9);
        let res = ao_bmf(&x, 1, Method::Zeta, 50, 0);
        assert_eq!(res.error, 0, "une seule colonne : rang booleen <= 1");
    }

    #[test]
    #[should_panic(expected = "au moins 1")]
    fn rank_zero_panics() {
        let x = rand_matrix(10, 10, 0.3, 0);
        ao_bmf(&x, 0, Method::Zeta, 50, 0);
    }
}
