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
