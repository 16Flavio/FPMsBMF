use boolmat::{BitMatrix, BitVec, Word, NUMBER_OF_BITS};
use std::fmt;
use std::str::FromStr;

pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bit(&mut self) -> bool {
        self.next() & 1 == 1
    }
}

fn ceil_log2(r: usize) -> usize {
    if r <= 1 {
        0
    } else {
        usize::BITS as usize - (r - 1).leading_zeros() as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Method {
    #[default]
    Naive,
    Zeta,
    Greedy,
    GreedyLs,
}

impl Method {
    pub const NAMES: &'static [&'static str] = &["naive", "zeta", "greedy", "greedy-ls"];

    pub fn is_exact(self) -> bool {
        matches!(self, Method::Naive | Method::Zeta)
    }
}

#[derive(Debug, Clone)]
pub struct InvalidMethod(String);

impl fmt::Display for InvalidMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "methode inconnue : '{}'. Valeurs acceptees : {}",
            self.0,
            Method::NAMES.join(", ")
        )
    }
}

impl std::error::Error for InvalidMethod {}

impl FromStr for Method {
    type Err = InvalidMethod;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "naive" => Ok(Method::Naive),
            "zeta" => Ok(Method::Zeta),
            "greedy" => Ok(Method::Greedy),
            "greedy-ls" => Ok(Method::GreedyLs),
            _ => Err(InvalidMethod(s.to_string())),
        }
    }
}

pub struct BoolLs {
    m: usize,
    r: usize,
    patterns: Vec<Word>,
    cnt: Vec<u32>,
    cols: Vec<BitVec>,
}

impl BoolLs {
    pub fn new(w: &BitMatrix) -> Self {
        assert!(
            w.cols() <= NUMBER_OF_BITS,
            "Interdit d'avoir un rang supérieur à la taille d'un mot"
        );

        assert!(w.cols() <= 26, "Sécurité pour ne pas surcharger la mémoire");

        let m = w.rows();
        let r = w.cols();

        let mut patterns = Vec::with_capacity(m);
        for i in 0..m {
            patterns.push(w.row(i)[0]);
        }

        let mut cnt = vec![0u32; 1usize << r];
        for &p in &patterns {
            debug_assert!(p < (1 << r));
            cnt[p as usize] += 1;
        }

        let mut cols = Vec::with_capacity(r);
        let w_t = w.transpose();
        for k in 0..r {
            cols.push(BitVec::from_words(w_t.row(k), m));
        }

        Self {
            m,
            r,
            patterns,
            cnt,
            cols,
        }
    }

    pub fn solve(&self, x: &BitVec, method: Method) -> (Word, usize) {
        self.solve_seeded(x, method, 0)
    }

    pub fn solve_seeded(&self, x: &BitVec, method: Method, seed: u64) -> (Word, usize) {
        assert_eq!(
            x.len(),
            self.m,
            "x a {} bits, W a {} lignes",
            x.len(),
            self.m
        );
        match method {
            Method::Naive => self.solve_naive(x),
            Method::Zeta => self.solve_zeta(x),
            Method::Greedy => self.solve_greedy(x),
            Method::GreedyLs => self.solve_greedy_ls(x, seed),
        }
    }

    fn solve_naive(&self, x: &BitVec) -> (Word, usize) {
        let r = self.r;
        let mut best_cost: usize = self.m + 1;
        let mut best_h = 0;

        for h_cand in 0..(1 << r) {
            let mut cost = 0;
            for i in 0..self.m {
                if x.get(i) != ((self.patterns[i] & h_cand) != 0) {
                    cost += 1;
                }
            }

            if cost < best_cost {
                best_h = h_cand;
                best_cost = cost;
            }
        }

        (best_h, best_cost)
    }

    fn solve_zeta(&self, x: &BitVec) -> (Word, usize) {
        let r = self.r;
        let m = self.m;

        let mut best_cost: usize = m + 1;
        let mut best_h = 0;
        let mut d: Vec<i32> = self.cnt.iter().map(|&c| -(c as i32)).collect();

        for (w, &word) in x.as_words().iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let l = w * NUMBER_OF_BITS + bits.trailing_zeros() as usize;
                debug_assert!(l < m, "bit {l} hors de [0, {})", self.m);

                d[self.patterns[l] as usize] += 2;

                bits &= bits - 1;
            }
        }

        for k in 0..r {
            let bit = 1usize << k;
            for mask in 0..(1usize << r) {
                if mask & bit != 0 {
                    d[mask] += d[mask ^ bit];
                }
            }
        }

        let b = m - x.count_ones();

        let mask = (1 << r) - 1;

        for h in 0..(1 << r) {
            let cost = b as i32 + d[(!h) & mask];
            debug_assert!(cost >= 0, "cout negatif {cost} pour h = {h}");
            if (cost as usize) < best_cost {
                best_cost = cost as usize;
                best_h = h as u64;
            }
        }

        (best_h, best_cost)
    }

    fn solve_greedy(&self, x: &BitVec) -> (Word, usize) {
        let r = self.r;
        let m = self.m;

        let mut best_z = BitVec::zeros(m);
        let mut best_h: Word = 0;
        let mut best_cost: usize = x.count_ones();

        loop {
            let h = best_h;
            let mut upgrade = false;

            let mut temp_h = best_h;
            let mut temp_cost = best_cost;
            let mut temp_z = best_z.clone();

            for k in 0..r {
                let mask = 1 << k;
                if h & mask == 0 {
                    let mut z = best_z.clone();
                    z.or(&self.cols[k]);
                    let cost = z.hamming(x);
                    if cost < temp_cost {
                        temp_cost = cost;
                        temp_h = h | mask;
                        temp_z = z.clone();
                        upgrade = true;
                    }
                }
            }

            if !upgrade {
                break;
            } else {
                best_cost = temp_cost;
                best_h = temp_h;
                best_z = temp_z.clone();
            }
        }

        (best_h, best_cost)
    }

    fn solve_greedy_ls(&self, x: &BitVec, seed: u64) -> (Word, usize) {
        let (h, cost) = self.solve_greedy(x);

        let t = self.r;
        let q = ceil_log2(self.r).max(2).min(self.r);

        let mut rng = Rng::new(seed);
        self.local_search(x, h, cost, t, q, &mut rng)
    }

    fn local_search(
        &self,
        x: &BitVec,
        h: Word,
        cost: usize,
        t: usize,
        q: usize,
        rng: &mut Rng,
    ) -> (Word, usize) {
        if t == 0 {
            return (h, cost);
        }

        let mut h = h;
        let mut cost = cost;

        for _ in 0..t {
            for k in 2..=q {
                let cand = h ^ self.random_mask(k, rng);
                let cand_cost = self.cost_of(x, cand);
                if cand_cost < cost {
                    (h, cost) = self.local_search(x, cand, cand_cost, t - 1, q, rng);
                }
            }
        }

        (h, cost)
    }

    fn random_mask(&self, k: usize, rng: &mut Rng) -> Word {
        debug_assert!(k <= self.r, "poids {k} impossible sur {} bits", self.r);
        let mut u: Word = 0;
        while (u.count_ones() as usize) < k {
            u |= 1 << rng.below(self.r);
        }
        u
    }

    fn cost_of(&self, x: &BitVec, h: Word) -> usize {
        let mut z = BitVec::zeros(self.m);
        let mut bits = h;
        while bits != 0 {
            let k = bits.trailing_zeros() as usize;
            z.or(&self.cols[k]);
            bits &= bits - 1;
        }
        z.hamming(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_from_rows(rows: &[Vec<bool>]) -> BitMatrix {
        let refs: Vec<&[bool]> = rows.iter().map(|r| r.as_slice()).collect();
        BitMatrix::from_bools(&refs)
    }

    fn cost_via_product(w: &BitMatrix, x: &BitVec, h: Word) -> usize {
        let (m, r) = (w.rows(), w.cols());

        let mut hm = BitMatrix::zeros(r, 1);
        for k in 0..r {
            if (h >> k) & 1 == 1 {
                hm.set(k, 0, true);
            }
        }

        let mut xm = BitMatrix::zeros(m, 1);
        for i in 0..m {
            if x.get(i) {
                xm.set(i, 0, true);
            }
        }

        w.product(&hm).hamming(&xm)
    }

    #[test]
    fn histogram_sums_to_m() {
        let mut rng = Rng(1);
        for &(m, r) in &[(1usize, 1usize), (7, 2), (50, 5), (200, 8), (13, 12)] {
            let rows: Vec<Vec<bool>> = (0..m)
                .map(|_| (0..r).map(|_| rng.bit()).collect())
                .collect();
            let solver = BoolLs::new(&matrix_from_rows(&rows));

            assert_eq!(solver.patterns.len(), m, "{m}x{r}");
            assert_eq!(solver.cnt.len(), 1 << r, "{m}x{r}");
            assert_eq!(
                solver.cnt.iter().map(|&c| c as usize).sum::<usize>(),
                m,
                "l'histogramme de {m}x{r} ne totalise pas m"
            );

            let distinct = solver.cnt.iter().filter(|&&c| c > 0).count();
            assert!(distinct <= m.min(1 << r), "{distinct} motifs pour {m}x{r}");
        }
    }

    #[test]
    fn patterns_encode_rows_of_w() {
        let mut rng = Rng(2);
        for &(m, r) in &[(1usize, 1usize), (30, 6), (100, 10)] {
            let rows: Vec<Vec<bool>> = (0..m)
                .map(|_| (0..r).map(|_| rng.bit()).collect())
                .collect();
            let w = matrix_from_rows(&rows);
            let solver = BoolLs::new(&w);

            for i in 0..m {
                let p = solver.patterns[i];
                assert!(p < (1 << r), "motif {p} hors de [0, 2^{r}) en ligne {i}");
                for k in 0..r {
                    assert_eq!(
                        (p >> k) & 1 == 1,
                        w.get(i, k),
                        "ligne {i}, colonne {k} de {m}x{r}"
                    );
                }
            }
        }
    }

    #[test]
    fn kolomvakis_counterexample() {
        let rows = vec![
            vec![true, true],
            vec![true, true],
            vec![true, true],
            vec![true, false],
            vec![true, false],
            vec![false, true],
            vec![false, true],
        ];
        let w = matrix_from_rows(&rows);
        let x = BitVec::from_bools(&[false, false, false, true, true, true, true]);

        let solver = BoolLs::new(&w);

        for (h, expected) in [(0u64, 4usize), (1, 5), (2, 5), (3, 3)] {
            assert_eq!(cost_via_product(&w, &x, h), expected, "cout de h = {h}");
        }

        for method in [Method::Naive, Method::Zeta] {
            let (h, cost) = solver.solve(&x, method);
            assert_eq!(cost, 3, "{method:?} : l'optimum vaut 3");
            assert_eq!(h, 3, "{method:?} :  les deux colonnes");
        }

        assert_eq!(
            solver.solve(&x, Method::Greedy),
            (0, 4),
            "le glouton echoue ici"
        );
    }

    #[test]
    fn exact_column_gives_zero_cost() {
        let rows = vec![
            vec![true, false],
            vec![false, true],
            vec![true, true],
            vec![false, false],
        ];
        let x = BitVec::from_bools(&[false, true, true, false]);
        let solver = BoolLs::new(&matrix_from_rows(&rows));

        for method in [Method::Naive, Method::Zeta] {
            assert_eq!(solver.solve(&x, method), (2, 0), "{method:?}");
        }
    }

    #[test]
    fn zero_x_is_solved_by_zero_h() {
        let rows = vec![vec![true, true], vec![true, false], vec![false, true]];
        let x = BitVec::from_bools(&[false, false, false]);
        let solver = BoolLs::new(&matrix_from_rows(&rows));

        for method in [Method::Naive, Method::Zeta] {
            assert_eq!(solver.solve(&x, method), (0, 0), "{method:?}");
        }
    }

    #[test]
    fn optimum_never_worse_than_all_zero() {
        let mut rng = Rng(3);
        for &(m, r) in &[(20usize, 4usize), (60, 6), (150, 8)] {
            let rows: Vec<Vec<bool>> = (0..m)
                .map(|_| (0..r).map(|_| rng.bit()).collect())
                .collect();
            let bools: Vec<bool> = (0..m).map(|_| rng.bit()).collect();
            let x = BitVec::from_bools(&bools);

            let (_, cost) = BoolLs::new(&matrix_from_rows(&rows)).solve(&x, Method::Naive);
            assert!(
                cost <= x.count_ones(),
                "{m}x{r} : cout {cost} > |x| = {}",
                x.count_ones()
            );
            assert!(cost <= m, "{m}x{r} : cout {cost} > m");
        }
    }

    #[test]
    fn naive_is_exactly_optimal() {
        let mut rng = Rng(4);
        for &(m, r) in &[(1usize, 1usize), (7, 2), (25, 4), (40, 5), (80, 6)] {
            for _ in 0..5 {
                let rows: Vec<Vec<bool>> = (0..m)
                    .map(|_| (0..r).map(|_| rng.bit()).collect())
                    .collect();
                let w = matrix_from_rows(&rows);
                let bools: Vec<bool> = (0..m).map(|_| rng.bit()).collect();
                let x = BitVec::from_bools(&bools);

                let (h, cost) = BoolLs::new(&w).solve(&x, Method::Naive);

                assert!(h < (1 << r), "{m}x{r} : h = {h} hors domaine");
                assert_eq!(
                    cost,
                    cost_via_product(&w, &x, h),
                    "{m}x{r} : le cout annonce pour h = {h} est faux"
                );

                for cand in 0..(1u64 << r) {
                    let c = cost_via_product(&w, &x, cand);
                    assert!(
                        cost <= c,
                        "{m}x{r} : h = {cand} coute {c} < {cost} annonce comme optimal"
                    );
                }
            }
        }
    }

    #[test]
    fn ties_are_broken_by_smallest_h() {
        let rows = vec![vec![true, true], vec![false, false], vec![true, true]];
        let x = BitVec::from_bools(&[true, false, true]);
        let solver = BoolLs::new(&matrix_from_rows(&rows));

        for method in [Method::Naive, Method::Zeta] {
            assert_eq!(solver.solve(&x, method), (1, 0), "{method:?}");
        }
    }

    #[test]
    fn method_parsing() {
        assert_eq!("naive".parse::<Method>().unwrap(), Method::Naive);
        assert_eq!("  NAIVE ".parse::<Method>().unwrap(), Method::Naive);
        assert_eq!(Method::default(), Method::Naive);
        assert_eq!("zeta".parse::<Method>().unwrap(), Method::Zeta);
        assert!("zeta2".parse::<Method>().is_err());

        for name in Method::NAMES {
            assert!(
                name.parse::<Method>().is_ok(),
                "NAMES annonce '{name}' que from_str refuse"
            );
        }
    }

    #[test]
    #[should_panic(expected = "bits")]
    fn mismatched_length_panics() {
        let rows = vec![vec![true, false], vec![false, true]];
        let x = BitVec::from_bools(&[true, false, true]);
        BoolLs::new(&matrix_from_rows(&rows)).solve(&x, Method::Naive);
    }

    #[test]
    fn zeta_matches_naive() {
        let mut rng = Rng(5);
        for &(m, r) in &[
            (1usize, 1usize),
            (7, 2),
            (25, 4),
            (40, 5),
            (80, 6),
            (200, 8),
            (500, 10),
        ] {
            for _ in 0..5 {
                let rows: Vec<Vec<bool>> = (0..m)
                    .map(|_| (0..r).map(|_| rng.bit()).collect())
                    .collect();
                let solver = BoolLs::new(&matrix_from_rows(&rows));
                let bools: Vec<bool> = (0..m).map(|_| rng.bit()).collect();
                let x = BitVec::from_bools(&bools);

                assert_eq!(
                    solver.solve(&x, Method::Zeta),
                    solver.solve(&x, Method::Naive),
                    "divergence sur {m}x{r}"
                );
            }
        }
    }

    #[test]
    fn zeta_matches_naive_on_extreme_densities() {
        let mut rng = Rng(6);
        for &(m, r) in &[(30usize, 5usize), (100, 8)] {
            let rows: Vec<Vec<bool>> = (0..m)
                .map(|_| (0..r).map(|_| rng.bit()).collect())
                .collect();
            let solver = BoolLs::new(&matrix_from_rows(&rows));

            for x in [BitVec::zeros(m), BitVec::ones(m)] {
                assert_eq!(
                    solver.solve(&x, Method::Zeta),
                    solver.solve(&x, Method::Naive),
                    "densite extreme sur {m}x{r}"
                );
            }
        }
    }

    #[test]
    fn heuristics_report_consistent_costs() {
        let mut rng = Rng(7);
        for &(m, r) in &[(20usize, 4usize), (60, 6), (150, 8), (300, 10)] {
            let rows: Vec<Vec<bool>> = (0..m)
                .map(|_| (0..r).map(|_| rng.bit()).collect())
                .collect();
            let w = matrix_from_rows(&rows);
            let solver = BoolLs::new(&w);
            let bools: Vec<bool> = (0..m).map(|_| rng.bit()).collect();
            let x = BitVec::from_bools(&bools);

            for method in [Method::Greedy, Method::GreedyLs] {
                let (h, cost) = solver.solve(&x, method);
                assert!(
                    h < (1 << r),
                    "{method:?} : h = {h} hors domaine sur {m}x{r}"
                );
                assert_eq!(
                    cost,
                    cost_via_product(&w, &x, h),
                    "{method:?} : cout annonce faux pour h = {h} sur {m}x{r}"
                );
            }
        }
    }

    #[test]
    fn heuristics_are_sandwiched() {
        let mut rng = Rng(8);
        for &(m, r) in &[(20usize, 4usize), (60, 6), (150, 8), (300, 10)] {
            for _ in 0..3 {
                let rows: Vec<Vec<bool>> = (0..m)
                    .map(|_| (0..r).map(|_| rng.bit()).collect())
                    .collect();
                let solver = BoolLs::new(&matrix_from_rows(&rows));
                let bools: Vec<bool> = (0..m).map(|_| rng.bit()).collect();
                let x = BitVec::from_bools(&bools);

                let exact = solver.solve(&x, Method::Zeta).1;
                let ls = solver.solve(&x, Method::GreedyLs).1;
                let greedy = solver.solve(&x, Method::Greedy).1;

                assert!(exact <= ls, "{m}x{r} : exact {exact} > greedy-ls {ls}");
                assert!(ls <= greedy, "{m}x{r} : greedy-ls {ls} > greedy {greedy}");
                assert!(
                    greedy <= x.count_ones(),
                    "{m}x{r} : greedy {greedy} > |x| = {}",
                    x.count_ones()
                );
            }
        }
    }

    #[test]
    fn greedy_ls_is_reproducible() {
        let mut rng = Rng(9);
        let (m, r) = (200usize, 8usize);
        let rows: Vec<Vec<bool>> = (0..m)
            .map(|_| (0..r).map(|_| rng.bit()).collect())
            .collect();
        let solver = BoolLs::new(&matrix_from_rows(&rows));
        let bools: Vec<bool> = (0..m).map(|_| rng.bit()).collect();
        let x = BitVec::from_bools(&bools);

        for seed in [0u64, 1, 42, 12345] {
            let a = solver.solve_seeded(&x, Method::GreedyLs, seed);
            let b = solver.solve_seeded(&x, Method::GreedyLs, seed);
            assert_eq!(a, b, "graine {seed} : resultats differents");
        }
    }
}
