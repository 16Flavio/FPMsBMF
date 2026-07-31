use bmf_core::{BoolLs, Method};
use boolmat::{BitMatrix, BitVec};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn draw(&mut self, threshold: u64) -> bool {
        self.next() < threshold
    }
}

fn threshold(density: f64) -> u64 {
    (density * (1u64 << 31) as f64) as u64
}

fn random_w(m: usize, r: usize, density: f64, seed: u64) -> BitMatrix {
    let mut rng = Rng(seed);
    let t = threshold(density);
    let mut w = BitMatrix::zeros(m, r);
    for i in 0..m {
        for k in 0..r {
            if rng.draw(t) {
                w.set(i, k, true);
            }
        }
    }
    w
}

fn random_x(m: usize, density: f64, seed: u64) -> BitVec {
    let mut rng = Rng(seed);
    let t = threshold(density);
    let bools: Vec<bool> = (0..m).map(|_| rng.draw(t)).collect();
    BitVec::from_bools(&bools)
}

const M: usize = 1608;
const DENSITY: f64 = 0.3;

fn bench_par_rang(c: &mut Criterion) {
    let mut g = c.benchmark_group("boolls_par_rang");
    g.sample_size(10);

    for &r in &[4usize, 6, 8, 10, 12, 13] {
        let w = random_w(M, r, DENSITY, 1);
        let x = random_x(M, DENSITY, 2);
        let solver = BoolLs::new(&w);

        g.bench_function(BenchmarkId::new("naive", r), |b| {
            b.iter(|| black_box(&solver).solve(black_box(&x), Method::Naive))
        });
        g.bench_function(BenchmarkId::new("zeta", r), |b| {
            b.iter(|| black_box(&solver).solve(black_box(&x), Method::Zeta))
        });
    }
    g.finish();
}

fn bench_par_m(c: &mut Criterion) {
    let r = 10;
    let mut g = c.benchmark_group("boolls_par_m");
    g.sample_size(10);

    for &m in &[100usize, 500, 2000, 10000] {
        let w = random_w(m, r, DENSITY, 3);
        let x = random_x(m, DENSITY, 4);
        let solver = BoolLs::new(&w);

        g.bench_function(BenchmarkId::new("naive", m), |b| {
            b.iter(|| black_box(&solver).solve(black_box(&x), Method::Naive))
        });
        g.bench_function(BenchmarkId::new("zeta", m), |b| {
            b.iter(|| black_box(&solver).solve(black_box(&x), Method::Zeta))
        });
    }
    g.finish();
}

fn bench_zeta_grand_rang(c: &mut Criterion) {
    let mut g = c.benchmark_group("zeta_grand_rang");
    g.sample_size(10);

    for &r in &[14usize, 16, 18, 20, 22] {
        let w = random_w(M, r, DENSITY, 5);
        let x = random_x(M, DENSITY, 6);
        let solver = BoolLs::new(&w);

        g.bench_function(BenchmarkId::from_parameter(r), |b| {
            b.iter(|| black_box(&solver).solve(black_box(&x), Method::Zeta))
        });
    }
    g.finish();
}

fn bench_preprocessing(c: &mut Criterion) {
    let mut g = c.benchmark_group("boolls_new");

    for &(m, r) in &[(1608usize, 10usize), (1608, 16), (1608, 20), (10000, 10)] {
        let w = random_w(m, r, DENSITY, 7);
        g.bench_function(BenchmarkId::from_parameter(format!("{m}x{r}")), |b| {
            b.iter(|| BoolLs::new(black_box(&w)))
        });
    }
    g.finish();
}

fn bench_full_h_update(c: &mut Criterion) {
    let (m, n, r) = (1608usize, 517usize, 10usize);
    let w = random_w(m, r, DENSITY, 8);
    let solver = BoolLs::new(&w);
    let columns: Vec<BitVec> = (0..n)
        .map(|j| random_x(m, DENSITY, 100 + j as u64))
        .collect();

    let mut g = c.benchmark_group("update_complet_de_H");
    g.sample_size(10);
    g.bench_function("zeta", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for x in &columns {
                total += black_box(&solver).solve(x, Method::Zeta).1;
            }
            total
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_par_rang,
    bench_par_m,
    bench_zeta_grand_rang,
    bench_preprocessing,
    bench_full_h_update,
);
criterion_main!(benches);
