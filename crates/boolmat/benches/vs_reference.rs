use boolmat::reference::RefMatrix;
use boolmat::BitMatrix;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

struct Rng(u64);

impl Rng {
    fn next_u31(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn random(rows: usize, cols: usize, density: f64, seed: u64) -> BitMatrix {
    let mut rng = Rng(seed);
    let threshold = (density * (1u64 << 31) as f64) as u64;
    let mut m = BitMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            if rng.next_u31() < threshold {
                m.set(i, j, true);
            }
        }
    }
    m
}

const SIZES: &[(usize, usize)] = &[(302, 917), (1608, 517), (3000, 3000), (10729, 8200)];

const PROD_M: usize = 1608;
const PROD_N: usize = 517;

fn bench_hamming(c: &mut Criterion) {
    let mut g = c.benchmark_group("hamming_bitpacke_vs_naif");

    for &(rows, cols) in SIZES {
        let a = random(rows, cols, 0.3, 1);
        let b = random(rows, cols, 0.3, 2);
        let ra = RefMatrix::from_bit(&a);
        let rb = RefMatrix::from_bit(&b);

        let n = rows * cols;
        g.throughput(Throughput::Elements(n as u64));

        g.bench_function(BenchmarkId::new("BitMatrix", n), |bencher| {
            bencher.iter(|| black_box(&a).hamming(black_box(&b)))
        });
        g.bench_function(BenchmarkId::new("RefMatrix", n), |bencher| {
            bencher.iter(|| black_box(&ra).hamming(black_box(&rb)))
        });
    }
    g.finish();
}

fn bench_count_ones(c: &mut Criterion) {
    let mut g = c.benchmark_group("count_ones_bitpacke_vs_naif");

    for &(rows, cols) in SIZES {
        let a = random(rows, cols, 0.3, 3);
        let ra = RefMatrix::from_bit(&a);

        let n = rows * cols;
        g.throughput(Throughput::Elements(n as u64));

        g.bench_function(BenchmarkId::new("BitMatrix", n), |bencher| {
            bencher.iter(|| black_box(&a).count_ones())
        });
        g.bench_function(BenchmarkId::new("RefMatrix", n), |bencher| {
            bencher.iter(|| black_box(&ra).count_ones())
        });
    }
    g.finish();
}

fn bench_product(c: &mut Criterion) {
    let mut g = c.benchmark_group("product_bitpacke_vs_naif");
    g.sample_size(10);

    for &r in &[5usize, 10, 15, 20] {
        let w = random(PROD_M, r, 0.3, 5);
        let h = random(r, PROD_N, 0.3, 6);
        let rw = RefMatrix::from_bit(&w);
        let rh = RefMatrix::from_bit(&h);

        g.throughput(Throughput::Elements((PROD_M * PROD_N) as u64));

        g.bench_function(BenchmarkId::new("BitMatrix", r), |bencher| {
            bencher.iter(|| black_box(&w).product(black_box(&h)))
        });
        g.bench_function(BenchmarkId::new("RefMatrix", r), |bencher| {
            bencher.iter(|| black_box(&rw).product(black_box(&rh)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_hamming, bench_count_ones, bench_product);
criterion_main!(benches);
