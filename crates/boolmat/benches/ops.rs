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

const SIZES: &[(usize, usize)] = &[(302, 917), (1608, 517), (10729, 8200)];

const PROD_M: usize = 1608;
const PROD_N: usize = 517;

fn label(rows: usize, cols: usize) -> String {
    format!("{rows}x{cols}")
}

fn bench_hamming(c: &mut Criterion) {
    let mut g = c.benchmark_group("hamming");
    for &(rows, cols) in SIZES {
        let a = random(rows, cols, 0.3, 1);
        let b = random(rows, cols, 0.3, 2);
        g.throughput(Throughput::Elements((rows * cols) as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(label(rows, cols)),
            &(a, b),
            |bencher, (a, b)| bencher.iter(|| black_box(a).hamming(black_box(b))),
        );
    }
    g.finish();
}

fn bench_count_ones(c: &mut Criterion) {
    let mut g = c.benchmark_group("count_ones");
    for &(rows, cols) in SIZES {
        let a = random(rows, cols, 0.3, 3);
        g.throughput(Throughput::Elements((rows * cols) as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(label(rows, cols)),
            &a,
            |bencher, a| bencher.iter(|| black_box(a).count_ones()),
        );
    }
    g.finish();
}

fn bench_or_rows(c: &mut Criterion) {
    let mut g = c.benchmark_group("or_rows");
    for &(rows, cols) in SIZES {
        let mut m = random(rows, cols, 0.3, 4);
        g.throughput(Throughput::Elements(cols as u64));
        g.bench_function(BenchmarkId::from_parameter(label(rows, cols)), |bencher| {
            bencher.iter(|| m.or_rows(black_box(0), black_box(1)))
        });
    }
    g.finish();
}

fn bench_product_rank(c: &mut Criterion) {
    let mut g = c.benchmark_group("product_par_rang");
    for &r in &[5usize, 10, 15, 20] {
        let w = random(PROD_M, r, 0.3, 5);
        let h = random(r, PROD_N, 0.3, 6);
        g.throughput(Throughput::Elements((PROD_M * PROD_N) as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(r),
            &(w, h),
            |bencher, (w, h)| bencher.iter(|| black_box(w).product(black_box(h))),
        );
    }
    g.finish();
}

fn bench_product_density(c: &mut Criterion) {
    let r = 10;
    let h = random(r, PROD_N, 0.3, 7);
    let mut g = c.benchmark_group("product_par_densite");
    for &d in &[0.05f64, 0.25, 0.5, 0.9] {
        let w = random(PROD_M, r, d, 8);
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("{d}")),
            &(w, &h),
            |bencher, (w, h)| bencher.iter(|| black_box(w).product(black_box(*h))),
        );
    }
    g.finish();
}

fn bench_product_alloc_vs_reuse(c: &mut Criterion) {
    let mut g = c.benchmark_group("product_alloc_vs_reuse");
    for &r in &[5usize, 10, 15, 20] {
        let w = random(PROD_M, r, 0.3, 5);
        let h = random(r, PROD_N, 0.3, 6);
        g.throughput(Throughput::Elements((PROD_M * PROD_N) as u64));

        g.bench_with_input(
            BenchmarkId::new("alloue", r),
            &(&w, &h),
            |bencher, (w, h)| bencher.iter(|| black_box(*w).product(black_box(*h))),
        );

        let mut out = BitMatrix::zeros(PROD_M, PROD_N);
        g.bench_function(BenchmarkId::new("reutilise", r), |bencher| {
            bencher.iter(|| black_box(&w).product_into(black_box(&h), &mut out))
        });
    }
    g.finish();
}

fn bench_product_reuse_sparse(c: &mut Criterion) {
    let r = 10;
    let h = random(r, PROD_N, 0.3, 7);
    let mut g = c.benchmark_group("product_reuse_par_densite");
    for &d in &[0.05f64, 0.25, 0.5, 0.9] {
        let w = random(PROD_M, r, d, 8);
        let mut out = BitMatrix::zeros(PROD_M, PROD_N);
        g.bench_function(BenchmarkId::from_parameter(format!("{d}")), |bencher| {
            bencher.iter(|| black_box(&w).product_into(black_box(&h), &mut out))
        });
    }
    g.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let mut g = c.benchmark_group("transpose");
    for &(rows, cols) in SIZES {
        let a = random(rows, cols, 0.3, 9);
        g.throughput(Throughput::Elements((rows * cols) as u64));
        g.bench_with_input(
            BenchmarkId::from_parameter(label(rows, cols)),
            &a,
            |bencher, a| bencher.iter(|| black_box(a).transpose()),
        );
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_hamming,
    bench_count_ones,
    bench_or_rows,
    bench_product_rank,
    bench_product_density,
    bench_product_alloc_vs_reuse,
    bench_product_reuse_sparse,
    bench_transpose,
);
criterion_main!(benches);
