use crate::prelude::*;
use ruint::aliases::U64;
use proptest::prelude::*;

pub fn group(criterion: &mut Criterion) {
    // Test different scenarios where Montgomery multiplication might perform differently
    
    // Scenario 1: Random full-size values (original benchmark)
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);

        let name = format!("modexp_random/{BITS}");
        let mut group = criterion.benchmark_group(&name);
        group.sample_size(30); // Reduce sample size for expensive operations

        // Regular pow_mod
        group.bench_function("pow_mod", |bencher| {
            bencher.iter_batched(
                || {
                    let runner = &mut TestRunner::deterministic();
                    let base = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    let exp = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    let mut modulus = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    modulus |= Uint::from(1u64); // Make sure modulus is odd
                    (base, exp, modulus)
                },
                |(base, exp, modulus)| {
                    black_box(base.pow_mod(exp, modulus))
                },
                BatchSize::SmallInput,
            )
        });

        // Optimized pow_mod_redc
        group.bench_function("pow_mod_redc", |bencher| {
            bencher.iter_batched(
                || {
                    let runner = &mut TestRunner::deterministic();
                    let base = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    let exp = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    let mut modulus = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    modulus |= Uint::from(1u64); // Make sure modulus is odd
                    let inv = U64::from(modulus.as_limbs()[0]).inv_ring().unwrap();
                    let inv = (-inv).as_limbs()[0];
                    (base, exp, modulus, inv)
                },
                |(base, exp, modulus, inv)| {
                    black_box(base.pow_mod_redc(exp, modulus, inv))
                },
                BatchSize::SmallInput,
            )
        });

        group.finish();
    });
    
    // Scenario 2: Small exponents (where setup cost dominates)
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);

        let name = format!("modexp_small_exp/{BITS}");
        let mut group = criterion.benchmark_group(&name);

        group.bench_function("pow_mod", |bencher| {
            bencher.iter_batched(
                || {
                    let runner = &mut TestRunner::deterministic();
                    let base = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    let exp = Uint::<BITS, LIMBS>::from(u32::arbitrary().new_tree(runner).unwrap().current()); // Small exponent
                    let mut modulus = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    modulus |= Uint::from(1u64);
                    (base, exp, modulus)
                },
                |(base, exp, modulus)| {
                    black_box(base.pow_mod(exp, modulus))
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function("pow_mod_redc", |bencher| {
            bencher.iter_batched(
                || {
                    let runner = &mut TestRunner::deterministic();
                    let base = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    let exp = Uint::<BITS, LIMBS>::from(u32::arbitrary().new_tree(runner).unwrap().current()); // Small exponent
                    let mut modulus = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                    modulus |= Uint::from(1u64);
                    let inv = U64::from(modulus.as_limbs()[0]).inv_ring().unwrap();
                    let inv = (-inv).as_limbs()[0];
                    (base, exp, modulus, inv)
                },
                |(base, exp, modulus, inv)| {
                    black_box(base.pow_mod_redc(exp, modulus, inv))
                },
                BatchSize::SmallInput,
            )
        });

        group.finish();
    });
    
    // Scenario 3: Repeated operations with same modulus (amortized setup)
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);

        let name = format!("modexp_amortized/{BITS}");
        let mut group = criterion.benchmark_group(&name);

        // Pre-compute a fixed modulus and inv
        let runner = &mut TestRunner::deterministic();
        let mut fixed_modulus = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
        fixed_modulus |= Uint::from(1u64);
        let fixed_inv = U64::from(fixed_modulus.as_limbs()[0]).inv_ring().unwrap();
        let fixed_inv = (-fixed_inv).as_limbs()[0];

        group.bench_function("pow_mod_10x", |bencher| {
            bencher.iter_batched(
                || {
                    let runner = &mut TestRunner::deterministic();
                    // Generate 10 different base/exp pairs
                    let pairs: Vec<_> = (0..10).map(|_| {
                        let base = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                        let exp = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                        (base, exp)
                    }).collect();
                    pairs
                },
                |pairs| {
                    let mut result = Uint::<BITS, LIMBS>::ZERO;
                    for (base, exp) in pairs {
                        result = result.wrapping_add(base.pow_mod(exp, fixed_modulus));
                    }
                    black_box(result)
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function("pow_mod_redc_10x", |bencher| {
            bencher.iter_batched(
                || {
                    let runner = &mut TestRunner::deterministic();
                    // Generate 10 different base/exp pairs
                    let pairs: Vec<_> = (0..10).map(|_| {
                        let base = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                        let exp = Uint::<BITS, LIMBS>::arbitrary().new_tree(runner).unwrap().current();
                        (base, exp)
                    }).collect();
                    pairs
                },
                |pairs| {
                    let mut result = Uint::<BITS, LIMBS>::ZERO;
                    for (base, exp) in pairs {
                        result = result.wrapping_add(base.pow_mod_redc(exp, fixed_modulus, fixed_inv));
                    }
                    black_box(result)
                },
                BatchSize::SmallInput,
            )
        });

        group.finish();
    });
}
