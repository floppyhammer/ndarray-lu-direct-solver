use rand::prelude::*;
use ndarray::{Array1, Array2};
use std::time::Instant;

mod fns;

fn display_array(array: &Array2<f64>) {
    for i in 0..array.dim().0 {
        for j in 0..array.dim().1 {
            print!("{} ", array[[i, j]]);
        }
        println!();
    }
}

fn randomize_array2(array: &mut Array2<f64>) {
    let mut rng = rand::thread_rng();

    for i in 0..array.dim().0 {
        for j in 0..array.dim().1 {
            array[[i, j]] = rng.gen();
        }
    }
}

fn main() {
    // Matrix size.
    let n = 50;

    let mut a = Array2::<f64>::zeros((n, n));
    let mut b = Array1::<f64>::zeros(n);
    let mut x = Array1::<f64>::zeros(n);

    randomize_array2(&mut a);

    let mut rng = rand::thread_rng();
    for i in 0..b.dim() {
        b[[i]] = rng.gen();
    }

    let now = Instant::now();

    fns::solve(&a, &b, &mut x);

    let elapsed = now.elapsed();
    let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
    
    println!("Solution: {}", x);
    println!("Time cost (in seconds): {}", sec);
}
