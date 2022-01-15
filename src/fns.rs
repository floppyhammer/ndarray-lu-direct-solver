use ndarray::{Array1, Array2};

fn lu_decompose(a: &Array2<f64>, l: &mut Array2<f64>, u: &mut Array2<f64>) {
    let n = a.dim().0;
    let mut s;

    for i in 0..n{
        l[[i, i]] = 1.0;

        for j in i..n {
            s = 0.0;

            for k in 0..i {
                s += l[[i, k]] * u[[k, j]];
            }

            u[[i, j]] = a[[i, j]] - s;
        }

        for j in i + 1..n {
            s = 0.0;

            for k in 0..i {
                s += l[[j, k]] * u[[k, i]];
            }

            l[[j, i]] = (a[[j, i]] - s) / u[[i, i]];
        }
    }
}

fn lower_tri_solve(a: &Array2<f64>, x: &mut Array1<f64>, b: &Array1<f64>) {
    let n = a.dim().0;
    let mut s;

    x[[0]] = b[[0]] / a[[0, 0]];

    for i in 1..n {
        s = 0.0;

        for j in 0..i {
            s += a[[i, j]] * x[[j]];
        }
        
        x[[i]] = (b[[i]] - s) / a[[i, i]];
    }
}

fn upper_tri_solve(a: &Array2<f64>, x: &mut Array1<f64>, b: &Array1<f64>) {
    let n = a.dim().0;
    let mut s;

    x[[n - 1]] = b[[n - 1]] / a[[n - 1, n - 1]];

    for i in (0..n - 1).rev() {
        s = 0.0;

        for j in i + 1..n {
            s += a[[i, j]] * x[[j]];
        }
            
        x[[i]] = (b[[i]] - s) / a[[i, i]];
    }
}

pub fn solve(a: &Array2<f64>, b: &Array1<f64>, mut x: &mut Array1<f64>) {
    let n = a.dim().0;

    let mut l = Array2::<f64>::zeros((n, n));
    let mut u = Array2::<f64>::zeros((n, n));
    let mut y = Array1::<f64>::zeros(n);

    lu_decompose(&a, &mut l, &mut u);
    lower_tri_solve(&l, &mut y, &b);
    upper_tri_solve(&u, &mut x, &y);
}
