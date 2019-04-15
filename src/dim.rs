pub struct Matrix<T>(Vec<T>);
pub struct Matrix_XY<T>(Vec<Vec<T>>);

impl<T> Matrix<T> {
    //Create 1-dimensional matrix
    pub fn new() -> Matrix<T> {
        let v: Vec<T> = Vec::new();
        let mat = Matrix(v);
        mat
    }
    //Create 2-dimensional matrix
    // pub fn new_xy() -> Matrix {
    //     let mat = Matrix();
    //     mat
    // }
    pub fn check(&self) -> bool {
        println!("1-dim");
        true
    }
}

impl<T> Matrix_XY<T> {
    //Create 1-dimensional matrix
    pub fn new() -> Matrix_XY<T> {
        let v: Vec<Vec<T>> = Vec::new();
        let mat = Matrix_XY(v);
        mat
    }
    //Create 2-dimensional matrix
    // pub fn new_xy() -> Matrix {
    //     let mat = Matrix();
    //     mat
    // }
    pub fn check(&self) -> bool {
        println!("2-dim");
        true
    }
}
