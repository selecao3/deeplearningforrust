mod matrix;
use matrix::MatrixTwo;
struct MatMul{
    params:usize,
    grads:MatrixTwo<T>,
    x:f32
}

impl MatMul {
    fn init(W:MatrixTwo<T>) -> RetType {
        MatMul{
            W,
        }

    }
}