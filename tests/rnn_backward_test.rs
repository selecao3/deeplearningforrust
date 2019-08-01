extern crate win_target_sample;
use win_target_sample::*;
use matrix::*;
use rand::Rng;

#[test]
fn backward_test(){
    let mut rng = rand::thread_rng();
    let N:usize = rng.gen_range(0,50);
    let H:usize = rng.gen_range(0,50);
    let D:usize = rng.gen_range(0,50);
    let matrix1:MatrixTwo<f32> = MatrixTwo::rand_generate(H,D);
    let matrix2:MatrixTwo<f32> = MatrixTwo::rand_generate(H,H);
    let matrix3:MatrixTwo<f32> = MatrixTwo::rand_generate(H,N);
    let dh_next = MatrixTwo::rand_generate(H,N);
    let x:MatrixTwo<f32> = MatrixTwo::rand_generate(D, N);
    let h_prev:MatrixTwo<f32> = MatrixTwo::rand_generate(H, N);
    let mut rnn = layers::RNN::init(matrix1,matrix2,matrix3);
    rnn.forward(x, &h_prev);
    let ans = rnn.backward(dh_next);
    println!("ans: {:?}",ans);
    
}