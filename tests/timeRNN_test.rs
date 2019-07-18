extern crate win_target_sample;
use win_target_sample::*;
use matrix::*;
use layers::*;
use rand::Rng;

fn test_init() -> (MatrixTwo<f32>,MatrixTwo<f32>,MatrixTwo<f32>){
    let mut rng = rand::thread_rng();
    let N:usize = rng.gen_range(0,50);
    let H:usize = rng.gen_range(0,50);
    let D:usize = rng.gen_range(0,50);
    let matrix1:MatrixTwo<f32> = MatrixTwo::rand_generate(H,D);
    let matrix2:MatrixTwo<f32> = MatrixTwo::rand_generate(H,H);
    let matrix3:MatrixTwo<f32> = MatrixTwo::rand_generate(H,N);
    (matrix1,matrix2,matrix3)
}

#[test]
fn timeRNN_init_test(){
    let (mat1,mat2,mat3) = test_init();
    let timeRNN = TimeRNN::init(mat1,mat2,mat3,true);
    println!("{:?}",timeRNN );
}

#[test]
fn set_state_test(){
    let (mat1,mat2,mat3) = test_init();
    let timeRNN = TimeRNN::init(mat1,mat2,mat3,true);
    let h:MatrixTwo<f32> = MatrixTwo::rand_generate(5, 10);
    &timeRNN.clone().set_state(h);

    //println!("{:?}",timeRNN);
}

#[test]
fn reset_state_test(){
}