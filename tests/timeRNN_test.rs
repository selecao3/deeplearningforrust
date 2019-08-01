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

fn test_forward_init() -> (MatrixTwo<f32>,MatrixTwo<f32>,MatrixTwo<f32>,MatrixThree<f32>){
    // let mut rng = rand::thread_rng();
    // let N:usize = rng.gen_range(0,50);
    // let H:usize = rng.gen_range(0,50);
    // let D:usize = rng.gen_range(0,50);
    let N:usize = 5;
    let H:usize = 3;
    let D:usize = 5;
    let matrix1:MatrixTwo<f32> = MatrixTwo::rand_generate(H,D);
    let matrix2:MatrixTwo<f32> = MatrixTwo::rand_generate(H,H);
    let matrix3:MatrixTwo<f32> = MatrixTwo::rand_generate(H,N);
    let xs:MatrixThree<f32> = MatrixThree::rand_generate(5, 10,5);
    (matrix1,matrix2,matrix3,xs)
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
    let mut timeRNN = TimeRNN::init(mat1,mat2,mat3,true);
    let h:MatrixTwo<f32> = MatrixTwo::rand_generate(5, 10);
    &timeRNN.set_state(h);
    assert_eq!(timeRNN.h.is_some(),true);

    //println!("{:?}",timeRNN);
}

#[test]
fn reset_state_test(){
    let (mat1,mat2,mat3) = test_init();
    let mut timeRNN = TimeRNN::init(mat1,mat2,mat3,true);
    let h:MatrixTwo<f32> = MatrixTwo::rand_generate(5, 10);
    &timeRNN.reset_state();
    assert_eq!(timeRNN.h.is_none(),true);
}

//精査する必要あり
#[test]
fn timeRNN_forward_test(){
    let (mat1,mat2,mat3,xs) = test_forward_init();
    let mut timeRNN = TimeRNN::init(mat1,mat2,mat3,true);
    timeRNN.forward(xs);
}