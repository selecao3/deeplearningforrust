extern crate win_target_sample;
use win_target_sample::*;
use matrix::{MatrixThree, MatrixTwo};
use layers::*;
use rand::Rng;

fn test_init() -> (MatrixTwo<f32>,MatrixTwo<f32>,MatrixTwo<f32>){
    let mut rng = rand::thread_rng();
    let N:usize = rng.gen_range(1,50);
    let H:usize = rng.gen_range(1,50);
    let D:usize = rng.gen_range(1,50);
    let matrix1:MatrixTwo<f32> = MatrixTwo::rand_generate(H,D);
    let matrix2:MatrixTwo<f32> = MatrixTwo::rand_generate(H,H);
    let matrix3:MatrixTwo<f32> = MatrixTwo::rand_generate(H,N);
    (matrix1,matrix2,matrix3)
}

fn test_forward_init() -> (MatrixTwo<f32>,MatrixTwo<f32>,MatrixTwo<f32>,MatrixThree<f32>){
    let mut rng = rand::thread_rng();
    let N:usize = rng.gen_range(1,5);
    let H:usize = rng.gen_range(1,5);
    let D:usize = rng.gen_range(1,5);
    let matrix1:MatrixTwo<f32> = MatrixTwo::rand_generate(H,D);
    let matrix2:MatrixTwo<f32> = MatrixTwo::rand_generate(H,H);
    let matrix3:MatrixTwo<f32> = MatrixTwo::rand_generate(H,N);
    let xs:MatrixThree<f32> = MatrixThree::rand_generate(D, 10,N);

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
    let ret = timeRNN.forward(xs);
    ret.print_matrix();
}
#[test]
fn copy_two_matrix_test(){
    let mut mat3 = MatrixThree::zeros(2, 3, 2);
    //mat3(N,T,H) , target(N,H) になるはず
    let target = MatrixTwo::rand_generate(2, 3);
    mat3.print_matrix();
    println!("================");
    for i in 0..mat3.len_y(){
        mat3.copy_two_matrix(&target, i, 1);
    }

    mat3.print_matrix();
}