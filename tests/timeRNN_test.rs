extern crate win_target_sample;
use win_target_sample::*;
use matrix::{MatrixThree, MatrixTwo};
use time_layers::*;
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
fn test_backward_init() -> (MatrixTwo<f32>,MatrixTwo<f32>,MatrixTwo<f32>,MatrixThree<f32>,MatrixThree<f32>){
    let mut rng = rand::thread_rng();
    let N:usize = rng.gen_range(1,5);
    let H:usize = rng.gen_range(1,5);
    let D:usize = rng.gen_range(1,5);
    let matrix1:MatrixTwo<f32> = MatrixTwo::rand_generate(H,D);
    let matrix2:MatrixTwo<f32> = MatrixTwo::rand_generate(H,H);
    let matrix3:MatrixTwo<f32> = MatrixTwo::rand_generate(H,N);
    let xs:MatrixThree<f32> = MatrixThree::rand_generate(D, 10,N);
    let dhs:MatrixThree<f32> = MatrixThree::rand_generate(H, 10,N);

    (matrix1,matrix2,matrix3,xs,dhs)
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

#[test]
fn timeRNN_forward_test(){
    let (mat1,mat2,mat3,xs) = test_forward_init();
    let mut timeRNN = TimeRNN::init(mat1,mat2,mat3,true);
    let ret = timeRNN.forward(xs);
}
#[test]
fn copy_two_matrix_test(){
    let mut mat3 = MatrixThree::zeros(2, 3, 2);
    //mat3(N,T,H) , target(N,H) になるはず
    let target = MatrixTwo::rand_generate(2, 3);
    for i in 0..mat3.len_y(){
        mat3.copy_two_matrix(&target, i, 1);
    }
}

#[test]
fn timeRNN_backward_test(){
    let (mat1,mat2,mat3,xs,dhs) = test_backward_init();
    let mut timeRNN = TimeRNN::init(mat1,mat2,mat3,true);
    //dhsはzがN, xがHでなければならない
    timeRNN.forward(xs.clone());
    let ret = timeRNN.backward(dhs);
    println!("=================");
    ret.print_matrix();
}

#[test]
fn timeEmbedding_forward_test(){
    let T:usize = 5;
    let W = MatrixTwo::rand_generate(T, 4);
    let mut te = TimeEmbedding::init(W);
    let vec:Vec<Vec<usize>> = vec![vec![1,2,3];T];
    let xs = MatrixTwo::create_matrix(vec);
    let ans = te.forward(xs);
    ans.print_matrix();
}


#[test]
fn timeAffine_forward_test() {
    let N = 4;
    let T = 5;
    let D = 6;
    let W = MatrixTwo::rand_generate(N*T, D);
    let b = MatrixTwo::rand_generate(N*T,N*T);
    let mut time_affine = TimeAffine::init(W, b);
    let x = MatrixThree::rand_generate(D, T, N);
    let ans = time_affine.forward(x);
    ans.print_matrix();
}

#[test]
fn timeAffine_backward_test() {
    let N = 4;
    let T = 5;
    let D = 6;
    let W = MatrixTwo::rand_generate(N*T, D);
    let b = MatrixTwo::rand_generate(N*T,N*T);
    let mut time_affine = TimeAffine::init(W, b);
    let x = MatrixThree::rand_generate(D, T, N);
    time_affine.forward(x);
    let dout = MatrixThree::rand_generate(D, T, N);
    let ans = time_affine.backward(dout);
    ans.print_matrix();
}