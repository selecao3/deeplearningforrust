extern crate win_target_sample;
use win_target_sample::*;
use matrix::*;
use rand::Rng;


//N:13,H:1,D:41
#[test]
fn rnn_forward_test(){
    let mut rng = rand::thread_rng();
    //0の時、panicになる
    //D==1の時、panicになる
    let N:usize = rng.gen_range(1,5);
    let H:usize = rng.gen_range(1,5);
    let D:usize = rng.gen_range(1,5);
    // let N:usize = 13;
    // let H:usize = 4;
    // let D:usize = 1;
    let matrix1:MatrixTwo<f32> = MatrixTwo::rand_generate(H,D);
    let matrix2:MatrixTwo<f32> = MatrixTwo::rand_generate(H,H);
    let matrix3:MatrixTwo<f32> = MatrixTwo::rand_generate(H,N);
    let mut rnn = layers::RNN::init(matrix1,matrix2,matrix3);
    let x:MatrixTwo<f32> = MatrixTwo::rand_generate(D, N);
    let h_prev:MatrixTwo<f32> = MatrixTwo::rand_generate(H, N);
    let forward_ret = rnn.forward(x, &h_prev);
    if let Some(ans) = rnn.cache{
        println!("{:?}",ans);
    }else {
        panic!("rnn.cacheが保存されていない");
    }
    forward_ret.print_matrix();
}

#[test]
fn sum_mut2_test(){
    let matrix1:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(4,2);
    let matrix2:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(4,2);
    matrix1.sum(&matrix2);
}

#[test]
fn matrix_inverse_test(){
    let x:MatrixTwo<f32> = MatrixTwo::create_matrix(vec![vec![2.0;1];2]);
    let expected_ans:MatrixTwo<f32> = MatrixTwo::create_matrix(vec![vec![2.0;2];1]);
    let ans = x.inverse();
    assert_eq!(ans,expected_ans);
}
#[test]
fn matrix_dot_n_cross_1_test(){
    let x:MatrixTwo<f32> = MatrixTwo::create_matrix(vec![vec![3.0;2];3]);
    let mat2:MatrixTwo<f32> = MatrixTwo::create_matrix(vec![vec![2.0;1];2]);
    let expected_ans = MatrixTwo::create_matrix(vec![vec![12.0;1];3]);
    let ans = x.dot(&mat2);
    assert_eq!(ans,expected_ans);
}
#[test]
fn matrix_dot_n_cross_n_test(){
    let x_1:MatrixTwo<f32> = MatrixTwo::create_matrix(vec![vec![1.0,2.0,3.0];3]);
    let mat2_1:MatrixTwo<f32> = MatrixTwo::create_matrix(vec![vec![1.0,1.0],vec![2.0,2.0],vec![3.0,3.0]]);
    let expected_ans_1 = MatrixTwo::create_matrix(vec![vec![14.0;2];3]);
    let ans_1 = x_1.dot(&mat2_1);
    assert_eq!(ans_1,expected_ans_1);
}