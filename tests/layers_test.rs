extern crate win_target_sample;
use win_target_sample::*;
use matrix::{MatrixBase,MatrixThree, MatrixTwo};
use time_layers::*;
use layers::*;
use rand::Rng;

#[test]
fn sigmoid_test() {
    let sigmoid = Sigmoid::init();
    let tmp_vec = vec![vec![1.0,2.0];2];
    // let x = MatrixTwo::rand_generate(1, 4);
    let x = MatrixTwo::create_matrix(tmp_vec);
    let ans = sigmoid.forward(x);
    ans.print_matrix();
}

#[test]
fn affine_test() {
    let N = 4;
    let H = 5;
    let W = MatrixTwo::rand_generate(N, H);
    let b = MatrixTwo::rand_generate(N, N);
    let x = MatrixTwo::rand_generate(H, N);
    let affine = Affine::init(W, b);
    let ans = affine.forward(x);
    ans.print_matrix();
}

#[test]
fn embedding_forward_test() {
    //Wの行（idx内の値)を取得し、ansに格納
    let W = MatrixTwo::rand_generate(4, 4);
    let mut e = Embedding::init(W);
    let idx = vec![1,1,1];
    let ans = e.forward(idx);
    ans.print_matrix();
}

#[test]
fn embedding_backward_test() {
    //Wの行（idx内の値)を取得し、ansに格納
    let row = 10;
    let W = MatrixTwo::rand_generate(4, row);
    let mut e = Embedding::init(W);
    let dout = MatrixTwo::rand_generate(4, row);
    //idxはrow以下の列数でないといけない
    let idx = vec![1;row];
    e.forward(idx);
    e.backward(dout);
}
