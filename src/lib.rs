extern crate libm;
extern crate rand;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::iter::FromIterator;

//mod matrix;

pub mod matrix;
pub mod layers;
pub mod time_layers;
use matrix::MatrixOne;
use matrix::MatrixThree;
use matrix::MatrixTwo;

//RNN Language Model

#[derive(Debug)]
struct SimpleRnnLM {
    layers:()
}