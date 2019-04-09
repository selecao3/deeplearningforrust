extern crate ndarray;
extern crate ndarray_linalg;
use ndarray::Array;
use ndarray::Array2;
use ndarray_linalg::*;
use std::collections::HashMap;

fn preprocess(
    text: &str,
) -> (
    Array<usize, ndarray::Dim<[usize; 1]>>,
    HashMap<String, usize>,
    HashMap<usize, String>,
) {
    let mut new_id: usize;
    let mut id_vec: Vec<usize> = Vec::new();
    let mut word_to_id: HashMap<String, usize> = HashMap::new();
    let mut id_to_word: HashMap<usize, String> = HashMap::new();
    let text = text.to_lowercase().replace(".", " .");
    let words: Vec<String> = text.split(" ").map(|str: &str| str.to_string()).collect();
    for word in words {
        if word_to_id.keys().find(|&x| x == &word).is_none() {
            new_id = word_to_id.len();
            word_to_id.insert(word.clone(), new_id);
            id_to_word.insert(new_id, word);
            id_vec.push(new_id);
        }
    }
    let corpus = Array::from_vec(id_vec);
    return (corpus, word_to_id, id_to_word);
}

fn create_co_matrix(
    corpus: Array<usize, ndarray::Dim<[usize; 1]>>,
    vocab_size: usize,
) -> Array<usize, ndarray::Dim<[usize; 2]>> {
    let windows_size: usize = 1;
    let corpus_size: usize = corpus.len();
    let mut co_matrix = Array2::<usize>::zeros((vocab_size, vocab_size));

    for (idx, word_id) in corpus.indexed_iter() {
        for i in 1..windows_size + 1 {
            let left_idx: usize;
            let right_idx: usize = idx + i;
            let word_id: usize = word_id.clone();

            if idx != 0 {
                left_idx = idx - i;
                if let Some(left_word_id) = corpus.get(left_idx) {
                    co_matrix[[word_id, left_word_id.clone()]] += 1; //word_idとleft_word_idの型が一致していないとだめ＆参照だとダメ（left_word_id）
                }
            }

            if right_idx < corpus_size {
                if let Some(right_word_id) = corpus.get(right_idx) {
                    co_matrix[[word_id, right_word_id.clone()]] += 1;
                }
            }
        }
    }
    println!("{:?}", co_matrix);
    co_matrix
}

// 引数（usize）=> mapvでf32にキャスト=>norm_l1()
fn cos_similarity(
    x: Array<usize, ndarray::Dim<[usize; 1]>>,
    y: Array<usize, ndarray::Dim<[usize; 1]>>,
) -> () {
    let eps = 0.00000001;
    let float_x = x.mapv(|tmp: usize| tmp as f32);
    let float_y = y.mapv(|tmp: usize| tmp as f32);
    //let nx = float_x.clone() / float_x.mapv(|tmp: f32| tmp * tmp + eps).norm_l2();
    let nx = float_x.clone() / float_x.mapv(|tmp: f32| tmp + eps).norm_l2();
    //let ny = float_y.clone() / float_y.mapv(|tmp: f32| tmp * tmp + eps).norm_l2();
    let ny = float_y.clone() / float_y.mapv(|tmp: f32| tmp + eps).norm_l2();
    let hoge = ny.dot(&nx);
    println!("{:?}", hoge);
}

fn main() {
    let vocab_size: usize;
    let text = "You say goodbye and I say hello.";
    let (corpus, word_to_id, id_to_word) = preprocess(text);
    vocab_size = word_to_id.len();
    println!("{:?}", word_to_id);
    println!("{:?}", id_to_word);

    let C = create_co_matrix(corpus, vocab_size);
    let mut c0 = Array::default(1);
    let mut c1 = Array::default(1);
    if let Some(target_word) = word_to_id.get("you") {
        c0 = C.row(target_word.clone()).to_owned();
        println!("c0");
        println!("{:?}", c0);
    }
    if let Some(target_word) = word_to_id.get("goodbye") {
        c1 = C.row(target_word.clone()).to_owned();
    }
    cos_similarity(c0, c1);
}
