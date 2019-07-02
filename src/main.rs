extern crate libm;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::iter::FromIterator;

//mod matrix;

pub mod matrix;
pub mod layers;
use matrix::MatrixOne;
use matrix::MatrixThree;
use matrix::MatrixTwo;




//学習データを返却する関数
fn preprocess(
    text: &str,
) -> (
    MatrixOne<usize>,
    HashMap<String, usize>,
    HashMap<usize, String>,
) {
    let mut new_id: usize;
    let mut corpus: MatrixOne<usize> = MatrixOne::new();
    let mut word_to_id: HashMap<String, usize> = HashMap::new();
    let mut id_to_word: HashMap<usize, String> = HashMap::new();
    let text = text.to_lowercase().replace(".", " .");
    let words: Vec<String> = text.split(" ").map(|str: &str| str.to_string()).collect();
    for word in words {
        if word_to_id.keys().find(|&x| x == &word).is_none() {
            new_id = word_to_id.len();
            word_to_id.insert(word.clone(), new_id);
            id_to_word.insert(new_id, word.clone());
        }
        corpus.push(word_to_id[&word]);
    }
    return (corpus, word_to_id, id_to_word);
}

fn create_co_matrix(corpus: MatrixOne<usize>, vocab_size: usize) -> MatrixTwo<usize> {
    let windows_size: usize = 1;
    let corpus_size: usize = corpus.len();
    let mut co_matrix: MatrixTwo<usize> = MatrixTwo::<usize>::zeros(vocab_size, vocab_size);

    for (idx, word_id) in corpus.iter().enumerate() {
        for i in 1..windows_size + 1 {
            let left_idx: usize;
            let right_idx: usize = idx + i;
            let word_id: usize = word_id.clone();

            if idx != 0 {
                left_idx = idx - i;
                if let Some(left_word_id) = corpus.get(left_idx) {
                    //co_matrix[[word_id, left_word_id.clone()]] += 1; //word_idとleft_word_idの型が一致していないとだめ＆参照だとダメ（left_word_id）
                    co_matrix[word_id][left_word_id.clone()] += 1; //word_idとleft_word_idの型が一致していないとだめ＆参照だとダメ（left_word_id）
                }
            }

            if right_idx < corpus_size {
                if let Some(right_word_id) = corpus.get(right_idx) {
                    //co_matrix[[word_id, right_word_id.clone()]] += 1;
                    co_matrix[word_id][right_word_id.clone()] += 1;
                }
            }
        }
    }
    co_matrix
}

// 引数（usize）=> mapvでf32にキャスト=>norm_l1()
fn cos_similarity(x: Vec<usize>, y: Vec<usize>) -> f32 {
    let eps = 0.00000001;
    let float_x: Vec<f32> = x.iter().map(|&tmp| tmp as f32).collect();
    let float_y: Vec<f32> = y.iter().map(|&tmp| tmp as f32).collect();
    //let nx = float_x.clone() / float_x.mapv(|tmp: f32| tmp * tmp + eps).norm_l2();
    let norm = |vec: Vec<f32>| -> f32 {
        let vec_pow2 = vec.iter().map(|x| x * x);
        let vec_sum: f32 = vec_pow2.sum();
        vec_sum.sqrt()
    };
    //let nx = float_x.clone() / norm(float_x);
    let nx: Vec<f32> = float_x
        .clone()
        .iter()
        .map(|x| x / (norm(float_x.clone()) + eps))
        .collect();
    let ny: Vec<f32> = float_y
        .clone()
        .iter()
        .map(|y| y / (norm(float_y.clone()) + eps))
        .collect();
    let ret_v: f32 = ny.iter().zip(nx.iter()).map(|(x, y)| x * y).sum();
    ret_v
}

fn most_similar(
    query: String,
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    word_matrix: MatrixTwo<usize>,
    top: usize,
) -> () {
    if word_to_id.keys().find(|&x| x == &query).is_none() {
        println!("{:?} is not found", query);
    }
    println!("[query] {:?}", query);
    let query_id = word_to_id[&query];
    let query_vec = word_matrix[query_id].clone();

    let vocab_size = id_to_word.keys().len();
    let mut similarity: BTreeMap<usize, f32> = BTreeMap::new();
    for i in 0..vocab_size {
        similarity.insert(
            i,
            cos_similarity(word_matrix[i].to_owned(), query_vec.clone()),
        );
    }
    let mut count = 0;
    let mut v = Vec::from_iter(similarity);
    v.sort_by(|&(_, a), &(_, b)| b.partial_cmp(&a).unwrap());
    //上記の書き方でBtreeMapのkeyでsortができる
    for i in v.iter() {
        let (inx, sim) = i;
        if id_to_word[inx] == query {
            continue;
        }
        println!("{:?}: {:?}", id_to_word[inx], sim);
        count += 1;
        if count >= top {
            return;
        }
    }
}

fn create_contexts_target(corpus: MatrixOne<usize>) -> (MatrixTwo<usize>, MatrixOne<usize>) {
    let window_size: i32 = 1;
    let mut contexts: MatrixTwo<usize> = MatrixTwo::new();
    let mut target: MatrixOne<usize> = MatrixOne::new();
    let corpus_len: i32 = corpus.len() as i32;
    for (idx, v) in corpus.iter().enumerate() {
        let idx: i32 = idx as i32;
        if idx != 0 && idx != corpus_len - 1 {
            target.push(v.clone());
        }
    }
    for idx in window_size..corpus_len - window_size {
        let mut cs: Vec<usize> = Vec::new();
        //for文の条件式をisizeにすればマイナスでループも可能
        for t in -window_size..window_size + 1 {
            if t == 0 {
                continue;
            }
            let idx_t: i32 = idx + t;
            let idx_t: usize = idx_t as usize;
            cs.push(corpus[idx_t]);
        }
        contexts.push(cs);
    }
    return (contexts, target);
}

fn main() {
    let vocab_size: usize;
    let target_onehoted: MatrixTwo<usize>;
    let contexts_onehoted: MatrixThree<f32>;
    let text = "You say goodbye and I say hello.";
    let (corpus, word_to_id, id_to_word) = preprocess(text);
    vocab_size = word_to_id.len();

    let C = create_co_matrix(corpus.clone(), vocab_size);
    let mut c0 = vec![0; vocab_size];
    let mut c1 = vec![0; vocab_size];
    if let Some(target_word) = word_to_id.get("you") {
        let target_word = target_word.clone();
        c0 = C[target_word].to_owned();
    }
    if let Some(target_word) = word_to_id.get("goodbye") {
        c1 = C[target_word.clone()].to_owned();
    }
    cos_similarity(c0, c1);
    most_similar("you".to_string(), word_to_id, id_to_word, C, 5);
    let (contexts, target) = create_contexts_target(corpus.clone());
    target_onehoted = target.convert_one_hot(vocab_size);
    contexts_onehoted = contexts.convert_one_hot(vocab_size);
    let zeros_contexts = MatrixTwo::<usize>::zeros(contexts.len_x(), contexts.len_y());
    println!("debug!!");
    zeros_contexts.print_matrix();

}
