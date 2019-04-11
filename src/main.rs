use std::collections::HashMap;

fn preprocess(text: &str) -> (Vec<usize>, HashMap<String, usize>, HashMap<usize, String>) {
    let mut new_id: usize;
    let mut corpus: Vec<usize> = Vec::new();
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

fn create_co_matrix(corpus: Vec<usize>, vocab_size: usize) -> Vec<Vec<usize>> {
    let windows_size: usize = 1;
    let corpus_size: usize = corpus.len();
    let mut co_matrix = vec![vec![0; vocab_size]; vocab_size];

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
    println!("{:?}", co_matrix);
    co_matrix
}

// 引数（usize）=> mapvでf32にキャスト=>norm_l1()
fn cos_similarity(x: Vec<usize>, y: Vec<usize>) -> () {
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
    //let ny = float_y.clone() / float_y.mapv(|tmp: f32| tmp * tmp + eps).norm_l2();
    //let ny = float_y.clone() / float_y.mapv(|tmp: f32| tmp + eps).norm_l2();
    let hoge: f32 = ny.iter().zip(nx.iter()).map(|(x, y)| x * y).sum();
    println!("{:?}", hoge);
}

fn main() {
    let vocab_size: usize;
    let text = "You say goodbye and I say hello.";
    let (corpus, word_to_id, id_to_word) = preprocess(text);
    vocab_size = word_to_id.len();
    println!("{:?}", word_to_id);
    println!("{:?}", id_to_word);
    println!("{:?}", corpus);

    let C = create_co_matrix(corpus, vocab_size);
    let mut c0 = vec![0; vocab_size];
    let mut c1 = vec![0; vocab_size];
    if let Some(target_word) = word_to_id.get("you") {
        let target_word = target_word.clone();
        c0 = C[target_word].to_owned();
        println!("c0");
        println!("{:?}", c0);
    }
    if let Some(target_word) = word_to_id.get("goodbye") {
        c1 = C[target_word.clone()].to_owned();
    }
    cos_similarity(c0, c1);
}
