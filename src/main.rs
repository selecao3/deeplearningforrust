use std::collections::HashMap;

fn preprocess(text: String) {
    let mut new_id: usize;
    let text = text.to_lowercase();
    let text = text.replace(".", " .");
    let words: Vec<&str> = text.split(" ").collect();
    let mut word_to_id: HashMap<&str, usize> = HashMap::new();
    let mut id_to_word: HashMap<usize, &str> = HashMap::new();
    for word in words {
        if word_to_id.keys().find(|&&x| x == word).is_none() {
            new_id = word_to_id.len();
            word_to_id.insert(word, new_id);
            id_to_word.insert(new_id, word);
        }
    }
    println!("{:?}", word_to_id);
    println!("{:?}", id_to_word);
}

fn main() {
    let text = "I'm kazuha Ishida.";
    preprocess(text.to_string());
}
