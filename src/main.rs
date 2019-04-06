use std::collections::HashMap;

fn preprocess(text: &str) -> (HashMap<String, usize>, HashMap<usize, String>) {
    let mut new_id: usize;
    let text = text.to_lowercase().replace(".", " .");
    let words: Vec<String> = text.split(" ").map(|str: &str| str.to_string()).collect();
    let mut word_to_id: HashMap<String, usize> = HashMap::new();
    let mut id_to_word: HashMap<usize, String> = HashMap::new();
    for word in words {
        if word_to_id.keys().find(|&x| x == &word).is_none() {
            new_id = word_to_id.len();
            word_to_id.insert(word.clone(), new_id);
            id_to_word.insert(new_id, word.clone());
        }
    }
    println!("{:?}", word_to_id);
    println!("{:?}", id_to_word);
    return (word_to_id, id_to_word);
}

fn main() {
    let text = "I'm kazuha Ishida.";
    preprocess(text);
}
