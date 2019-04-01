use std::collections::HashMap;

fn preprocess(text: String) {
    let text = text.to_lowercase();
    let text = text.replace(".", " .");
    let words: Vec<&str> = text.split(" ").collect();
    let mut word_to_id: HashMap<&str, i32> = HashMap::new();
    let mut id_to_word: HashMap<i32, &str> = HashMap::new();
    for word in words.iter() {
        if word != word_to_id.keys() {
            unimplemented!();
        }
    }
}

fn main() {
    let text = "I'm kazuha Ishida.";
    preprocess(text.to_string());
}
