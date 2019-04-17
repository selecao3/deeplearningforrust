use std::ops::Index;
use std::ops::IndexMut;
use std::slice::Iter;
use std::slice::SliceIndex;

#[derive(Clone)]
pub struct MatrixOne<T> {
    vec: Vec<T>,
    dim: usize,
}
#[derive(Clone)]
pub struct MatrixTwo<T> {
    vec: Vec<Vec<T>>,
    dim: usize,
}
#[derive(Clone)]
pub struct MatrixThree<T> {
    vec: Vec<Vec<Vec<T>>>,
    dim: usize,
}
//配列へアクセスするためにIndexトレイトをMatrixOneに実装
impl<T, I: SliceIndex<[T]>> Index<I> for MatrixOne<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.vec, index)
    }
}
//配列へアクセスするためにIndexトレイトをMatrixTwoに実装
impl<T: Clone, I: SliceIndex<[Vec<T>]>> Index<I> for MatrixTwo<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.vec, index)
    }
}
//配列へアクセスするためにIndexトレイトをMatrixThreeに実装
impl<T: Clone, I: SliceIndex<[Vec<Vec<T>>]>> Index<I> for MatrixThree<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.vec, index)
    }
}
impl<T, I: SliceIndex<[T]>> IndexMut<I> for MatrixOne<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.vec, index)
    }
}
impl<T: std::clone::Clone, I: SliceIndex<[std::vec::Vec<T>]>> IndexMut<I> for MatrixTwo<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.vec, index)
    }
}
impl<T: Clone, I: SliceIndex<[Vec<Vec<T>>]>> IndexMut<I> for MatrixThree<T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut self.vec, index)
    }
}

//T type for MatrixOne
impl<T> MatrixOne<T> {
    //Create 1-dimensional matrix
    pub fn new() -> MatrixOne<T> {
        let v: Vec<T> = Vec::new();
        MatrixOne { vec: v, dim: 1 }
    }
    pub fn push(&mut self, value: T) -> () {
        self.vec.push(value)
    }
    pub fn len(&self) -> usize {
        self.vec.len()
    }
    pub fn get<I>(&self, value: I) -> Option<&I::Output>
    where
        I: SliceIndex<[T]>,
    {
        self.vec.get(value)
    }
    pub fn iter(&self) -> Iter<T> {
        self.vec.iter()
    }
    pub fn dim_get(&self) -> usize {
        self.dim
    }
}

//T type for MatrixTwo
impl<T> MatrixTwo<T> {
    //Create 1-dimensional matrix
    pub fn new() -> MatrixTwo<T> {
        let v: Vec<Vec<T>> = Vec::new();
        MatrixTwo { vec: v, dim: 2 }
    }
    pub fn iter(&self) -> Iter<Vec<T>> {
        self.vec.iter()
    }
    pub fn dim_get(&self) -> usize {
        self.dim
    }
    pub fn push(&mut self, value: Vec<T>) -> () {
        self.vec.push(value)
    }
    pub fn len(&self) -> usize {
        self.vec.len()
    }
}
// impl<T> MatrixThree<T> {
//     pub fn get_row(&self, row: usize) -> RetType {
//         let got_matrix = self.vec;
//     }
// }

impl MatrixOne<usize> {
    //1次元or二次元で挙動が変わる（前者：2次元のone_hot,後者：3次元のone_hot）
    pub fn convert_one_hot(&self, vocab_size: usize) -> () {
        let N: usize = self.len();
        //8*7行列
        let mut one_hot = MatrixTwo::zeros(vocab_size, N);
        println!("{:?}", one_hot.vec);
        for (idx, word_id) in self.iter().enumerate() {
            println!("idx: {:?}", idx);
            println!("word_id: {:?}", word_id);
            let word_id: usize = word_id.clone();
            //idx行のword_id列に1を代入
            one_hot[idx][word_id] = 1;
        }
        one_hot.print_matrix();
    }
}

impl MatrixTwo<usize> {
    pub fn zeros(x: usize, y: usize) -> MatrixTwo<usize> {
        let v: Vec<Vec<usize>> = vec![vec![0; x]; y];
        MatrixTwo { vec: v, dim: 2 }
    }
    pub fn zeros_square(window_size: usize) -> MatrixTwo<usize> {
        let v: Vec<Vec<usize>> = vec![vec![0; window_size]; window_size];
        MatrixTwo { vec: v, dim: 2 }
    }
    pub fn convert_one_hot(&self, vocab_size: usize) -> () {
        let N: usize = self.len();
        let C: usize = self.vec[0].len();
        let mut one_hot = MatrixThree::zeros(vocab_size, C, N);
        for (idx_0, word_ids) in self.iter().enumerate() {
            for (idx_1, word_id) in word_ids.iter().enumerate() {
                let word_id: usize = word_id.clone();
                //word_id:x, idx_1:y, idx_0:z
                one_hot[idx_0][idx_1][word_id] = 1;
            }
        }
        one_hot.print_matrix();
    }
    pub fn print_matrix(&self) -> () {
        for target in self.vec.iter() {
            println!("{:?}", target);
        }
    }
}

impl MatrixThree<usize> {
    pub fn zeros(x: usize, y: usize, z: usize) -> MatrixThree<usize> {
        let v: Vec<Vec<Vec<usize>>> = vec![vec![vec![0; x]; y]; z];
        MatrixThree { vec: v, dim: 3 }
    }
    pub fn print_matrix(&self) -> () {
        for target in self.vec.iter() {
            println!("{:?}", target);
        }
    }
}
