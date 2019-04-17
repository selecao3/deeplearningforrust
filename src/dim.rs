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
//配列へアクセスするためにIndexトレイトをMatrixOneに実装
impl<T, I: SliceIndex<[T]>> Index<I> for MatrixOne<T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&self.vec, index)
    }
}
//配列へアクセスするためにIndexトレイトをMatrixTwoに実装
impl<T: std::clone::Clone, I: SliceIndex<[std::vec::Vec<T>]>> Index<I> for MatrixTwo<T> {
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
    //1次元or二次元で挙動が変わる（前者：2次元のone_hot,後者：3次元のone_hot）
    pub fn convert_one_hot(&self, vocab_size: usize) -> () {
        let N: usize = self.len();
        let one_hot = MatrixTwo::zeros(N, vocab_size);
        for (idx, word_id) in self.iter().enumerate() {
            one_hot[idx][word_id] = 1;
        }
    }
}

impl<T> MatrixTwo<T> {
    //Create 1-dimensional matrix
    pub fn new() -> MatrixTwo<T> {
        let v: Vec<Vec<T>> = Vec::new();
        MatrixTwo { vec: v, dim: 2 }
    }

    pub fn dim_get(&self) -> usize {
        self.dim
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
}
