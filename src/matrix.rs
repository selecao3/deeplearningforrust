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
    pub fn len_y(&self) -> usize {
        self.vec.len()
    }
    pub fn len_x(&self) -> usize {
        self.vec[0].len()
    }

}

impl MatrixOne<usize> {
    //1次元or二次元で挙動が変わる（前者：2次元のone_hot,後者：3次元のone_hot）
    pub fn convert_one_hot(&self, vocab_size: usize) -> MatrixTwo<usize> {
        let N: usize = self.len();
        //8*7行列
        let mut one_hot = MatrixTwo::<usize>::zeros(vocab_size, N);
        println!("{:?}", one_hot.vec);
        for (idx, word_id) in self.iter().enumerate() {
            let word_id: usize = word_id.clone();
            //idx行のword_id列に1を代入
            one_hot[idx][word_id] = 1;
        }
        one_hot
    }
}

impl MatrixTwo<usize> {
    pub fn zeros(x: usize, y: usize) -> MatrixTwo<usize> {
        let v: Vec<Vec<usize>> = vec![vec![0; x]; y];
        MatrixTwo { vec: v, dim: 2 }
    }

    pub fn zeros_like(&self) -> MatrixTwo<usize> {
        MatrixTwo::<usize>::zeros(self.len_x(),self.len_y())
    }


    pub fn convert_one_hot(&self, vocab_size: usize) -> MatrixThree<usize> {
        let N: usize = self.len_y();
        let C: usize = self.len_x();
        let mut one_hot = MatrixThree::zeros(vocab_size, C, N);
        for (idx_0, word_ids) in self.iter().enumerate() {
            for (idx_1, word_id) in word_ids.iter().enumerate() {
                let word_id: usize = word_id.clone();
                //word_id:x, idx_1:y, idx_0:z
                one_hot[idx_0][idx_1][word_id] = 1;
            }
        }
        one_hot
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

//行列演算：map関数とかでもうちょいスマートにできるんじゃね？
impl MatrixTwo<f32>{
    pub fn zeros(x: usize, y: usize) -> MatrixTwo<f32> {
        let v: Vec<Vec<f32>> = vec![vec![0.0; x]; y];
        MatrixTwo { vec: v, dim: 2 }
    }

    pub fn zeros_like(&self) -> MatrixTwo<f32> {
        MatrixTwo::<f32>::zeros(self.len_x(),self.len_y())
    }
    pub fn sum(&self,mat:&MatrixTwo<f32>) -> MatrixTwo<f32>{
        let mut ans:MatrixTwo<f32> = MatrixTwo::new();
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = self[i][j] + mat[i][j];
            }
        }
        ans
    }
    pub fn scalar_minus(&self,target:f32) -> MatrixTwo<f32>{
        let mut ans:MatrixTwo<f32> = MatrixTwo::new();
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = self[i][j] - target;
            }
        }
        ans
    }
    pub fn dot(&self, mat2:&MatrixTwo<f32>) -> MatrixTwo<f32>{
        let mut tmp:usize = 0;
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut ans = MatrixTwo::new();
        let inv_mat = mat2.inverse();
        for y in 0..len_y{
            for x in 0..len_x{
                ans[y][tmp] += self[y][x] * inv_mat[y][x];
            }
            if tmp != len_x {
                tmp += 1;
            }else {
                tmp = 0;
            }
        }
        ans
    }
    pub fn mat_scalar(&self, mat:MatrixTwo<f32>) -> MatrixTwo<f32>{
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut ans = MatrixTwo::new();
        for y in 0..len_y{
            for x in 0..len_x{
                ans[y][x] = self[y][x] * mat[y][x];
            }
        }
        ans
    }
    pub fn pow(&self, pow_target:f32) -> MatrixTwo<f32>{
        let mut tmp:usize = 0;
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut ans = MatrixTwo::new();
        for y in 0..len_y{
            for x in 0..len_x{
                ans[y][tmp] = libm::pow(self[y][x] as f64 , pow_target as f64) as f32;
            }
            if tmp != len_x {
                tmp += 1;
            }else {
                tmp = 0;
            }
        }
        ans
    }
    //行列の列要素を全て足し上げ
    pub fn matrixAllSum(&self) -> MatrixOne<f32>{
        let mut ans:MatrixOne<f32> = MatrixOne::new();
        for x in 0..self.len_x() {
            for y in 0..self.len_y() {
                ans[x] += self[y][x];
            }
        }
        ans
    }
    pub fn tanh(&self) -> MatrixTwo<f32>{
        let mut ans = MatrixTwo::new();
        let len_y = self.len_y();
        let len_x = self.len_x();
        for y in 0..len_y {
            for x in 0..len_x {
                ans[y][x] = libm::tanh(self[y][x] as f64) as f32;
            }
        }
        ans
        
    }
    fn inverse(&self)->MatrixTwo<f32>{
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut inv_mat = MatrixTwo::<f32>::zeros(len_x, len_y);
        for i in 0..len_x {
            for j in 0..len_y {
                inv_mat[i][j] = self[j][i];
            }
        }
        inv_mat
    }
}

