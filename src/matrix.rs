use std::vec::IntoIter;
use std::ops::Index;
use std::ops::IndexMut;
use std::slice::Iter;
use std::slice::SliceIndex;
use rand::Rng;

#[derive(Clone,Debug,PartialEq)]
pub struct MatrixOne<T> {
    vec: Vec<T>,
    dim: usize,
}
#[derive(Clone,Debug,PartialEq)]
pub struct MatrixTwo<T> {
    vec: Vec<Vec<T>>,
    dim: usize,
}
#[derive(Clone,Debug,PartialEq)]
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
    pub fn create_matrix(v: Vec<Vec<T>>) -> MatrixTwo<T>{
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
    pub fn into_iter(self) -> IntoIter<Vec<T>>{
        self.vec.into_iter()
    }
}

impl<T> MatrixThree<T> {
    //Create 1-dimensional matrix
    pub fn new() -> MatrixThree<T> {
        let v: Vec<Vec<Vec<T>>> = Vec::new();
        MatrixThree{ vec: v, dim: 3 }
    }
    pub fn iter(&self) -> Iter<Vec<Vec<T>>> {
        self.vec.iter()
    }
    pub fn dim_get(&self) -> usize {
        self.dim
    }
    pub fn push(&mut self, value: Vec<Vec<T>>) -> () {
        self.vec.push(value)
    }
    pub fn len_z(&self) -> usize {
        self.vec.len()
    }
    pub fn len_y(&self) -> usize {
        self.vec[0].len()
    }
    pub fn len_x(&self) -> usize {
        self.vec[1].len()
    }
    pub fn shape(self) -> (usize,usize,usize){
        let z = self.vec;
        let y = &z[0];
        let x = &y[0];
        (z.len(),y.len(),x.len())
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


    pub fn convert_one_hot(&self, vocab_size: usize) -> MatrixThree<f32> {
        let N: usize = self.len_y();
        let C: usize = self.len_x();
        let mut one_hot = MatrixThree::zeros(vocab_size, C, N);
        for (idx_0, word_ids) in self.iter().enumerate() {
            for (idx_1, word_id) in word_ids.iter().enumerate() {
                let word_id: usize = word_id.clone();
                //word_id:x, idx_1:y, idx_0:z
                one_hot[idx_0][idx_1][word_id] = 1.0;
            }
        }
        one_hot
    }
    pub fn print_matrix(&self){
        for target in self.vec.iter() {
            println!("{:?}", target);
        }
    }
}

impl MatrixThree<f32> {
    pub fn zeros(x: usize, y: usize, z: usize) -> MatrixThree<f32> {
        let v: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; x]; y]; z];
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
    //This function used tests;
    pub fn rand_generate(x:usize,y:usize) -> MatrixTwo<f32>{
        let mut png = rand::thread_rng();
        let mut vec:Vec<Vec<f32>> = Vec::new();

        for _i in 0..y{
            let mut tmp_vec:Vec<f32> = Vec::<f32>::with_capacity(x);
            for _j in 0..x{
                let r:f32 = png.gen_range(-1.0,1.0);
                tmp_vec.push(r);
            }
            vec.push(tmp_vec);
        }
        MatrixTwo::create_matrix(vec)
    }
    pub fn shape(self) -> (usize,usize){
        let z = self.vec;
        let y = &z[0];
        (z.len(),y.len())
    }
    pub fn sum(&self,mat:&MatrixTwo<f32>) -> MatrixTwo<f32>{
        if !self.match_check(&mat){
            println!("self:");
            self.print_matrix();
            println!("mat:");
            mat.print_matrix();
            panic!("行と列が一致していないのでsum演算ができない");
        }
        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = self[i][j] + mat[i][j];
            }
        }
        ans
    }
    //行列-スカラのメソッド
    pub fn scalar_minus(&self,target:f32) -> MatrixTwo<f32>{

        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = self[i][j] - target;
            }
        }
        ans
    }
    //dot演算
    //selfのy>mat2のyの時でないと成立しない＝＞それ以外はpanic!を起こすシステムを作る
    pub fn dot(&self, mat2:&MatrixTwo<f32>) -> MatrixTwo<f32>{
        self.dot_check(mat2);
        let ans_y = self.len_y();
        let ans_x = self.len_x();
        //let ans_x = mat2.len_x();
        let mat_x = mat2.len_x();
        // let mat2_x = mat2.len_x();
        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(mat_x, ans_y);
        let mat2_new:MatrixTwo<f32>;
        if mat2.len_y() == 1 {
            mat2_new = mat2.clone();
        }else {
            mat2_new = mat2.inverse();
        }
        //ここの3段forループどうにかしたい
        for y in 0..ans_y{
            for t in 0..mat_x{
                for x in 0..ans_x{
                    ans[y][t] = ans[y][t] + self[y][x] * mat2_new[t][x];
                }

            }
        }
        ans
    }
    fn dot_check(&self,mat2:&MatrixTwo<f32>){
        if &self.len_x() != &mat2.len_y() && &self.len_x() != &mat2.len_x() {
            panic!("行と列が一致しない");
        }
    }
    fn match_check(&self,mat:&MatrixTwo<f32>) -> bool{
        if self.len_x() == mat.len_x() && self.len_y() == mat.len_y() {
            true
        }else{
            false
        }
    }
    //行列の掛け算
    pub fn mat_scalar(&self, mat:MatrixTwo<f32>) -> MatrixTwo<f32>{
        if !self.match_check(&mat){
            panic!("行と列が一致していないのでスカラ演算ができない");
        }
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut ans = MatrixTwo::<f32>::zeros(len_x,len_y);
        self.print_matrix();
        mat.print_matrix();
        for y in 0..len_y{
            for x in 0..len_x{
                ans[y][x] = self[y][x] * mat[y][x];
            }
        }
        ans
    }
    //行列版powメソッド
    pub fn pow(&self, pow_target:f32) -> MatrixTwo<f32>{
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut ans = MatrixTwo::<f32>::zeros(len_x,len_y);
        for y in 0..len_y{
            for x in 0..len_x{
                ans[y][x] = libm::pow(self[y][x] as f64 , pow_target as f64) as f32;
            }
        }
        ans
    }
    //行列の列要素を全て足し上げ
    pub fn matrixAllSum(&self) -> MatrixTwo<f32>{
        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        for x in 0..self.len_x() {
            for y in 0..self.len_y() {
                ans[0][x] += self[y][x];
            }
        }
        ans
    }
    
    //行列の各項に対し、tanhの計算
    pub fn tanh(&self) -> MatrixTwo<f32>{

        let mut ans = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        let len_y = self.len_y();
        let len_x = self.len_x();
        for y in 0..len_y {
            for x in 0..len_x {
                ans[y][x] = libm::tanh(self[y][x] as f64) as f32;
            }
        }
        ans
        
    }
    //転置行列:privete method!!
    pub fn inverse(&self)->MatrixTwo<f32>{
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut inv_mat = MatrixTwo::<f32>::zeros(len_y, len_x);
        for i in 0..len_x {
            for j in 0..len_y {
                inv_mat[i][j] = self[j][i];
            }
        }
        inv_mat
    }
    pub fn print_matrix(&self){
        for target in self.vec.iter() {
            println!("{:?}", target);
        }
    }
}

