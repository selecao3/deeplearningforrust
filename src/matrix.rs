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

#[derive(Clone,Debug,PartialEq)]
pub enum MatrixBase<T> {
    MatrixOne(MatrixOne<T>),
    MatrixTwo(MatrixTwo<T>),
    MatrixThree(MatrixThree<T>),
    None
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
    pub fn create_matrix(v: Vec<T>) -> MatrixOne<T>{
        MatrixOne { vec: v, dim: 1 }
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
    pub fn shape(&self) -> (usize,usize){
        let y = &self.vec;
        let x = &y[0];
        (y.len(),x.len())
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
    pub fn create_matrix(v: Vec<Vec<Vec<T>>>) -> MatrixThree<T>{
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
        self.vec[0][0].len()
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
    //転置行列
    pub fn inverse(&self)->MatrixTwo<usize>{
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut inv_mat = MatrixTwo::<usize>::zeros(len_y, len_x);
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

impl MatrixThree<f32> {
    pub fn zeros(x: usize, y: usize, z: usize) -> MatrixThree<f32> {
        let v: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; x]; y]; z];
        MatrixThree { vec: v, dim: 3 }
    }
    //This function used tests;
    pub fn rand_generate(x:usize,y:usize,z:usize) -> MatrixThree<f32>{
        let mut png = rand::thread_rng();
        let mut vec_two:Vec<Vec<f32>> = Vec::new();
        let mut mat_three:MatrixThree<f32> = MatrixThree::new();

        for _k in 0..z{
            for _i in 0..y{
                let mut tmp_vec:Vec<f32> = Vec::<f32>::with_capacity(x);
                for _j in 0..x{
                    let r:f32 = png.gen_range(-1.0,1.0);
                    tmp_vec.push(r);
                }
                vec_two.push(tmp_vec.clone());
                //clearしないと以前のデータを引き継いだままループを再開する
                tmp_vec.clear();
            }
            mat_three.push(vec_two.clone());
            //clearしないと以前のデータを引き継いだままループを再開する
            vec_two.clear();
        }
        mat_three
    }
    fn convert_one_matrix(self) -> MatrixOne<f32>{
        let mut tmp_vec = Vec::new();
        for z in 0..self.len_z(){
            for y in 0..self.len_y(){
                for x in 0..self.len_x(){
                    tmp_vec.push(self.vec[z][y][x]);
                }
            }
        }
        MatrixOne::create_matrix(tmp_vec)
    }
    pub fn reshape(self, len_z:usize, len_y:usize, len_x:usize) -> MatrixBase<f32>{
        let source = self.convert_one_matrix();
        let mut source_len = 0;
        if len_z == 0 && len_y == 0 {
            return MatrixBase::MatrixOne(source)
        }else if len_z == 0 && len_y != 0{
            let mut target_vec:Vec<Vec<f32>> = Vec::new();
            for y in 0..len_y{
                let mut tmp_vec:Vec<f32> = Vec::<f32>::with_capacity(len_x);
                for x in 0..len_x {
                    tmp_vec.push(source[source_len]);
                    if source_len < source.len() {
                        source_len = source_len + 1;
                    }else {
                        panic!("Exceed source's length");
                    }
                }
                target_vec.push(tmp_vec.clone());
                tmp_vec.clear();
            }
            let mat2 = MatrixTwo::create_matrix(target_vec);
            return MatrixBase::MatrixTwo(mat2)
        }else{
            let mut target_vec:Vec<Vec<Vec<f32>>> = Vec::new();
            for z in 0..len_z{
                let mut tmp_two_vec:Vec<Vec<f32>> = Vec::<Vec<f32>>::with_capacity(len_y);
                for y in 0..len_y{
                    let mut tmp_one_vec:Vec<f32> = Vec::<f32>::with_capacity(len_x);
                    for x in 0..len_x {
                        //Vec::new()だとIndexにアクセスできないことに注意＝＞with_capacityを使う
                        // tmp_vec[z][y][x] = source[source_len];
                        tmp_one_vec.push(source[source_len]);
                        if source_len < source.len() {
                            source_len = source_len + 1;
                        }else {
                            panic!("Exceed source's length");
                        }
                    }
                    tmp_two_vec.push(tmp_one_vec.clone());
                    tmp_one_vec.clear();
                }
                target_vec.push(tmp_two_vec.clone());
                tmp_two_vec.clear();
            }
            let mat3 = MatrixThree::create_matrix(target_vec);
            return MatrixBase::MatrixThree(mat3)
        }
    }


    //usize型の引数によって、特定の２行列に破壊的変更を加える
    //status:0=>selfのx-y行列を変更
    //status:1=>selfのx-z行列を変更
    //もし、z:N,y:Y,x:Hの三次元行列ならば、y:N,x,Hの二次元行列に変換して計算する
    pub fn copy_two_matrix(&mut self,target:&MatrixTwo<f32>,point:usize,status:usize ){
        if status== 0 {
           self[point] = target.vec.clone();
        }else if (status == 1){
            //selfのx-z行列にtargetを代入する形にしないといけない
            for z in 0..self.len_z(){
                // ans_mat2.push(self[z][point].clone());
                self[z][point] = target[z].clone();
            }
        }
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
    //特定の1行に１行ベクトルを加算する
    pub fn sum_oneline(&mut self,mat:Vec<f32>,point:usize){
        if self.len_x() != mat.len(){
            panic!("列数が一致していないのでsum演算ができない");
        }
        for i in 0..self.len_x(){
            self[point][i] = self[point][i] + mat[i];
        }
    }
    //行列+スカラのメソッド
    pub fn scalar_plus(&self,target:f32) -> MatrixTwo<f32>{

        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = self[i][j] + target;
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
    //行列×スカラのメソッド
    pub fn scalar_multiplication(&self,target:f32) -> MatrixTwo<f32>{

        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = self[i][j] * target;
            }
        }
        ans
    }
    //行列÷スカラのメソッド
    pub fn scalar_division(&self,target:f32) -> MatrixTwo<f32>{

        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = self[i][j] / target;
            }
        }
        ans
    }
    //スカラ÷行列のメソッド
    pub fn division_scalar(&self,target:f32) -> MatrixTwo<f32>{

        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),self.len_y());
        for i in 0..self.len_y() {
            for j in 0..self.len_x() {
                ans[i][j] = target / self[i][j];
            }
        }
        ans
    }
    //dot演算
    //selfのy>mat2のyの時でないと成立しない＝＞それ以外はpanic!を起こすシステムを作る
    pub fn dot(&self, mat2:&MatrixTwo<f32>) -> MatrixTwo<f32>{
        self.dot_check(mat2);
        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(mat2.len_x(), self.len_y());
        let mat2_new:MatrixTwo<f32>;
        if mat2.len_y() == 1 {
            mat2_new = mat2.clone();
        }else {
            mat2_new = mat2.inverse();
        }
        //ここの3段forループどうにかしたい
        for y in 0..self.len_y(){
            for t in 0..mat2_new.len_y(){
                for x in 0..self.len_x(){
                    //1×1の時、mat2_new[1]にアクセスするためpanicになる
                    //右の行列が1列の場合、1列に変換してもいいためmat_xがバグる
                    ans[y][t] = ans[y][t] + self[y][x] * mat2_new[t][x];
                }

            }
        }
        ans
    }
    fn dot_check(&self,mat2:&MatrixTwo<f32>){
        if self.len_x() != mat2.len_y(){
            println!("len_x: {:?}",self.len_x());
            println!("len_y: {:?}",mat2.len_y());
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
            println!("====================");
            self.print_matrix();
            println!("====================");
            mat.print_matrix();
            println!("====================");
            panic!("行と列が一致していないのでスカラ演算ができない");
        }
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut ans = MatrixTwo::<f32>::zeros(len_x,len_y);
        for y in 0..len_y{
            for x in 0..len_x{
                ans[y][x] = self[y][x] * mat[y][x];
            }
        }
        ans
    }
    //行列版expメソッド
    pub fn exp(&self) -> MatrixTwo<f32>{
        let len_x = self.len_x();
        let len_y = self.len_y();
        let mut ans = MatrixTwo::<f32>::zeros(len_x,len_y);
        for y in 0..len_y{
            for x in 0..len_x{
                ans[y][x] = libm::F32Ext::exp(self[y][x]);
                // ans[y][x] = libm::F32Ext::powf(self[y][x], pow_target);
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
                ans[y][x] = libm::F32Ext::powf(self[y][x], pow_target);
            }
        }
        ans
    }
    //行列の列要素を全て足し上げ
    pub fn matrixAllSum(&self) -> MatrixTwo<f32>{
        let mut ans:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(self.len_x(),1);
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
    //転置行列
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
    fn convert_one_matrix(self) -> MatrixOne<f32>{
        let mut tmp_vec = Vec::new();
        for y in 0..self.len_y(){
            for x in 0..self.len_x(){
                tmp_vec.push(self.vec[y][x]);
            }
        }
        MatrixOne::create_matrix(tmp_vec)
    }
    pub fn reshape(self, len_z:usize, len_y:usize, len_x:usize) -> MatrixBase<f32>{
        let source = self.convert_one_matrix();
        let mut source_len = 0;
        if len_z == 0 && len_y == 0 {
            return MatrixBase::MatrixOne(source)
        }else if len_z == 0 && len_y != 0{
            let mut target_vec:Vec<Vec<f32>> = Vec::new();
            for y in 0..len_y{
                let mut tmp_one_vec:Vec<f32> = Vec::<f32>::with_capacity(len_x);
                for x in 0..len_x {
                    tmp_one_vec.push(source[source_len]);
                    if source_len < source.len() {
                        source_len = source_len + 1;
                    }else {
                        panic!("Exceed source's length");
                    }
                }
                target_vec.push(tmp_one_vec);
            }
            let mat2 = MatrixTwo::create_matrix(target_vec);
            return MatrixBase::MatrixTwo(mat2)
        }else{
            let mut target_vec:Vec<Vec<Vec<f32>>> = Vec::new();
            for z in 0..len_z{
                let mut tmp_two_vec:Vec<Vec<f32>> = Vec::<Vec<f32>>::with_capacity(len_y);
                for y in 0..len_y{
                    let mut tmp_one_vec:Vec<f32> = Vec::<f32>::with_capacity(len_x);
                    for x in 0..len_x {
                        tmp_one_vec.push(source[source_len]);
                        if source_len < source.len() {
                            source_len = source_len + 1;
                        }else {
                            panic!("Exceed source's length");
                        }
                    }
                    tmp_two_vec.push(tmp_one_vec.clone());
                    tmp_one_vec.clear();
                }
                target_vec.push(tmp_two_vec.clone());
                tmp_two_vec.clear();
            }
            let mat3 = MatrixThree::create_matrix(target_vec);
            return MatrixBase::MatrixThree(mat3)
        }
    }
}

