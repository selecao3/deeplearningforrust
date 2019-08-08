use crate::matrix::MatrixTwo;
use crate::matrix::MatrixThree;


#[derive(Clone,Debug)]
pub struct Sigmoid {
    params: Vec<MatrixTwo<f32>>
}

#[derive(Clone,Debug)]
pub struct Affine{
    params: Vec<MatrixTwo<f32>>
}

#[derive(Clone,Debug)]
pub struct Embedding{
    params: Vec<MatrixTwo<f32>>,
    pub grads: Vec<MatrixTwo<f32>>,
    idx:Option<Vec<usize>>
}


impl Sigmoid {
    pub fn init() -> Sigmoid{
        Sigmoid{
            params:Vec::new()
        }
    }
    pub fn forward(&self,x:MatrixTwo<f32>) -> MatrixTwo<f32>{
        let e:f32 = 2.718281828459045235360287471352;
        let x_minus = x.scalar_multiplication(-1.0);
        let bunbo:MatrixTwo<f32> = (x_minus.exp().scalar_plus(1.0));
        bunbo.division_scalar(1.0)
    }
}

impl Affine {
    pub fn init(W:MatrixTwo<f32>,b:MatrixTwo<f32>) -> Affine{
        Affine{
            params:vec![W,b]
        }
    }
    pub fn forward(&self, x:MatrixTwo<f32>) -> MatrixTwo<f32>{
        let W = &self.params[0];
        let b = &self.params[1];
        let out = x.dot(&W).sum(&b);
        out
    }
}

impl Embedding {
    pub fn init(W:MatrixTwo<f32>) ->Embedding{
        Embedding{
            params: vec![W.clone()],
            grads: vec![W.zeros_like()],
            idx: None
        }
    }
    pub fn forward(&mut self , idx:Vec<usize>) -> MatrixTwo<f32>{
        let W:MatrixTwo<f32> = self.params[0].clone();
        self.idx = Some(idx.clone());
        let mut tmp_vec:Vec<Vec<f32>> = Vec::new();
        for i in idx.iter(){
            tmp_vec.push(W[*i].clone());
        }
        let out:MatrixTwo<f32> = MatrixTwo::create_matrix(tmp_vec);
        out
    }

    //何も返さない形だが後ほど修正
    pub fn backward(&self , dout:MatrixTwo<f32>){
        let mut dW:MatrixTwo<f32> = MatrixTwo::<f32>::zeros_like(&self.grads[0]);
        //np.add.at(hoge,[0,1,2,2],1)
        //一行N列hogeの0番目,1番目,２番目,２番目に1を足すという意味
        
        if let Some(idx) = &self.idx{
            //一行と一行をそれぞれ加算する関数が必要
            //dWを引数にとる関数
            dW.print_matrix();
            for (i,word_id) in idx.iter().enumerate(){
                dW.sum_oneline(dout[i].clone(), *word_id);
            }
            dW.print_matrix();
        }

    }
}