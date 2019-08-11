use crate::matrix::MatrixTwo;
use crate::matrix::MatrixThree;
use crate::matrix::MatrixBase;
use crate::layers::*;


#[derive(Clone,Debug)]
pub struct RNN {
    params: Vec<MatrixTwo<f32>>,
    grads: Vec<MatrixTwo<f32>>,
    pub cache: Option<(MatrixTwo<f32>, MatrixTwo<f32>, MatrixTwo<f32>)>,
}

#[derive(Clone,Debug)]
pub struct TimeRNN {
    params: Vec<MatrixTwo<f32>>,
    grads: Vec<MatrixTwo<f32>>,
    layers: Option<Vec<RNN>>,
    pub h: Option<MatrixTwo<f32>>,
    dh: Option<MatrixTwo<f32>>,
    stateful: bool,
}


#[derive(Clone,Debug)]
pub struct TimeEmbedding{
    params: Vec<MatrixTwo<f32>>,
    grads: Vec<MatrixTwo<f32>>,
    layers: Option<Vec<Embedding>>,
    W:MatrixTwo<f32>
}

#[derive(Clone,Debug)]
pub struct TimeAffine{
    params: Vec<MatrixTwo<f32>>,
    grads: Vec<MatrixTwo<f32>>,
    x:Option<MatrixThree<f32>>
}

impl RNN {
    pub fn init(Wx: MatrixTwo<f32>, Wh: MatrixTwo<f32>, b: MatrixTwo<f32>) -> RNN {
        let params = vec![Wx.clone(), Wh.clone(), b.clone()];
        let grads = vec![Wx.zeros_like(), Wh.zeros_like(), b.zeros_like()];
        let cache = None;
        RNN {
            params: params,
            grads: grads,
            cache: cache,
        }
    }
    pub fn forward(&mut self, x: MatrixTwo<f32>, h_prev: &MatrixTwo<f32>) -> MatrixTwo<f32> {
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        let t = h_prev.dot(Wh).sum(&x.dot(Wx)).sum(b);
        let h_next = t.tanh();

        self.cache = Some((x, h_prev.to_owned(), h_next.clone()));
        h_next
    }
    //3点リーダ：データの上書きを意味する(python)
    pub fn backward(&mut self, dh_next: MatrixTwo<f32>) -> (MatrixTwo<f32>, MatrixTwo<f32>) {
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        let (x, h_prev, h_next) = self.cache.clone().unwrap();

        let dt = dh_next.mat_scalar(h_next.pow(2f32).scalar_minus(1f32));
        let db = dt.matrixAllSum(0);
        let dWh = h_prev.inverse().dot(&dt);
        let dh_prev = dt.dot(&Wh.inverse());
        let dWx = x.inverse().dot(&dt);
        let dx = dt.dot(&Wx.inverse());
        &self.grads.clear();
        &self.grads.insert(0, dWx);
        &self.grads.insert(1, dWh);
        &self.grads.insert(2, db);
        (dx, dh_prev)
    }
}

impl TimeRNN {
    pub fn init(Wx: MatrixTwo<f32>, Wh: MatrixTwo<f32>, b: MatrixTwo<f32>, stateful: bool) -> TimeRNN {
        let params = vec![Wx.clone(), Wh.clone(), b.clone()];
        let grads = vec![Wx.zeros_like(), Wh.zeros_like(), b.zeros_like()];
        let stateful = stateful;
        TimeRNN {
            params: params,
            grads: grads,
            layers: None,
            h: None,
            dh: None,
            stateful: stateful,
        }
    }
    pub fn set_state(&mut self,h:MatrixTwo<f32>) ->(){
        // 構造体のメンバは参照にできない。つまり、普通の代入では構造体のメンバの値を変更することはできない
        // self.h = Some(h);
        // 下のget_or_insertメソッドを使えば変更することができる
        self.h.get_or_insert(h);
    }
    pub fn reset_state(&mut self){
        &self.h.take();
    }
    pub fn forward(&mut self,mut xs:MatrixThree<f32>) -> MatrixThree<f32>{
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        //z,y,xの順
        let (N,T,D) = xs.clone().shape();
        let (D,H) = Wx.clone().shape();
        
        //x,y,zの順(zeros)
        let mut hs = MatrixThree::zeros(D, T, N);
        if !self.stateful || self.h.is_none()  {
            self.h = Some(MatrixTwo::<f32>::zeros(H,N));
        }

        let mut tmp_vec:Vec<RNN> = Vec::new();
        for t in 0..T {

            let mut layer = RNN::init(Wx.clone(),Wh.clone(),b.clone());
            let xs_mat_two:MatrixTwo<f32> = mat3_convert_mat2(&xs, t);
            

            //get_or_insertはsomeでラップされていると引数の値に変更されない。つまりNone出ないといけない
            //self.h.get_or_insert(layer.forward(xs_mat_two, &self.h.clone().unwrap()));
            self.h = Some(layer.forward(xs_mat_two, &self.h.clone().unwrap()));
            if let Some(h) = &self.h{
                hs.copy_two_matrix(&h, t, 1);
            }
            //self.h.get_or_insert(layer.forward(xs_mat_two, &self.h.clone().unwrap()));
            // hs_t = &self.h.clone().unwrap();
            tmp_vec.push(layer);
            self.layers = Some(tmp_vec.clone());
        }
        hs
    }
    pub fn backward(mut self,dhs:MatrixThree<f32>) -> MatrixThree<f32>{
        let mut layer:RNN;
        // let mut dxs_2:MatrixTwo<f32>;
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        //全てD:x,N:y,T:zの順にする方が後々よい
        //let (N,T,D) = dhs.clone().shape();
        let (N,T,H) = dhs.clone().shape();
        let (H,D) = Wx.clone().shape();
        //n4t10h3
        //d3h2

        //let mut dxs = MatrixThree::zeros(D, T, N);
        //pythonだとz,y,xの順：np.empty
        //N,T,D => D,T,N
        let mut dxs = MatrixThree::zeros(D, T, N);
        //dhをインスタンス化すると、後のsum関数でpanicになる（行と列が0だから）
        // let mut dh:MatrixTwo<f32> = MatrixTwo::new();
        let mut dh:MatrixTwo<f32> = MatrixTwo::<f32>::zeros(dhs.len_x(), dhs.len_z());

        //macroを使った方がいい：macroは任意の引数を取れるため
        //Vec<Option<MatrixTwo<f32>..>>>でNoneにしないと後のsum関数で面倒なことになる（zerosを使って初期化しないといけないことになる）
        type Grad = Option<MatrixTwo<f32>>;
        let mut grads:Vec<Grad> = vec![None,None,None];

        for t in (0..T).rev() {
            if let Some(layer_some) = self.layers.clone(){
                layer = layer_some[t].clone();
            }else {
                panic!("Not layer");
            }
            //dhsをy:tの２行列へ
            let dhs_mat_two:MatrixTwo<f32> = mat3_convert_mat2(&dhs, t);

            //dxsのt固定の２行列にdxを代入したい...
            //dhに代入したいがタプルは束縛しないとダメなので一旦dh_memoryに代入し、次の行でdhに代入
            let (dx,dh_memory) = layer.backward(dhs_mat_two.sum(&dh));
            dh = dh_memory;

            dxs.copy_two_matrix(&dx, t, 1);
            for (i,grad) in layer.grads.iter().enumerate() {
                if let Some(grad_some) = &grads[i]{
                    grads[i] = Some(grad_some.sum(grad));
                }else {
                    grads[i] = Some(grad.clone());
                }
            }
        }
        for (i,grad) in grads.iter().enumerate() {
            if let Some(grad_some) = grad{
                self.grads[i] = grad_some.clone();
            }else {
                panic!("grad is none");
            }

        }
        self.dh = Some(dh);
        dxs
    }
}
    //xs[:,t,:]に当たる二次元行列を作成する
    fn mat3_convert_mat2(mat3:&MatrixThree<f32>,point:usize) -> MatrixTwo<f32>{
        let mut mat2:MatrixTwo<f32> = MatrixTwo::new();
        for z in 0..mat3.len_z(){
            mat2.push(mat3[z][point].clone());
        }
        mat2
    }

//Embedding implを作成してから再開すること
impl TimeEmbedding {
    pub fn init(W:MatrixTwo<f32>) -> TimeEmbedding{
        TimeEmbedding{
            grads:vec![W.zeros_like()],
            params:vec![W.clone()],
            layers:None,
            W:W
        }
    }
    pub fn forward(&mut self , xs:MatrixTwo<usize>) -> MatrixThree<f32>{
        //y:N , x:Tの行列＝＞xs
        let (N,T) = xs.shape();
        //y:V , x:Dの行列＝＞W
        let (V,D) = self.W.shape();

        let mut out = MatrixThree::zeros(N, T, D);
        self.layers = Some(Vec::new());

        let mut tmp_vec:Vec<Embedding> = Vec::new();
        for t in 0..T{
            let mut layer = Embedding::init(self.W.clone());
            // out.copy_two_matrix(layer.forward(xs[:,t]), t, 1);
            //第一引数outはWに、第二引数はxsに影響されている=>Wはy行がT、xsもy行がT行無いと成立しない
            out.copy_two_matrix(&layer.forward(xs.inverse()[t].clone()), t, 1);
            tmp_vec.push(layer);
            self.layers = Some(tmp_vec.clone());
        }
        out
    }
    pub fn backward(&mut self,dout:MatrixThree<f32>){
        let (N,T,D) = dout.clone().shape();
        let mut grad:Option<MatrixTwo<f32>> = None;
        for t in 0..T{
            if let Some(layer) = &self.layers{
                layer[t].backward(mat3_convert_mat2(&dout, t));
                if let Some(grad_some) = grad{
                    grad = Some(grad_some.sum(&layer[t].grads[0]))
                }else {
                    grad = Some(layer[t].grads[0].clone());
                }
            }else {
                panic!("not layer.");
            }
        }
        self.grads[0] = grad.unwrap();
    }
}
impl TimeAffine {
    pub fn init(W:MatrixTwo<f32>,b:MatrixTwo<f32>) -> TimeAffine{
        TimeAffine{
            grads: vec![W.zeros_like(),b.zeros_like()],
            params:vec![W,b],
            x: None
        }
    }
    pub fn forward(&mut self, x:MatrixThree<f32>) -> MatrixThree<f32>{
        let (N,T,D) = x.clone().shape();
        let W = &self.params[0];
        let b = &self.params[1];
        let mut ans:MatrixThree<f32> = MatrixThree::new();
        if let MatrixBase::MatrixTwo(rx) = x.clone().reshape(0, N*T, D){
            let out = rx.dot(W).sum(b);
            self.x = Some(x.clone());
            if let MatrixBase::MatrixThree(mat3) = out.reshape(N, T, D){
                ans = mat3;
            }else {
                panic!("reshape failed");
            }
        }else {
            panic!("reshape failed.");
        }
        ans
    }
    pub fn backward(&mut self, dout:MatrixThree<f32>) -> MatrixThree<f32>{
        if let Some(x) = &self.x{
            let (N,T,D) = x.clone().shape();
            let W = &self.params[0];
            let b = &self.params[1];

            let dout = dout.reshape(0, N*T, D);
            let rx = x.clone().reshape(0, N*T, D);

            if let MatrixBase::MatrixTwo(dout_some) = dout{
                let db = dout_some.matrixAllSum(0);
                if let MatrixBase::MatrixTwo(rx_some) = rx{
                    let dW = rx_some.inverse().dot(&dout_some);
                    // let dx = dout_some.dot(&W.inverse());
                    //W.inverseするとdot演算ができない。。。
                    let dx = dout_some.dot(&W.inverse());
                    let dx_reshaped = dx.reshape(N,T,D);
                    if let MatrixBase::MatrixThree(dx_some) = dx_reshaped{
                        self.grads[0] = dW;
                        self.grads[1] = db;
                        return dx_some
                    }else {
                        panic!("dx have not could be gotten.");
                    }
                }else {
                    panic!("rx have not could be gotten.");
                }
            }else {
                panic!("dout have not could be gotten.");
            }
        }else {
            panic!("self.x is none.");
        }
    }
}