use crate::matrix::MatrixTwo;
use crate::matrix::MatrixThree;


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
        let db = dt.matrixAllSum();
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
            //t列だけ抜き取りたい：xs
            // let xs_vec:Vec<Vec<f32>> = xs.iter().map(|v| &v[t]).cloned().collect();
            // let xs_t:MatrixTwo<f32> = MatrixTwo::create_matrix(xs_vec.clone());
            // let hs_vec:Vec<Vec<f32>> = xs.iter().map(|v| &v[t]).cloned().collect();
            // let mut hs_t:&MatrixTwo<f32> = &MatrixTwo::create_matrix(xs_vec);

            //xs[:,t,:]に当たる二次元行列を作成する
            // let mut xs_mat_two:MatrixTwo<f32> = MatrixTwo::new();
            // for z in 0..xs.len_z(){
            //     xs_mat_two.push(xs[z][t].clone());
            //     println!("xs_mat_two: ");
            //     xs_mat_two.print_matrix();
            // }
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