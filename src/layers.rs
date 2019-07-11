use crate::matrix::MatrixOne;
use crate::matrix::MatrixTwo;
use crate::matrix::MatrixThree;

#[derive(Clone)]
struct RNN {
    params: Vec<MatrixTwo<f32>>,
    grads: Vec<MatrixTwo<f32>>,
    cache: Option<(MatrixTwo<f32>, MatrixTwo<f32>, MatrixTwo<f32>)>,
}

#[derive(Clone)]
struct TimeRNN {
    params: Vec<MatrixTwo<f32>>,
    grads: Vec<MatrixTwo<f32>>,
    layers: Option<Vec<RNN>>,
    h: Option<MatrixTwo<f32>>,
    dh: Option<MatrixTwo<f32>>,
    stateful: bool,
}

impl RNN {
    fn init(Wx: MatrixTwo<f32>, Wh: MatrixTwo<f32>, b: MatrixTwo<f32>) -> RNN {
        let params = vec![Wx.clone(), Wh.clone(), b.clone()];
        let grads = vec![Wx.zeros_like(), Wh.zeros_like(), b.zeros_like()];
        let cache = None;
        RNN {
            params: params,
            grads: grads,
            cache: cache,
        }
    }
    fn forward(&mut self, x: MatrixTwo<f32>, h_prev: &MatrixTwo<f32>) -> MatrixTwo<f32> {
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
        let dWh = h_prev.dot(&dt);
        let dh_prev = dt.dot(Wh);
        let dWx = x.dot(&dt);
        let dx = dt.dot(&Wx);
        &self.grads.clear();
        &self.grads.insert(0, dWx);
        &self.grads.insert(1, dWh);
        &self.grads.insert(2, db);
        (dx, dh_prev)
    }
}

impl TimeRNN {
    fn init(Wx: MatrixTwo<f32>, Wh: MatrixTwo<f32>, b: MatrixTwo<f32>, stateful: bool) -> TimeRNN {
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
    fn set_state(mut self,h:MatrixTwo<f32>){
        self.h = Some(h);
    }
    fn reset_state(mut self){
        self.h = Some(MatrixTwo::new());
    }
    fn forward(&mut self,xs:MatrixThree<f32>) -> MatrixThree<f32>{
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        //z,y,xの順
        let (N,T,D) = xs.clone().shape();
        let (D,H) = Wx.clone().shape();
        
        //x,y,zの順(zeros)
        let hs = MatrixThree::zeros(D, T, N);
        if !self.stateful || self.h.is_none()  {
            self.h = Some(MatrixTwo::<f32>::zeros(H,N));
        }

        for t in 0..T {

            let mut layer = RNN::init(Wx.clone(),Wh.clone(),b.clone());
            //t列だけ抜き取りたい：xs
            let xs_vec:Vec<Vec<f32>> = xs.iter().map(|v| &v[t]).cloned().collect();
            let xs_t:MatrixTwo<f32> = MatrixTwo::create_matrix(xs_vec.clone());
            let hs_vec:Vec<Vec<f32>> = xs.iter().map(|v| &v[t]).cloned().collect();
            let mut hs_t:&MatrixTwo<f32> = &MatrixTwo::create_matrix(xs_vec);
            //self.hを破壊的変更する
            layer.forward(xs_t, &self.h.clone().unwrap());
            hs_t = &self.h.clone().unwrap();
            self.layers.clone().unwrap().push(layer);
        }
        hs
    }
    fn backward(mut self,dhs:MatrixThree<f32>) -> MatrixThree<f32>{
        let mut layer:RNN;
        let mut dh:MatrixTwo<f32>;
        // let mut dxs_2:MatrixTwo<f32>;
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        //全てD:x,N:y,T:zの順にする方が後々よい
        //let (N,T,D) = dhs.clone().shape();
        let (H,N,T) = dhs.clone().shape();
        let (D,T) = Wx.clone().shape();

        //let mut dxs = MatrixThree::zeros(D, T, N);
        //pythonだとz,y,xの順：np.empty
        //N,T,D => D,T,N
        let mut dxs = MatrixThree::zeros(D, T, N);
        dh = MatrixTwo::new();
        //macroを使った方がいい：macroは任意の引数を取れるため
        let mut grads = vec![MatrixTwo::<f32>::new(),MatrixTwo::<f32>::new(),MatrixTwo::<f32>::new()];
        for t in T..0 {
            if let Some(layer_some) = self.layers.clone(){
                layer = layer_some[t].clone();
            }else {
                panic!("Not layer");
            }
            // let dhs_vec:Vec<Vec<f32>> = dhs.iter().map(|v| &v[t]).cloned().collect();
            // let dhs_t:MatrixTwo<f32> = MatrixTwo::create_matrix(dhs_vec.clone());

            // let dxs_vec:Vec<Vec<f32>> = dxs.iter().map(|v| &v[t]).cloned().collect();
            // let mut dxs_t:MatrixTwo<f32> = MatrixTwo::create_matrix(dxs_vec.clone());
            let dhs_2 = MatrixTwo::create_matrix(dhs[t].clone());
            // dxs_2 = MatrixTwo::create_matrix(dxs[t].clone());
            // let aaa = dxs[t];

            //dxs[t].spl
            //

            //dxsのt固定の２行列にdxを代入したい...
            let (dx,dh_tmp) = layer.backward(dhs_2.sum(&dh));
            dh = dh_tmp;
            //dxsを最終的に返さないといけない
            // dxs_2 = dx;
            &dxs[t].splice(0..,dx.into_iter());
            // let a = dxs[0][t];
            // dxs.iter().map(|v| v.splice(0.., dx.vec));
            for (i,grad) in layer.grads.iter().enumerate() {
                grads[i] = grads[i].sum(grad);
            }
        }
        for (i,grad) in grads.iter().enumerate() {
            self.grads[i] = grad.clone();
        }
        self.dh = Some(dh);
        dxs
    }
}
