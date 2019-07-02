use crate::matrix::MatrixOne;
use crate::matrix::MatrixTwo;
use crate::matrix::MatrixThree;

struct RNN {
    params: Vec<MatrixTwo<f32>>,
    grads: Vec<MatrixTwo<f32>>,
    cache: Option<(MatrixTwo<f32>, MatrixTwo<f32>, MatrixTwo<f32>)>,
}

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
    fn forward(&mut self, x: MatrixTwo<f32>, h_prev: MatrixTwo<f32>) -> MatrixTwo<f32> {
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        let t = h_prev.dot(Wh).sum(&x.dot(Wx)).sum(b);
        let h_next = t.tanh();

        self.cache = Some((x, h_prev, h_next.clone()));
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
    fn forward(mut self,xs:MatrixThree<f32>) -> MatrixThree<f32>{
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        //z,y,xの順
        let (N,T,D) = &xs.shape();
        let (D,H) = Wx.clone().shape();
        
        //x,y,zの順(zeros)
        let hs = MatrixThree::zeros(D, T, N);
        if !self.stateful || self.h.is_none()  {
            self.h = Some(MatrixTwo::<f32>::zeros(H,N));
        }

        for t in 0..T {
            let layer = RNN::init(Wx.clone(),Wh.clone(),b.clone());
            //t列だけ抜き取りたい：xs
            let xs_vec:Vec<Vec<f32>> = xs.iter().map(|v| v[t]).collect();
            let xs_t:MatrixTwo<f32> = MatrixTwo{vec:xs_vec.to_owned() ,dim:2};
            let hs_vec:Vec<Vec<f32>> = xs.iter().map(|v| v[t]).collect();
            let hs_t:MatrixTwo<f32> = MatrixTwo{vec:xs_vec.to_owned(),dim:2};
            self.h = Some(layer.forward(xs_t, self.h.unwrap()));
            hs_t = self.h.unwrap();
            self.layers.unwrap().push(layer);
        }
        hs
        
    }
}
