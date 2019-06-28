use crate::matrix::MatrixOne;
use crate::matrix::MatrixTwo;

struct RNN {
    params:Vec<MatrixTwo<f32>>,
    grads:Vec<MatrixTwo<f32>>,
    cache:Option<(MatrixTwo<f32>,MatrixTwo<f32>,MatrixTwo<f32>)>
}

impl RNN {
    fn init(Wx:MatrixTwo<f32>,Wh:MatrixTwo<f32>,b:MatrixTwo<f32>) -> RNN {
        let params = vec!(Wx.clone(),Wh.clone(),b.clone());
        let grads = vec!(Wx.zeros_like(),Wh.zeros_like(),b.zeros_like());
        let cache = None;
        RNN{
            params:params,
            grads:grads,
            cache:cache
        }
    }
    fn forward(&mut self,x:MatrixTwo<f32>,h_prev:MatrixTwo<f32>) -> MatrixTwo<f32>{
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        let t = h_prev.dot(Wh).sum(&x.dot(Wx)).sum(b);
        let h_next = t.tanh();

        self.cache = Some((x,h_prev,h_next.clone()));
        h_next
    }
    pub fn backward(&self, dh_next:MatrixTwo<f32>) -> RetType {
        let Wx = &self.params[0];
        let Wh = &self.params[1];
        let b = &self.params[2];
        let (x,h_prev,h_next) = self.cache.unwrap();

        let dt = dh_next.mat_scalar(h_next.pow(2f32).scalar_minus(1f32));
        let db = dt.matrixAllSum();
        let dWh = h_prev.dot(&dt);
        let dh_prev = dt.dot(Wh);
        let dWx = x.dot(&dt);
        let dx = dt.dot(&Wx);
        self.grads[0] = dWx;
        self.grads[1] = dWh;
        self.grads[2] = db;
        (dx,dh_prev)

    }
}
