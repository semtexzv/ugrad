use defer::defer;
use crate::back::{Backend, BufferT};
use crate::eval::Evaluator;

pub trait Optimizer {
    fn step<E: Backend>(&mut self, e: &mut Evaluator<E>);
}

pub struct SGD {
    lr: f32,
    momentum: f32,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
        }
    }
}

impl Optimizer for SGD {
    fn step<E: Backend>(&mut self, e: &mut Evaluator<E>) {
        e.no_grad(|e| {
            let mut params = vec![];
            for (pid, p) in &mut e.params {
                let mut g = e.grads.get(&p.id()).unwrap().clone();
                // We're replacing the parameter tensor by calcucation tensor
                // And evaluating all of them later.
                // On next iteration of the model, the tensor will be replaced by
                // the param one once again. Not ideal, but better than nothing.
                *p += g * self.lr;
                params.push(p.clone());
            }
            for p in params {
                e.evaluate(&p);
            }
            e.clear();
            e.clear_bufs();
            e.zero_grads();
        });
    }
}