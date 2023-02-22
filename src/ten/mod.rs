pub mod ops;
pub mod shape;

pub use ops::*;
pub use shape::*;

use crate::back::{Backend, BufferT};
use crate::eval::Evaluator;
use ndarray::IxDyn;
use std::cell::Cell;
use std::fmt::Debug;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Index, Mul,
    MulAssign, Neg, RangeBounds, Sub, SubAssign,
};
use std::rc::Rc;

pub type BufferId = u64;
pub type ParamId = String;

#[derive(Debug)]
pub struct TensorData {
    pub(crate) id: Cell<u64>,
    pub(crate) grad: Cell<bool>,
    pub(crate) shape: Shape,
    pub(crate) ops: TensorOps,
}

impl TensorData {
    pub fn new(grad: bool, shape: Shape, ops: TensorOps) -> Self {
        Self {
            id: Cell::new(0),
            grad: Cell::new(grad),
            shape,
            ops,
        }
    }
}

/// Lazily coputed tensor
///
/// Each operation on tensor creates a new tensor, building up a giant
/// DAG of operations, that are then scheduled and executed using on
/// a specific backend.
///
/// Shape checking is performed when building the graph.
#[derive(Debug, Clone)]
pub struct Tensor(pub(crate) Rc<TensorData>);

impl From<TensorData> for Tensor {
    fn from(value: TensorData) -> Self {
        Self(Rc::new(value))
    }
}

impl From<&Tensor> for Tensor {
    fn from(value: &Tensor) -> Self {
        value.clone()
    }
}

impl Tensor {
    pub fn id(&self) -> u64 {
        self.0.id.get()
    }
    pub fn set_id(&self, id: u64) {
        self.0.id.set(id)
    }

    pub fn grad(&self) -> bool {
        self.0.grad.get()
    }

    pub fn shape(&self) -> &Shape {
        &self.0.shape
    }

    pub fn param(id: &str, shape: impl Into<Shape>) -> Tensor {
        TensorData {
            // Don't know what ID. provided by the context
            id: Cell::new(0),
            // Will require grad
            grad: Cell::new(true),
            shape: shape.into(),
            ops: TensorOps::Param(id.to_string()),
        }
        .into()
    }

    pub fn empty() -> Tensor {
        Tensor::from(TensorData::new(
            false,
            Shape::default(),
            TensorOps::Scalar(1.0),
        ))
    }

    pub fn zeros(shape: impl Into<Shape>) -> Tensor {
        Tensor::from(TensorData::new(false, shape.into(), TensorOps::Scalar(0.0)))
    }

    pub fn ones(shape: impl Into<Shape>) -> Tensor {
        Tensor::from(TensorData::new(false, shape.into(), TensorOps::Scalar(1.0)))
    }

    pub fn value(v: f32, shape: impl Into<Shape>) -> Tensor {
        Tensor::from(TensorData::new(false, shape.into(), TensorOps::Scalar(v)))
    }

    pub fn values(v: Vec<f32>, shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        assert_eq!(v.len(), shape.prod());

        Tensor::from(TensorData::new(false, shape, TensorOps::Values(v)))
    }

    pub fn evaluate<E: Backend>(&self, e: &mut Evaluator<E>) {
        e.evaluate(&self);
    }

    pub(crate) fn backprop<E: Backend>(
        &self,
        e: &mut Evaluator<E>,
    ) -> Result<(), crate::eval::Error> {
        if !self.grad() || !e.grad {
            return Ok(());
        }
        match &self.0.ops {
            TensorOps::Scalar(_) => {}
            TensorOps::Param { .. } => {}
            TensorOps::UnOp(UnOp::Abs, x) => {
                if x.grad() {
                    let out = e.grad(self)?.clone();
                    // y = |x|   dy = x/|x|
                    *e.grad(x)? += x.clone() / self.clone() * out;
                }
            }
            TensorOps::UnOp(UnOp::Neg, x) => {
                if x.grad() {
                    let out = e.grad(self)?.clone();
                    // y = -x   dy = -dx
                    *e.grad(x)? += -out;
                }
            }
            TensorOps::UnOp(UnOp::Logn, x) => {
                if x.grad() {
                    let out = e.grad(self)?.clone();
                    // y = log(x) => dy = 1/x
                    *e.grad(x)? += x.clone().reciprocal() * out;
                }
            }
            TensorOps::UnOp(UnOp::Exp, x) => {
                if x.grad() {
                    let out = e.grad(self)?.clone();
                    // y = e^x   dy = e^x
                    *e.grad(x)? += self.clone() * out;
                }
            }
            TensorOps::UnOp(UnOp::Relu, x) => {
                if x.grad() {
                    let out = e.grad(self)?.clone();
                    // Gets us 1 where x > 0
                    *e.grad(x)? += x.clone().gtz() * out;
                }
            }
            TensorOps::UnOp(UnOp::Sign, x) => {
                if x.grad() {
                    // Derivation is zero. Almost everywhere, undefined at y
                    e.grad(x)?;
                }
            }
            TensorOps::UnOp(UnOp::Tanh, x) => {
                if x.grad() {
                    let out = e.grad(self)?.clone();
                    // y = tanh x => dy = 1 (tanh(x))^2
                    *e.grad(x)? += (Tensor::ones(self.shape()) - self.clone() * self.clone()) * out;
                }
            }
            TensorOps::UnOp(UnOp::Sqrt, x) => {
                if x.grad() {
                    let out = e.grad(self)?.clone();
                    // y = sqrt(x) => dy = 1/(2 * sqrt(x))
                    *e.grad(x)? += x.clone().sqrt() * Tensor::value(0.5, x.shape()) * out;
                }
            }
            TensorOps::UnOp(UnOp::Reci, x) => {
                if x.grad() {
                    // let out = e.grad(self)?.clone();
                    // *e.grad(x)? += (Tensor::ones(self.shape()) - self.clone() * self.clone()) + out;
                }
            }
            TensorOps::BinOp(BinOp::Add, x, y) => {
                let out = e.grad(self)?.clone();
                if x.grad() {
                    *e.grad(x)? += out.clone();
                }
                if y.grad() {
                    *e.grad(y)? += out;
                }
            }
            TensorOps::BinOp(BinOp::Mul, x, y) => {
                let out = e.grad(self)?.clone();
                if x.grad() {
                    *e.grad(x)? += y.clone() * out.clone();
                }
                if y.grad() {
                    *e.grad(y)? += x.clone() * out;
                }
            }
            TensorOps::ReduceOp(_, _, _) => {}
            TensorOps::MatOp(MatOp::Mul, x, y) => {
                panic!()
            }
            _ => {
                panic!()
            }
        }
        Ok(())
    }

    pub fn backward<E: Backend>(&self, e: &mut Evaluator<E>) -> Result<(), crate::eval::Error> {
        e.evaluate(&self);
        e.backward(&self);

        Ok(())
    }

    pub fn download<E: Backend>(&self, e: &mut Evaluator<E>) -> ndarray::ArcArray<f32, IxDyn> {
        self.evaluate(e);
        e.buf
            .get(&self.id())
            .clone()
            .unwrap()
            .download(self.shape(), &mut e.b)
    }
}
