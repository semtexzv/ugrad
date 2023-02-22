use crate::ten::{ParamId, Shape, Tensor, TensorData};
use derivative::Derivative;
use std::fmt::Debug;
use std::ops::*;

#[derive(Debug, Clone)]
#[repr(u8)]
pub enum UnOp {
    Abs,
    Neg,
    Logn,
    Exp,
    Relu,
    Sign,
    Tanh,
    Sqrt,
    Reci,
    Gtz,
}

/// Per-element binary operation
#[derive(Debug, Clone)]
#[repr(u8)]
pub enum BinOp {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
#[repr(u8)]
pub enum RedOp {
    Sum,
    Max,
}

/// Special operation involving 2 matrices
#[derive(Debug, Clone)]
#[repr(u8)]
pub enum MatOp {
    Mul,
}

#[derive(Debug, Clone)]
#[repr(u8)]
pub enum ShapeOp {
    Broadcast { dim: usize, count: usize },
    Flip { dim: usize },
    Transpose { from: usize, to: usize },
}

fn fmt_tensor(t: &Tensor, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
    fmt.write_fmt(format_args!("{}", t.id()))
}

#[derive(Derivative)]
#[derivative(Debug)]
pub enum TensorOps {
    /// Special case for scalars. These are almost always inlinable
    Scalar(f32),
    Values(Vec<f32>),
    // Parameter, will be filled by uniform values
    Param(ParamId),
    UnOp(
        UnOp,
        #[derivative(Debug(format_with = "fmt_tensor"))] Tensor,
    ),
    BinOp(
        BinOp,
        #[derivative(Debug(format_with = "fmt_tensor"))] Tensor,
        #[derivative(Debug(format_with = "fmt_tensor"))] Tensor,
    ),
    ReduceOp(
        RedOp,
        usize,
        #[derivative(Debug(format_with = "fmt_tensor"))] Tensor,
    ),
    ShapeOp(
        ShapeOp,
        #[derivative(Debug(format_with = "fmt_tensor"))] Tensor,
    ),
    MatOp(
        MatOp,
        #[derivative(Debug(format_with = "fmt_tensor"))] Tensor,
        #[derivative(Debug(format_with = "fmt_tensor"))] Tensor,
    ),
}

impl<T: Into<Self>> Add<T> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: T) -> Self::Output {
        self._binop(rhs.into(), BinOp::Add)
    }
}

impl Add<f32> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f32) -> Self::Output {
        Tensor::value(rhs, self.shape()) + self
    }
}

impl<T> Add<T> for &Tensor
where
    Tensor: Add<T, Output = Tensor>,
{
    type Output = Tensor;

    fn add(self, rhs: T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: Into<Self>> AddAssign<T> for Tensor {
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into()
    }
}

impl<T: Into<Self>> Sub<T> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: T) -> Self::Output {
        self + -rhs.into()
    }
}

impl Sub<f32> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f32) -> Self::Output {
        let other = Tensor::value(rhs, self.shape());
        self - other
    }
}

impl<T> Sub<T> for &Tensor
where
    Tensor: Sub<T, Output = Tensor>,
{
    type Output = Tensor;

    fn sub(self, rhs: T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: Into<Self>> SubAssign<T> for Tensor {
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into()
    }
}

impl<T: Into<Self>> Mul<T> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: T) -> Self::Output {
        self._binop(rhs.into(), BinOp::Mul)
    }
}

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Self::Output {
        let rhs = Tensor::value(rhs, self.shape());
        self * rhs
    }
}

impl<T> Mul<T> for &Tensor
where
    Tensor: Mul<T, Output = Tensor>,
{
    type Output = Tensor;

    fn mul(self, rhs: T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: Into<Self>> MulAssign<T> for Tensor {
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into()
    }
}

impl<T: Into<Self>> Div<T> for Tensor {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self * rhs.into().reciprocal()
    }
}

impl Div<f32> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Self::Output {
        let rhs = Tensor::value(rhs, self.shape());
        self / rhs
    }
}

impl<T> Div<T> for &Tensor
where
    Tensor: Div<T, Output = Tensor>,
{
    type Output = Tensor;

    fn div(self, rhs: T) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: Into<Self>> DivAssign<T> for Tensor {
    fn div_assign(&mut self, rhs: T) {
        *self = self.clone() / rhs.into();
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self._unop(UnOp::Neg)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self.clone()._unop(UnOp::Neg)
    }
}

impl<T: Into<Self>> BitOr<T> for Tensor {
    type Output = Tensor;

    fn bitor(self, rhs: T) -> Self::Output {
        (self + rhs.into()).sign()
    }
}

impl<T: Into<Self>> BitOrAssign<T> for Tensor {
    fn bitor_assign(&mut self, rhs: T) {
        *self = self.clone() | rhs.into()
    }
}

impl<T: Into<Self>> BitAnd<T> for Tensor {
    type Output = Tensor;

    fn bitand(self, rhs: T) -> Self::Output {
        (self * rhs.into()).sign()
    }
}

impl<T: Into<Self>> BitAndAssign<T> for Tensor {
    fn bitand_assign(&mut self, rhs: T) {
        *self = self.clone() & rhs.into();
    }
}

pub trait TensorMethods: Into<Tensor> {
    fn shape(&self) -> &Shape;
    fn _unop(self, op: UnOp) -> Tensor;

    fn _binop(self, oth: Tensor, op: BinOp) -> Tensor;

    fn _redop(self, dim: isize, op: RedOp) -> Tensor;

    fn _shapop(self, op: ShapeOp) -> Tensor;

    fn _matop(self, oth: Tensor, op: MatOp) -> Tensor;

    fn abs(self) -> Tensor {
        self._unop(UnOp::Abs)
    }

    fn relu(self) -> Tensor {
        self._unop(UnOp::Relu)
    }

    fn exp(self) -> Tensor {
        self._unop(UnOp::Exp)
    }

    fn log(self) -> Tensor {
        self._unop(UnOp::Logn)
    }

    fn sign(self) -> Tensor {
        self._unop(UnOp::Sign)
    }

    fn gtz(self) -> Tensor {
        self._unop(UnOp::Gtz)
    }

    fn tanh(self) -> Tensor {
        self._unop(UnOp::Tanh)
    }

    fn sqrt(self) -> Tensor {
        self._unop(UnOp::Sqrt)
    }

    fn reciprocal(self) -> Tensor {
        self._unop(UnOp::Reci)
    }

    fn sum(self, dim: isize) -> Tensor {
        self._redop(dim, RedOp::Sum)
    }

    fn max(self, dim: isize) -> Tensor {
        self._redop(dim, RedOp::Max)
    }

    fn min(self, dim: isize) -> Tensor {
        -(-self.into()).max(dim)
    }

    fn matmul(self, o: Tensor) -> Tensor {
        self._matop(o, MatOp::Mul)
    }

    fn transpose(self, from: isize, to: isize) -> Tensor {
        let from = self.shape().wrap(from);
        let to = self.shape().wrap(to);
        self._shapop(ShapeOp::Transpose { from, to })
    }
}

impl TensorMethods for Tensor {
    fn shape(&self) -> &Shape {
        Tensor::shape(self)
    }

    fn _unop(self, op: UnOp) -> Tensor {
        Self::from(TensorData::new(
            self.grad(),
            self.shape().clone(),
            TensorOps::UnOp(op, self),
        ))
    }

    fn _binop(self, oth: Tensor, op: BinOp) -> Tensor {
        Self::from(TensorData::new(
            self.grad() | oth.grad(),
            self.shape().clone(),
            TensorOps::BinOp(op, self, oth),
        ))
    }

    fn _redop(self, dim: isize, op: RedOp) -> Tensor {
        let shape = self.shape();
        let dim = shape.wrap(dim);
        Self::from(TensorData::new(
            self.grad(),
            shape.set(dim as _, 1),
            TensorOps::ReduceOp(op, dim, self),
        ))
    }

    fn _shapop(self, op: ShapeOp) -> Tensor {
        let shape = self.shape();
        let shape = match op {
            ShapeOp::Broadcast { .. } => panic!(),
            ShapeOp::Transpose { from, to } => {
                let s = self.shape();
                s.set(from as _, s[to as isize])
                    .set(to as _, s[from as isize])
            }
            ShapeOp::Flip { .. } => {
                panic!()
            }
        };
        Self::from(TensorData::new(
            self.grad(),
            shape,
            TensorOps::ShapeOp(op, self),
        ))
    }

    fn _matop(self, oth: Tensor, op: MatOp) -> Tensor {
        let shape = Shape::from(&[self.shape()[-2], oth.shape()[-1]][..]);
        Self::from(TensorData::new(
            self.grad() | oth.grad(),
            shape,
            TensorOps::MatOp(op, self, oth),
        ))
    }
}

impl TensorMethods for &Tensor {
    fn shape(&self) -> &Shape {
        Tensor::shape(self)
    }

    fn _unop(self, op: UnOp) -> Tensor {
        self.clone()._unop(op)
    }

    fn _binop(self, oth: Tensor, op: BinOp) -> Tensor {
        self.clone()._binop(oth, op)
    }

    fn _redop(self, dim: isize, op: RedOp) -> Tensor {
        self.clone()._redop(dim, op)
    }

    fn _shapop(self, op: ShapeOp) -> Tensor {
        self.clone()._shapop(op)
    }

    fn _matop(self, oth: Tensor, op: MatOp) -> Tensor {
        self.clone()._matop(oth, op)
    }
}
