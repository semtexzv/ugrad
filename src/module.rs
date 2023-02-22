use crate::ten::{Shape, Tensor, TensorMethods};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ops::{Add, BitOr, Deref};

trait ModuleInput {
    /// What shape each element in a batch has.
    /// Within an invocation, different inputs might have different batch size
    type Shapes;
    fn zero(s: &Self::Shapes) -> Self;
}

impl ModuleInput for Tensor {
    type Shapes = Shape;

    fn zero(s: &Self::Shapes) -> Self {
        Tensor::zeros(s)
    }
}

impl ModuleInput for (Tensor,) {
    type Shapes = (Shape,);

    fn zero(s: &Self::Shapes) -> Self {
        (Tensor::zeros(&s.0),)
    }
}

impl ModuleInput for (Tensor, Tensor) {
    type Shapes = (Shape, Shape);

    fn zero(s: &Self::Shapes) -> Self {
        (Tensor::zeros(&s.0), Tensor::zeros(&s.1))
    }
}

trait Module<Input: ModuleInput> {
    type Output;

    fn forward(&mut self, i: Input) -> Self::Output;
}

impl<Input: ModuleInput, T, O> Module<Input> for T
where
    T: FnMut(Input) -> O,
{
    type Output = O;

    fn forward(&mut self, i: Input) -> Self::Output {
        (self)(i)
    }
}

pub struct Ten<T> {
    _t: PhantomData<T>,
}

pub trait Eval {
    fn grad(&self) -> bool;

    fn mkid(&mut self) -> u64;
    fn saved(&mut self, id: u64);
    fn add_grad(&mut self, e1: u64, g: &Ten<GradType>);
}

// All exprs are expected to be internally mutable
trait Expr<T> {
    fn id(&self) -> u64;
    fn forward(&self, e: &mut dyn Eval) -> Ten<T>;
    fn backward(&self, e: &mut dyn Eval, out_grad: &Ten<GradType>) {}
    fn boxed(self) -> Box<dyn Expr<T>>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
    fn cloned(&self) -> Self
    where
        Self: Sized,
    {
        todo!()
    }
}

trait IntoExpr<T> {
    type Expr: Expr<T>;
    fn into_expr(self) -> Self::Expr;
}

impl<T, E> IntoExpr<T> for E where E: Expr<T> {
    type Expr = E;

    fn into_expr(self) -> Self::Expr {
        self
    }
}

impl<T, E> IntoExpr<T> for &E where E: Expr<T> {
    type Expr = E;

    fn into_expr(self) -> Self::Expr {
        self.cloned()
    }
}

type GradType = f32;

struct ExprBase<T, S> {
    _pd: PhantomData<T>,
    _id: Cell<u64>,
    _saved: RefCell<Option<S>>,
}

impl<T, S> Default for ExprBase<T, S> {
    fn default() -> Self {
        Self {
            _pd: PhantomData,
            _id: Cell::new(0),
            _saved: RefCell::new(None),
        }
    }
}

impl<T, S> ExprBase<T, S> {
    fn save(&self, v: S) {
        self._saved.replace_with(|_| Some(v));
    }
    fn read(&self) -> Option<S> {
        self._saved.take().map(|s| s)
    }
    fn forward(&self, e: &mut dyn Eval) {
        // Assign ids at the start
        if self._id.get() == 0 {
            self._id.set(e.mkid());
        }
    }
    fn backward<F>(&self, e: &mut dyn Eval, f: F)
    where
        F: FnOnce(&mut dyn Eval, u64, S),
    {
        if let Some(s) = self._saved.take() {
            f(e, self._id.get(), s);
        } else {
            panic!("Backwards invoked without forwards")
        }
    }
}

pub struct Param<T> {
    _t: PhantomData<T>,
}

impl<T> Clone for Param<T> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<T> Expr<T> for Param<T> {
    fn id(&self) -> u64 {
        0
    }

    fn forward(&self, e: &mut dyn Eval) -> Ten<T> {
        todo!()
    }
}

pub struct Binary<T, L, R> {
    _base: ExprBase<T, (Ten<T>, Ten<T>)>,
    op: u8,
    l: L,
    r: R,
}

impl<T, L, R> Clone for Binary<T, L, R>
where
    L: Expr<T>,
    R: Expr<T>,
{
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<T, L, R> Expr<T> for Binary<T, L, R>
where
    L: Expr<T>,
    R: Expr<T>,
{
    fn id(&self) -> u64 {
        self._base._id.get()
    }

    fn forward(&self, e: &mut dyn Eval) -> Ten<T> {
        self._base.forward(e);
        let x = self.l.forward(e);
        let y = self.r.forward(e);
        if e.grad() {
            e.saved(self.l.id());
            e.saved(self.r.id());
            self._base.save((x, y));
        }
        todo!()
    }
    fn backward(&self, e: &mut dyn Eval, out_grad: &Ten<GradType>) {
        if let Some((x, y)) = self._base.read() {
            // e.add_grad(self.l.id(), y * out_grad);
            // e.add_grad(self.r.id(), x * out_grad);
        }
    }
}

pub struct Unary<T, L> {
    _t: PhantomData<T>,
    op: u8,
    l: L,
}

impl<T, L> Clone for Unary<T, L> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<T, L> Expr<T> for Unary<T, L> {
    fn id(&self) -> u64 {
        todo!()
    }

    fn forward(&self, e: &mut dyn Eval) -> Ten<T> {
        todo!()
    }
}

macro_rules! impl_ops {
    ($name:ident $(,)? $($g:ident),* $(; where $($bounded:ty : $bound:path$(, $others:path)*);*)?) => {
        impl<T, Other: Expr<T>, $($g),*> Add<Other> for $name<T, $($g),*>
            $(where $($bounded: $bound $(+ $others)*),*)?
        {
            type Output = Binary<T, Self, Other>;

            fn add(self, rhs: Other) -> Self::Output {
                Binary {
                    _base: Default::default(),
                    op: 0,
                    l: self,
                    r: rhs,
                }
            }
        }
    };
}

struct MatMul<T, L, R>
where
    L: Expr<T>,
    R: Expr<T>,
{
    _p: PhantomData<T>,
    l: L,
    r: R,
}

impl<T, L, R> Clone for MatMul<T, L, R>
where
    L: Expr<T>,
    R: Expr<T>,
{
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<T, L, R> Expr<T> for MatMul<T, L, R>
where
    L: Expr<T>,
    R: Expr<T>,
{
    fn id(&self) -> u64 {
        todo!()
    }

    fn forward(&self, e: &mut dyn Eval) -> Ten<T> {
        todo!()
    }
}

trait MatMulT<T>: Expr<T> {
    fn matmul<O: IntoExpr<T>>(self, o: O) -> MatMul<T, Self, O::Expr>
    where
        Self: Sized;
}

impl<T, E: Expr<T>> MatMulT<T> for E {
    fn matmul<O: IntoExpr<T>>(self, o: O) -> MatMul<T, Self, O::Expr> {
        todo!()
    }
}

impl_ops!(MatMul, L, R; where L: Expr<T>; R: Expr<T>);
impl_ops!(Binary, L, R);
impl_ops!(Unary, L);
impl_ops!(Param);

fn verify() -> impl Expr<f32> + 'static {
    let x = Param::<f32> { _t: PhantomData } + Param { _t: PhantomData };
    let o = x.clone().matmul(&x);
    // println!("{:?}", type_name_of_val(&o))
    // let y = &x + x;
    o
}

// fn model() -> impl Module<Tensor> {

// let p = Tensor::param("a", shape![0, 1]);
// let y = p.clone().matmul(&p);
// move |i: Tensor| (i.clone() + p.clone(), i)
// }

// fn process<I: ModuleInput, B: Backend>(
//     e: &mut Evaluator<B>,
//     mut m: impl Module<I, Output = Tensor>,
//     shapes: I::Shapes,
// ) {
//     let mut inputs = I::zero(&shapes);
//     let mut out = m.forward(inputs);
//     e.evaluate(&out);
//
//     // Generate the graph
// }

// #[cfg(test)]
// mod test {
//     use crate::back::wgpu::WgpuBackend;
//     use crate::module::process;
//     use crate::shape;
//     use crate::ten::Tensor;
//
//     #[test]
//     fn test_mod() {
//         let mut module = |x: Tensor| &x + &x;
//
//         let mut out = process::<_, WgpuBackend>(unimplemented!(), module, shape![1, 10]);
//     }
// }
