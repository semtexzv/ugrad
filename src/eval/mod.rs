use crate::back::{Backend, BufferT};
use crate::ten::{BinOp, ParamId, RedOp, Shape, Tensor, TensorOps, UnOp};
use defer::defer;
use ndarray::{ArcArray, ArrayBase, ArrayD, IxDyn, OwnedArcRepr};
use ndarray_rand::RandomExt;
use std::collections::{HashMap, HashSet};
use std::ops::Deref;

#[derive(Debug, Clone, thiserror::Error)]
pub enum ErrorKind {
    #[error("Wrong shape: {got:?} expected: {expected:?}")]
    WrongShape { expected: Shape, got: Shape },
    #[error("Unknown parameter: {name}")]
    UnknownParam { name: String },
}

pub type Error = Box<ErrorKind>;

#[derive(Debug)]
pub struct Evaluator<B: Backend> {
    pub grad: bool,
    pub b: B,
    pub buf: HashMap<u64, B::Buffer>,
    pub grads: HashMap<u64, Tensor>,

    // We're storing the last seen param values
    pub params: HashMap<ParamId, Tensor>,
    pub param_ids: HashSet<u64>,

    lastid: u64,
}

impl<B: Backend> Evaluator<B> {
    pub fn new(b: B) -> Self {
        Self {
            grad: true,
            b,
            buf: Default::default(),

            lastid: 1,
            params: Default::default(),
            grads: Default::default(),
            param_ids: Default::default(),
        }
    }

    fn genid(&mut self) -> u64 {
        self.lastid += 1;
        self.lastid
    }

    #[inline(always)]
    pub fn grad(&mut self, t: &Tensor) -> Result<&mut Tensor, Error> {
        let grad = self.grads.entry(t.id()).or_insert_with(|| {
            println!("mkgrad: {:?}", t.id());
            Tensor::zeros(t.shape())
        });

        if grad.shape() != t.shape() {
            panic!(
                "Grad shape mismatch: {:?}, tensor: {:?}",
                grad.shape(),
                t.shape()
            );
        }

        return Ok(grad);
    }

    fn assign_ids(&mut self, t: &Tensor) {
        let nest = if t.id() == 0 {
            t.set_id(self.genid());
            true
        } else {
            false
        };

        if self.grad && t.grad() {
            let gid = self.genid();
            let g = self.grad(t).unwrap();
            if g.id() == 0 {
                g.set_id(gid);
            }
        }
        match &t.0.ops {
            TensorOps::Scalar(_) => {}
            TensorOps::Values(_) => {}
            TensorOps::Param(p) => {
                // We can have param stored from previous invocation
                // Replace that tensor with new one
                self.params.insert(p.clone(), t.clone());
                self.param_ids.insert(t.id());
            }
            TensorOps::UnOp(_, x) if nest => {
                self.assign_ids(x);
            }
            TensorOps::BinOp(_, x, y) if nest => {
                self.assign_ids(x);
                self.assign_ids(y);
            }
            TensorOps::ReduceOp(_, _, x) if nest => {
                self.assign_ids(x);
            }
            TensorOps::ShapeOp(_, x) if nest => {
                self.assign_ids(x);
            }
            TensorOps::MatOp(_, x, y) if nest => {
                self.assign_ids(x);
                self.assign_ids(y);
            }
            TensorOps::UnOp(_, _) => {}
            TensorOps::BinOp(_, _, _) => {}
            TensorOps::ReduceOp(_, _, _) => {}
            TensorOps::ShapeOp(_, _) => {}
            TensorOps::MatOp(_, _, _) => {}
        }
    }

    pub fn evaluate(&mut self, t: &Tensor) {
        self.assign_ids(&t);
        self._eval(t);
    }

    /// Create a topological sort of the vector & it's dependencies.
    pub fn topo<'a>(&self, t: &'a Tensor) -> Vec<&'a Tensor> {
        fn _topo<'a>(t: &'a Tensor, n: &mut HashSet<u64>, l: &mut Vec<&'a Tensor>) {
            // Node already visited
            if n.contains(&t.id()) {
                return;
            }
            // Will visit
            n.insert(t.id());
            match &t.0.ops {
                TensorOps::Scalar(_) => {}
                TensorOps::Values(_) => {}
                TensorOps::Param(_) => {}
                TensorOps::UnOp(_, x) => _topo(x, n, l),
                TensorOps::BinOp(_, x, y) => {
                    _topo(x, n, l);
                    _topo(y, n, l);
                }
                TensorOps::ReduceOp(_, _, x) => {
                    _topo(x, n, l);
                }
                TensorOps::ShapeOp(_, x) => {
                    _topo(x, n, l);
                }
                TensorOps::MatOp(_, x, y) => {
                    _topo(x, n, l);
                    _topo(y, n, l);
                }
            }
            l.push(t);
        }
        let mut n = HashSet::new();
        let mut l = vec![];
        _topo(t, &mut n, &mut l);
        l
    }

    pub fn backward(&mut self, t: &Tensor) -> Result<(), Error> {
        self.assign_ids(t);
        let t = self.topo(t);
        for e in t.into_iter().rev() {
            e.backprop(self)?
        }

        self.no_grad(|e| {
            for p in &{ e.params.values().cloned().collect::<Vec<_>>() } {
                let g = e.grads.get(&p.id()).unwrap().clone();
                e.assign_ids(&g);
                e._eval(&g);
            }
        });

        Ok(())
    }

    pub fn clear(&mut self) {
        self.clear_bufs();
        self.zero_grads();
    }

    // We use dynamic buffers for grads. Just delete them after
    pub fn zero_grads(&mut self) {
        self.grads.clear();
    }

    /// Clears buffers for anything that's not a parameter of the model.
    /// TODO: Maybe keep some of the buffers around?
    pub fn clear_bufs(&mut self) {
        self.buf.retain(|b, _| self.param_ids.contains(b));
        self.param_ids.clear();
    }

    pub fn no_grad<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        self.grad = false;
        let mut out = f(self);
        defer(|| self.grad = true);
        out
    }

    /// Adds a calculation for specified
    fn _eval(&mut self, t: &Tensor) {
        let i = t.0.deref();
        let id = i.id.get();
        assert_ne!(id, 0);

        if self.buf.contains_key(&id) {
            return;
        }

        match &i.ops {
            TensorOps::Scalar(scl) => {
                let mut data = ArcArray::<f32, IxDyn>::from_elem::<ndarray::Shape<IxDyn>>(
                    (&i.shape).into(),
                    *scl,
                );
                let mut buf = self.b.buffer(&i.shape);
                buf.upload(&mut self.b, data);

                self.buf.insert(id, buf);
            }
            TensorOps::Values(v) => {
                let mut data = ArcArray::<f32, IxDyn>::from_shape_vec::<ndarray::Shape<IxDyn>>(
                    (&i.shape).into(),
                    v.clone(),
                )
                .unwrap();
                let mut buf = self.b.buffer(&i.shape);
                buf.upload(&mut self.b, data);

                self.buf.insert(id, buf);
            }
            TensorOps::Param(pid) => {
                let mut dist = rand::distributions::Standard;

                let mut data: ArrayBase<OwnedArcRepr<f32>, IxDyn> =
                    ArrayBase::<OwnedArcRepr<f32>, IxDyn>::random::<ndarray::Shape<IxDyn>, _>(
                        (&i.shape).into(),
                        dist,
                    );

                // This param already has buffer, skip it
                if self.buf.contains_key(&id) {
                    println!("Skipping param buffer creation");
                    return;
                }

                // Insert new buffer.
                let mut buf = self.b.buffer(&i.shape);
                buf.upload(&mut self.b, data);
                self.buf.insert(id, buf);
            }
            TensorOps::UnOp(op, x) => {
                self._eval(x);
                let xbuf = self.buf.get(&x.id()).unwrap();
                self.buf
                    .insert(id, self.b.unop(op, xbuf, x.shape(), &i.shape));
            }
            TensorOps::BinOp(op, x, y) => {
                self._eval(x);
                self._eval(y);
                let xbuf = self.buf.get(&x.id()).unwrap();
                let ybuf = self.buf.get(&y.id()).unwrap();
                self.buf.insert(
                    id,
                    self.b.binop(op, xbuf, x.shape(), ybuf, y.shape(), &i.shape),
                );
            }
            TensorOps::ReduceOp(op, dim, x) => {
                self._eval(x);
                let xbuf = self.buf.get(&x.id()).unwrap();
                self.buf
                    .insert(id, self.b.redop(op, *dim, xbuf, x.shape(), &i.shape));
            }
            TensorOps::ShapeOp(op, x) => {
                self._eval(x);
                let xbuf = self.buf.get(&x.id()).unwrap();
                self.buf
                    .insert(id, self.b.shapop(op, xbuf, x.shape(), &i.shape));
            }
            TensorOps::MatOp(op, x, y) => {
                self._eval(x);
                self._eval(y);
                let xbuf = self.buf.get(&x.id()).unwrap();
                let ybuf = self.buf.get(&y.id()).unwrap();
                self.buf.insert(
                    id,
                    self.b.matop(op, xbuf, x.shape(), ybuf, y.shape(), &i.shape),
                );
            }
        }
    }
}
