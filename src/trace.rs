use crate::tape::Tape;
use crate::Value;
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem::take;
use std::sync::Arc;

#[track_caller]
pub fn id() -> u64 {
    let mut h = fnv::FnvHasher::with_key(0);
    std::panic::Location::caller().hash(&mut h);
    h.finish()
}

pub trait Init<V> {
    fn init<R: Rng>(self, rand: &mut R) -> Value;
}

struct Uniform;

impl Init<Value> for Uniform {
    fn init<R: Rng>(self, rand: &mut R) -> Value {
        Value {
            v: Arc::new(rand::distributions::Uniform::new(-1.0f32, 1.0f32).sample(rand)),
            id: 0,
        }
    }
}

pub trait Trace<V> {
    #[track_caller]
    fn id(&mut self) -> u64;
    #[track_caller]
    fn param(&mut self) -> V;
    #[track_caller]
    fn param_init<I: Init<V>>(&mut self, i: I) -> V;
    #[track_caller]
    fn scope<F: FnOnce(&mut Self)>(&mut self, name: &'static str, act: F);
    #[track_caller]
    fn backop<F: FnOnce(&mut Self) + 'static>(&mut self, f: F);
}

pub trait ArithOps<V> {
    /// Implements addition.
    fn add(g: &mut Record<V>, x: V, y: V) -> V;
    fn mul(g: &mut Record<V>, x: V, y: V) -> V;
    fn neg(g: &mut Record<V>, x: V) -> V;
}

pub trait Arith<V>: Trace<V> {
    #[track_caller]
    fn add(&mut self, x: V, y: V) -> V;
    #[track_caller]
    fn mul(&mut self, x: V, y: V) -> V;
    #[track_caller]
    fn identity(&mut self, v: V) -> V {
        v
    }
}

pub trait Activ<V>: Arith<V> {
    fn relu(&mut self, v: V) -> V;
}

pub struct Record<V> {
    record: bool,
    rng: rand::rngs::ThreadRng,
    _id: u64,

    // Gradients.
    pub grad: HashMap<u64, V>,

    // Which gradients are actual parameters
    pub params: HashSet<u64>,

    // Tape of operations
    pub tape: Tape<Self>,
}

impl Record<Value> {
    pub fn new() -> Self {
        Self {
            record: true,
            rng: thread_rng(),
            _id: 0,

            grad: Default::default(),

            params: Default::default(),
            tape: Tape::default(),
        }
    }
    #[track_caller]
    pub fn bwop<FW, BW>(&mut self, f: FW) -> u64
        where
            FW: FnOnce(&mut Self, u64) -> BW,
            BW: FnOnce(&mut Self) + 'static,
    {
        if self.record {
            let id = self.id();
            let op = f(self, id);
            self.backop(op);
            id
        } else {
            0
        }
    }

    /// Run backwards pass on the
    pub fn backward(&mut self, v: Value) {
        self.record = false;
        self.grad.insert(v.id, Value::from(1.0));
        let tape = take(&mut self.tape);
        tape.run_all_reversed(self);
    }

    pub fn add_grad(&mut self, id: u64, v: Value) {
        if id != 0 {
            let g = self.grad.entry(id).or_default().clone();
            let v = self.add(g.clone(), v);
            self.grad.insert(id, v);
        }
    }

    pub fn done(&mut self, id: u64) {
        if !self.params.contains(&id) {
            self.grad.remove(&id);
        }
    }
}

impl Trace<Value> for Record<Value> {
    #[track_caller]
    fn id(&mut self) -> u64 {
        if self.record {
            self._id += 1;
            self._id
        } else {
            0
        }
    }

    #[track_caller]
    fn param(&mut self) -> Value {
        self.param_init(Uniform)
    }

    fn param_init<I: Init<Value>>(&mut self, i: I) -> Value {
        let id = self.id();
        self.params.insert(id);
        let mut out = i.init(&mut self.rng);
        out.id = id;

        out
    }

    fn scope<F: FnOnce(&mut Self)>(&mut self, name: &'static str, act: F) {
        panic!("")
    }

    fn backop<F: FnOnce(&mut Self) + 'static>(&mut self, f: F) {
        self.tape.push(f)
    }
}

impl Arith<Value> for Record<Value> {
    // y = a + b, da = dy, db = dy
    #[track_caller]
    fn add(&mut self, x: Value, y: Value) -> Value {
        let id = self.id();
        if self.record {
            let xg = x.id;
            let yg = y.id;
            self.backop(move |g| {
                println!("Add Backward");
                g.add_grad(xg, g.grad[&id].clone());
                g.add_grad(yg, g.grad[&id].clone());
                g.done(id);
            });
        };

        Value {
            v: Arc::new(*x.v + *y.v),
            id,
        }
    }

    #[track_caller]
    fn mul(&mut self, x: Value, y: Value) -> Value {
        let id = self.id();
        if self.record {
            let x = x.clone();
            let y = y.clone();
            self.backop(move |g| {
                println!("Mul Backward");
                let g1 = g.mul(y.clone(), g.grad[&id].clone());
                let g2 = g.mul(x.clone(), g.grad[&id].clone());

                g.add_grad(x.id, g1);
                g.add_grad(y.id, g2);
                g.done(id);
            });
        };
        Value {
            v: Arc::new(*x.v * *y.v),
            id,
        }
    }
}

impl Activ<Value> for Record<Value> {
    #[track_caller]
    fn relu(&mut self, v: Value) -> Value {
        let id = self.id();
        if self.record {
            let v = v.clone();
            self.backop(move |g: &mut Self| {
                println!("Relu backwards");
                let gd = g.mul(
                    g.grad[&id].clone(),
                    Value::from(if *v.v > 0.0 { 1.0 } else { 0.0 }),
                );
                g.add_grad(v.id, gd);
                g.done(id);
            });
        }

        Value {
            v: Arc::new(if *v.v > 0.0 { *v.v } else { 0.0 }),
            id,
        }
    }
}
