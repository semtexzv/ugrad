#![feature(allocator_api)]
#![feature(box_syntax)]
#![feature(bound_map)]
#![feature(map_many_mut)]
#![feature(negative_impls)]

pub mod back;
pub mod eval;
pub mod optim;
pub mod ten;
pub mod module;

use crate::back::{wgpu::WgpuBackend, Backend, BufferT};
use crate::eval::Evaluator;
use crate::optim::{Optimizer, SGD};
use crate::ten::{Shape, Tensor, TensorMethods};
use rand::Rng;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ops::{Add, DerefMut, Mul, Neg};

fn main() {
    env_logger::init();
    let mut backend = pollster::block_on(WgpuBackend::new());
    let mut eval = Evaluator::new(backend);

    let x = Tensor::ones(shape![3, 1]) * 2.0;
    let y = Tensor::values((2..5).map(|v| v as f32).collect(), shape![1, 3]);
    let out = x.clone().matmul(y.clone());
    let out = &out + &out;
    let out2 = out.clone();

    eval.evaluate(&out2);

    let out = out.download(&mut eval);
    let out2 = out2.download(&mut eval);
    let x = x.download(&mut eval);
    let y = y.download(&mut eval);

    println!("{x:?}\n{y:?}\n{out:?}\n{out2:?}");
}
