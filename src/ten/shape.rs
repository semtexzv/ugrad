use ndarray::{Dimension, IxDyn, ShapeBuilder};
use std::cmp::max;
use std::ops::{Index, Range, RangeFrom, RangeFull, RangeTo};

#[macro_export]
macro_rules! shape {
    ($($s:expr),* $(,)?) => {
        $crate::ten::shape::Shape::from(vec![ $($s),*])
    };
}

#[derive(Default, Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn remove(&self, i: isize) -> Shape {
        let mut d = self.dims.clone();
        let p = self.wrap(i);
        d.remove(p);
        Shape { dims: d }
    }
    pub fn set(&self, i: isize, dim: usize) -> Shape {
        let mut d = self.dims.clone();
        let p = self.wrap(i);
        d[p] = dim;
        Shape { dims: d }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self { dims: value }
    }
}

impl From<&Shape> for Shape {
    fn from(value: &Shape) -> Self {
        value.clone()
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self {
            dims: value.to_vec(),
        }
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(value: &[usize; N]) -> Self {
        Self {
            dims: value.to_vec(),
        }
    }
}
impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self {
            dims: value.to_vec(),
        }
    }
}

impl From<&ndarray::Shape<IxDyn>> for Shape {
    fn from(value: &ndarray::Shape<IxDyn>) -> Self {
        Self::from(value.raw_dim().slice())
    }
}

impl From<&Shape> for ndarray::Shape<IxDyn> {
    fn from(value: &Shape) -> Self {
        value.dims.as_slice().into_shape()
    }
}

impl Shape {
    pub fn wrap(&self, i: isize) -> usize {
        if i < 0 {
            max(self.dims.len() as isize + i, 0) as usize
        } else {
            i as usize
        }
    }
}

impl Index<isize> for Shape {
    type Output = usize;

    fn index(&self, index: isize) -> &Self::Output {
        &self.dims[self.wrap(index)]
    }
}

impl Index<Range<isize>> for Shape {
    type Output = [usize];

    fn index(&self, index: Range<isize>) -> &Self::Output {
        &self.dims[self.wrap(index.start)..self.wrap(index.end)]
    }
}

impl Index<RangeFrom<isize>> for Shape {
    type Output = [usize];

    fn index(&self, index: RangeFrom<isize>) -> &Self::Output {
        &self.dims[self.wrap(index.start)..]
    }
}

impl Index<RangeTo<isize>> for Shape {
    type Output = [usize];

    fn index(&self, index: RangeTo<isize>) -> &Self::Output {
        &self.dims[..self.wrap(index.end)]
    }
}

impl Index<RangeFull> for Shape {
    type Output = [usize];

    fn index(&self, index: RangeFull) -> &Self::Output {
        &self.dims[..]
    }
}

// impl<R: RangeBounds<isize>> Index<R> for Shape {
//     type Output = [usize];
//     #[inline(always)]
//     fn index(&self, index: R) -> &Self::Output {
//         let s = index.start_bound().map(|i| self.wrap(i));
//         let e = index.end_bound().map(map);
//         println!("{:?}, {:?}", s, e);
//         &self.dims[(s, e)]
//     }
// }

impl Shape {
    pub fn prod(&self) -> usize {
        self.dims.iter().product()
    }
}
