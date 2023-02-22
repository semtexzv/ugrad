pub mod wgpu;

use crate::ten::{BinOp, MatOp, RedOp, Shape, ShapeOp, UnOp};
use ndarray::{ArcArray, IxDyn};

pub trait Backend: Sized {
    type Buffer: BufferT<Self>;
    /// Allocate a buffer ready for COPY_READ | COPY_WRITE | STORAGE
    fn buffer(&mut self, shape: &Shape) -> Self::Buffer;

    /// Allocate output buffer, perform an operation
    fn unop(&mut self, op: &UnOp, x: &Self::Buffer, xsh: &Shape, osh: &Shape) -> Self::Buffer;
    fn binop(
        &mut self,
        op: &BinOp,
        x: &Self::Buffer,
        xsh: &Shape,
        y: &Self::Buffer,
        ysh: &Shape,
        osh: &Shape,
    ) -> Self::Buffer;

    fn redop(
        &mut self,
        op: &RedOp,
        dim: usize,
        x: &Self::Buffer,
        xsh: &Shape,
        osh: &Shape,
    ) -> Self::Buffer;

    fn shapop(&mut self, op: &ShapeOp, x: &Self::Buffer, xsh: &Shape, osh: &Shape) -> Self::Buffer;

    fn matop(
        &mut self,
        op: &MatOp,
        x: &Self::Buffer,
        xsh: &Shape,
        y: &Self::Buffer,
        ysh: &Shape,
        osh: &Shape,
    ) -> Self::Buffer;
}

/// Every tensor implementation must be able to materialize the generated tensor
pub trait BufferT<E: Backend<Buffer = Self>>: Sized {
    fn upload(&self, e: &mut E, n: ArcArray<f32, IxDyn>);
    fn download(&self, shape: &Shape, e: &mut E) -> ArcArray<f32, IxDyn>;
}
