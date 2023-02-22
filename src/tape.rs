use bumpalo::Bump;
use std::fmt::{Debug, Formatter};
use std::mem::take;

#[ouroboros::self_referencing]
pub struct Tape<T> {
    /// Arena storing our closure data for backward passes. Each entry is variable size closure data.
    alloc: Bump,

    #[covariant]
    #[borrows(alloc)]
    /// Sequence of backward passes. Each entry here is 24 bytes (data ptr, fn ptr, allocator ptr)
    ops: Vec<Box<dyn FnOnce(&mut T), &'this Bump>>,
}

impl<T> Default for Tape<T> {
    fn default() -> Self {
        Self::new(Bump::new(), |b| Default::default())
    }
}

impl<T> Debug for Tape<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.with(|fld| {
            f.debug_struct("Tape")
                .field("bytes", &fld.alloc.allocated_bytes())
                .field("ops", &fld.ops.len())
                .finish()
        })
    }
}

impl<T> Tape<T> {
    /// Record a new action to be replayed. Since we're using CoW values,
    /// we can implicitly pass the gradient indexes & variable values into the closures
    pub fn push<F>(&mut self, f: F)
    where
        for<'a> F: FnOnce(&'a mut T) + 'static,
    {
        self.with_mut(|fields| {
            let op = Box::new_in(f, fields.alloc);
            fields.ops.push(op);
        });
    }

    /// Run all backwards passes in reversed order. (back to front)
    pub fn run_all_reversed(mut self, g: &mut T) {
        self.with_mut(|fields| {
            let ops = take(fields.ops);
            for op in ops.into_iter().rev() {
                op(g)
            }
        });
        // Tape is dropped here along with the bump allocator
    }
}

#[cfg(test)]
mod test {
    use crate::tape::Tape;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_trace_drops_correctly() {
        let mut trace = Tape::default();
        let v = Arc::new(AtomicUsize::new(0));

        #[derive(Debug)]
        struct D {
            v: Arc<AtomicUsize>,
        }

        impl Drop for D {
            fn drop(&mut self) {
                self.v.store(1, Ordering::SeqCst);
            }
        }
        let d = D { v: v.clone() };

        trace.push(move |grad: &mut ()| {
            println!("{:?}", d);
        });

        drop(trace);
        assert_eq!(v.load(Ordering::SeqCst), 1);
    }
}
