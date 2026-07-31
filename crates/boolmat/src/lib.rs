mod bits;
mod matrix;
mod word;

#[cfg(any(test, feature = "reference"))]
pub mod reference;

pub use bits::BitVec;
pub use matrix::BitMatrix;
pub use word::{Word, NUMBER_OF_BITS};
