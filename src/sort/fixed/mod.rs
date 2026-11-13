//! Run format for fixed-size records (`K`-byte keys, `V`-byte values).
//!
//! Run files hold packed `[key][value]` records with no headers, and the final
//! merge can write the sorted result directly to a packed output file
//! (`SorterCore::set_packed_output`).

pub mod merge;
pub mod run;
pub mod sorter;
