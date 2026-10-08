//! The compiled-in model manifest.
//!
//! Every value here is a pin, not a lookup. The revision is a commit hash
//! rather than a branch, the digests were read off the repository at that
//! revision, and nothing is ever fetched from the network to decide what a
//! valid file is. A later checkpoint generation lands beside this one instead
//! of replacing it, which is what the `v1/` directory is for.

use neunote_types::ModelSize;

pub const MODEL_REPO: &str = "DamRsn/muscriptor-gguf";
pub const MODEL_REVISION: &str = "d7045f94e8b19427f4ff9542975035e66596e51c";
pub const MODEL_DIRECTORY: &str = "v1";

/// One downloadable model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelEntry {
    pub size: ModelSize,
    pub file_name: &'static str,
    pub num_bytes: u64,
    pub sha256: &'static str,
}

const SMALL: ModelEntry = ModelEntry {
    size: ModelSize::Small,
    file_name: "muscriptor-small-f16.gguf",
    num_bytes: 209_425_152,
    sha256: "925f55af65a20ebc4f8b45ceaf095a12b72493d436cb112623cd0041a1af23d4",
};

const MEDIUM: ModelEntry = ModelEntry {
    size: ModelSize::Medium,
    file_name: "muscriptor-medium-f16.gguf",
    num_bytes: 618_442_496,
    sha256: "3850cc9e5b436b17a09bd25b8f2615cb3366ab96a71e7b50f73a793a917fdf03",
};

const LARGE: ModelEntry = ModelEntry {
    size: ModelSize::Large,
    file_name: "muscriptor-large-f16.gguf",
    num_bytes: 2_739_142_176,
    sha256: "35a750fb1ab1e77195cdc2c0b9b4aeea2f4d59f11f729f02af9920c4854ef72e",
};

pub fn entry(size: ModelSize) -> &'static ModelEntry {
    match size {
        ModelSize::Small => &SMALL,
        ModelSize::Medium => &MEDIUM,
        ModelSize::Large => &LARGE,
    }
}

/// The pinned download URL for one model.
pub fn resolve_url(entry: &ModelEntry) -> String {
    format!(
        "https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_DIRECTORY}/{}",
        entry.file_name
    )
}
