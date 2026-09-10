// Re-export embedding functions from the classifiers crate.
// All embedding logic lives in classifiers::openai_embedding.
pub use classifiers::openai_embedding::{get_openai_embedding, get_openai_embeddings_batch};
