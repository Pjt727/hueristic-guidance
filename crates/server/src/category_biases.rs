use inference::{CategoryBias, EmbeddingBiasConfig, EmbeddingSignal, resolve_category_biases};
use sqlx::SqlitePool;

use crate::db::MessageMargin;

pub fn config_from_env() -> EmbeddingBiasConfig {
    let defaults = EmbeddingBiasConfig::default();
    EmbeddingBiasConfig {
        force_threshold: env_f32("EMBEDDING_FORCE_THRESHOLD", defaults.force_threshold)
            .clamp(0.0, 1.0),
        force_gap: env_f32("EMBEDDING_FORCE_GAP", defaults.force_gap).max(0.0),
        soft_decay: env_f32("EMBEDDING_SOFT_DECAY", defaults.soft_decay).max(f32::EPSILON),
    }
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &f32| value.is_finite())
        .unwrap_or(default)
}

pub async fn assemble(sqlite_db: &SqlitePool, margins: &[MessageMargin]) -> Vec<CategoryBias> {
    let mut signals = Vec::with_capacity(margins.len());
    for margin in margins {
        let kappa = crate::db::get_or_create_kappa(sqlite_db, margin.message_id)
            .await
            .unwrap_or(10.0);
        signals.push(EmbeddingSignal {
            category_name: margin.category_name.clone(),
            positive_similarity: margin.positive_similarity as f32,
            negative_similarity: margin.negative_similarity as f32,
            margin: margin.margin as f32,
            kappa: kappa as f32,
        });
    }

    resolve_category_biases(&signals, config_from_env())
        .into_iter()
        .map(CategoryBias::from)
        .collect()
}
