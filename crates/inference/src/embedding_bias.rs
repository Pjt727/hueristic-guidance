use inference_types::BiasRegime;

pub const DEFAULT_FORCE_THRESHOLD: f32 = 0.90;
pub const DEFAULT_FORCE_GAP: f32 = 0.05;
pub const DEFAULT_SOFT_DECAY: f32 = 30.0;

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingBiasConfig {
    pub force_threshold: f32,
    pub force_gap: f32,
    pub soft_decay: f32,
}

impl Default for EmbeddingBiasConfig {
    fn default() -> Self {
        Self {
            force_threshold: DEFAULT_FORCE_THRESHOLD,
            force_gap: DEFAULT_FORCE_GAP,
            soft_decay: DEFAULT_SOFT_DECAY,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingSignal {
    pub category_name: String,
    pub positive_similarity: f32,
    pub negative_similarity: f32,
    pub margin: f32,
    pub kappa: f32,
}

#[derive(Debug, Clone)]
pub struct ResolvedCategoryBias {
    pub category_name: String,
    pub weighted_margin: f32,
    pub sim_score: f32,
    pub positive_similarity: f32,
    pub negative_similarity: f32,
    pub regime: BiasRegime,
    pub force_target: bool,
}

pub fn soft_gain(positive_similarity: f32, config: EmbeddingBiasConfig) -> f32 {
    let capped = positive_similarity.min(config.force_threshold);
    (config.soft_decay * (capped - config.force_threshold)).exp()
}

pub fn resolve_category_biases(
    signals: &[EmbeddingSignal],
    config: EmbeddingBiasConfig,
) -> Vec<ResolvedCategoryBias> {
    if signals.is_empty() {
        return vec![];
    }

    let mut ranked: Vec<&EmbeddingSignal> = signals.iter().collect();
    ranked.sort_by(|a, b| b.positive_similarity.total_cmp(&a.positive_similarity));
    let top = ranked[0];
    let runner_up = ranked
        .get(1)
        .map(|signal| signal.positive_similarity)
        .unwrap_or(f32::NEG_INFINITY);
    let force_target = top.positive_similarity >= config.force_threshold
        && top.positive_similarity - runner_up >= config.force_gap;

    signals
        .iter()
        .map(|signal| {
            let is_force_target = force_target && signal.category_name == top.category_name;
            let regime = if is_force_target {
                BiasRegime::Force
            } else {
                BiasRegime::Soft
            };
            let weighted_margin = signal.kappa
                * signal.margin.max(0.0)
                * soft_gain(signal.positive_similarity, config);

            ResolvedCategoryBias {
                category_name: signal.category_name.clone(),
                weighted_margin,
                sim_score: signal.margin,
                positive_similarity: signal.positive_similarity,
                negative_similarity: signal.negative_similarity,
                regime,
                force_target: is_force_target,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(name: &str, positive: f32, margin: f32) -> EmbeddingSignal {
        EmbeddingSignal {
            category_name: name.to_string(),
            positive_similarity: positive,
            negative_similarity: positive - margin,
            margin,
            kappa: 10.0,
        }
    }

    #[test]
    fn forces_unique_near_match() {
        let resolved = resolve_category_biases(
            &[signal("dosing", 0.96, 0.3), signal("samples", 0.82, 0.1)],
            EmbeddingBiasConfig::default(),
        );
        assert!(resolved[0].force_target);
        assert_eq!(resolved[0].regime, BiasRegime::Force);
        assert!(!resolved[1].force_target);
    }

    #[test]
    fn collision_falls_back_to_soft() {
        let resolved = resolve_category_biases(
            &[signal("dosing", 0.93, 0.3), signal("samples", 0.91, 0.2)],
            EmbeddingBiasConfig::default(),
        );
        assert!(resolved.iter().all(|bias| bias.regime == BiasRegime::Soft));
        assert!(resolved.iter().all(|bias| !bias.force_target));
    }

    #[test]
    fn soft_bias_decays_sharply_below_threshold() {
        let config = EmbeddingBiasConfig::default();
        let low = soft_gain(0.70, config);
        let middle = soft_gain(0.85, config);
        let near = soft_gain(0.89, config);
        assert!(low < middle);
        assert!(middle < near);
        assert!(low < 0.01);
        assert!(near < 1.0);
    }

    #[test]
    fn negative_margin_is_not_promoted() {
        let resolved = resolve_category_biases(
            &[signal("dosing", 0.85, -0.1), signal("samples", 0.70, 0.1)],
            EmbeddingBiasConfig::default(),
        );
        assert_eq!(resolved[0].weighted_margin, 0.0);
    }
}
