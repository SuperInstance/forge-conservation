//! # forge-conservation
//!
//! Conservation ratio tracking for tile transforms.
//!
//! Implements the conservation ratio (CR) mathematics from the SuperInstance
//! spectral conservation theorems (T1–T5), applied to tile pipelines.
//! Measures how well transforms preserve information.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Conservation ratio report for a complete transform pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationReport {
    pub pipeline_id: Uuid,
    pub stage_reports: Vec<StageReport>,
    pub overall_cr: f64,
    pub worst_stage: usize,
    pub timestamp_ms: u64,
}

/// Report for a single transform stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageReport {
    pub stage_name: String,
    pub tiles_in: usize,
    pub tiles_out: usize,
    pub cr: f64,
    pub information_loss: f64,
    pub entropy_in: f64,
    pub entropy_out: f64,
}

/// Tracks conservation ratios across multiple pipeline stages.
pub struct ConservationTracker {
    pipeline_id: Uuid,
    stages: Vec<StageReport>,
}

impl Default for ConservationTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ConservationTracker {
    /// Create a new tracker for a pipeline.
    pub fn new() -> Self {
        Self {
            pipeline_id: Uuid::new_v4(),
            stages: Vec::new(),
        }
    }

    /// Record a pipeline stage, computing CR from input/output byte data.
    /// Returns the conservation ratio for this stage.
    pub fn record_stage(&mut self, name: &str, input_bytes: &[u8], output_bytes: &[u8]) -> f64 {
        let entropy_in = shannon_entropy(input_bytes);
        let entropy_out = shannon_entropy(output_bytes);
        let cr = conservation_ratio(input_bytes, output_bytes);
        let information_loss = 1.0 - cr;

        self.stages.push(StageReport {
            stage_name: name.to_string(),
            tiles_in: input_bytes.len(),
            tiles_out: output_bytes.len(),
            cr,
            information_loss,
            entropy_in,
            entropy_out,
        });

        cr
    }

    /// Generate a full conservation report for all recorded stages.
    pub fn report(&self) -> ConservationReport {
        let overall_cr = if self.stages.is_empty() {
            1.0
        } else {
            self.stages.iter().map(|s| s.cr).product::<f64>()
        };

        let worst_stage = self
            .stages
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.cr.partial_cmp(&b.cr).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        ConservationReport {
            pipeline_id: self.pipeline_id,
            stage_reports: self.stages.clone(),
            overall_cr,
            worst_stage,
            timestamp_ms,
        }
    }
}

/// Compute Shannon entropy of a byte distribution (bits).
pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let len = data.len() as f64;
    let mut freq: HashMap<u8, usize> = HashMap::new();
    for &b in data {
        *freq.entry(b).or_insert(0) += 1;
    }

    freq.values()
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

/// Compute the conservation ratio between input and output byte sequences.
///
/// CR = min(H(output) / H(input), 1.0)
/// Perfect conservation = 1.0, total loss = 0.0.
pub fn conservation_ratio(input: &[u8], output: &[u8]) -> f64 {
    if input.is_empty() {
        return if output.is_empty() { 1.0 } else { 1.0 };
    }

    let h_in = shannon_entropy(input);
    if h_in == 0.0 {
        // Input has zero entropy (constant bytes) — any output conserves perfectly
        return 1.0;
    }

    let h_out = shannon_entropy(output);
    (h_out / h_in).min(1.0)
}

/// Verify that the conservation ratio meets a threshold.
pub fn verify_conservation(input: &[u8], output: &[u8], threshold: f64) -> bool {
    conservation_ratio(input, output) >= threshold
}

/// Compute KL divergence D_KL(P_input || P_output) between byte distributions.
pub fn kl_divergence(input: &[u8], output: &[u8]) -> f64 {
    let p = byte_distribution(input);
    let q = byte_distribution(output);

    if p.is_empty() && q.is_empty() {
        return 0.0;
    }
    if p.is_empty() {
        return f64::INFINITY;
    }

    // Use a small epsilon to avoid log(0)
    let eps = 1e-10;
    let all_keys: std::collections::HashSet<u8> = p.keys().chain(q.keys()).copied().collect();

    all_keys
        .iter()
        .map(|&k| {
            let pi = *p.get(&k).unwrap_or(&eps);
            let qi = *q.get(&k).unwrap_or(&eps);
            if pi > eps {
                pi * (pi / qi).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Compute mutual information between input and output byte sequences.
///
/// I(X;Y) = H(X) + H(Y) - H(X,Y) where H(X,Y) is joint entropy.
/// For byte sequences we use a windowed pairing approach.
pub fn mutual_information(input: &[u8], output: &[u8]) -> f64 {
    if input.is_empty() || output.is_empty() {
        return 0.0;
    }

    let min_len = input.len().min(output.len());
    let h_x = shannon_entropy(input);
    let h_y = shannon_entropy(output);

    // Joint entropy from paired bytes
    let pairs: Vec<(u8, u8)> = (0..min_len).map(|i| (input[i], output[i])).collect();
    let h_xy = joint_entropy(&pairs);

    (h_x + h_y - h_xy).max(0.0)
}

/// Compute the information bottleneck score for a three-stage pipeline.
///
/// Measures how much information the intermediate representation preserves
/// relative to the input-output mutual information.
pub fn bottleneck_score(input: &[u8], intermediate: &[u8], output: &[u8]) -> f64 {
    if input.is_empty() || intermediate.is_empty() || output.is_empty() {
        return 0.0;
    }

    let mi_input_output = mutual_information(input, output);
    let mi_input_intermediate = mutual_information(input, intermediate);
    let mi_intermediate_output = mutual_information(intermediate, output);

    // Bottleneck score: geometric mean of the two MI ratios
    if mi_input_output == 0.0 {
        return 0.0;
    }

    let ratio_in = mi_input_intermediate / mi_input_output;
    let ratio_out = mi_intermediate_output / mi_input_output;

    (ratio_in * ratio_out).sqrt().min(1.0)
}

// --- Internal helpers ---

fn byte_distribution(data: &[u8]) -> HashMap<u8, f64> {
    if data.is_empty() {
        return HashMap::new();
    }
    let len = data.len() as f64;
    let mut freq: HashMap<u8, usize> = HashMap::new();
    for &b in data {
        *freq.entry(b).or_insert(0) += 1;
    }
    freq.into_iter().map(|(k, v)| (k, v as f64 / len)).collect()
}

fn joint_entropy(pairs: &[(u8, u8)]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let len = pairs.len() as f64;
    let mut freq: HashMap<(u8, u8), usize> = HashMap::new();
    for &p in pairs {
        *freq.entry(p).or_insert(0) += 1;
    }
    freq.values()
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Shannon entropy ---

    #[test]
    fn test_entropy_empty() {
        assert_eq!(shannon_entropy(&[]), 0.0);
    }

    #[test]
    fn test_entropy_single_byte() {
        // Single repeated byte has zero entropy
        assert_eq!(shannon_entropy(&[42u8; 100]), 0.0);
    }

    #[test]
    fn test_entropy_uniform_bytes() {
        // All 256 byte values equally likely → entropy = 8 bits
        let data: Vec<u8> = (0..=255).collect();
        let h = shannon_entropy(&data);
        assert!((h - 8.0).abs() < 0.01, "Expected ~8.0, got {}", h);
    }

    #[test]
    fn test_entropy_two_symbols() {
        // 50/50 mix → 1 bit
        let data = vec![0u8, 1, 0, 1, 0, 1, 0, 1];
        let h = shannon_entropy(&data);
        assert!((h - 1.0).abs() < 0.01, "Expected ~1.0, got {}", h);
    }

    // --- Conservation ratio ---

    #[test]
    fn test_cr_identical_data() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let cr = conservation_ratio(&data, &data);
        assert!((cr - 1.0).abs() < 1e-9, "Expected 1.0, got {}", cr);
    }

    #[test]
    fn test_cr_empty_output() {
        let input = vec![1, 2, 3, 4, 5];
        let cr = conservation_ratio(&input, &[]);
        assert!((cr - 0.0).abs() < 1e-9, "Expected 0.0, got {}", cr);
    }

    #[test]
    fn test_cr_empty_input_empty_output() {
        let cr = conservation_ratio(&[], &[]);
        assert!((cr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cr_compressed_data() {
        // Highly compressible input, output has different entropy
        let input: Vec<u8> = (0..100).flat_map(|_| [42u8].repeat(10)).collect();
        let output: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let cr = conservation_ratio(&input, &output);
        // Input entropy ~0 (constant), so CR should be 1.0 (special case)
        assert!((cr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cr_reduced_entropy() {
        // Input has high entropy, output has lower entropy
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let output = vec![0u8; 1024]; // zero entropy
        let cr = conservation_ratio(&input, &output);
        assert!(cr < 0.01, "Expected near 0.0, got {}", cr);
    }

    // --- verify_conservation ---

    #[test]
    fn test_verify_conservation_pass() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(verify_conservation(&data, &data, 0.99));
    }

    #[test]
    fn test_verify_conservation_fail() {
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let output = vec![0u8; 1024];
        assert!(!verify_conservation(&input, &output, 0.5));
    }

    // --- ConservationTracker ---

    #[test]
    fn test_tracker_single_stage() {
        let mut tracker = ConservationTracker::new();
        let data = vec![10, 20, 30, 40, 50];
        let cr = tracker.record_stage("compress", &data, &data);
        assert!((cr - 1.0).abs() < 1e-9);

        let report = tracker.report();
        assert_eq!(report.stage_reports.len(), 1);
        assert!((report.overall_cr - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tracker_multiple_stages() {
        let mut tracker = ConservationTracker::new();
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();

        // Stage 1: identity
        let cr1 = tracker.record_stage("identity", &input, &input);
        assert!((cr1 - 1.0).abs() < 1e-9);

        // Stage 2: reduced entropy
        let reduced: Vec<u8> = input.iter().map(|&b| b % 16).collect();
        let cr2 = tracker.record_stage("quantize", &input, &reduced);
        assert!(cr2 < 1.0);

        let report = tracker.report();
        assert_eq!(report.stage_reports.len(), 2);
        assert!(report.overall_cr < 1.0);
        assert_eq!(report.worst_stage, 1); // quantize stage
    }

    #[test]
    fn test_tracker_empty() {
        let tracker = ConservationTracker::new();
        let report = tracker.report();
        assert_eq!(report.stage_reports.len(), 0);
        assert!((report.overall_cr - 1.0).abs() < 1e-9);
    }

    // --- KL divergence ---

    #[test]
    fn test_kl_identical() {
        let data = vec![1, 2, 3, 4];
        let kl = kl_divergence(&data, &data);
        assert!(kl < 1e-6, "Expected ~0.0, got {}", kl);
    }

    #[test]
    fn test_kl_different() {
        let p = vec![0u8; 100];
        let q = vec![1u8; 100];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    // --- Mutual information ---

    #[test]
    fn test_mi_identical() {
        let data: Vec<u8> = (0..64).cycle().take(256).collect();
        let mi = mutual_information(&data, &data);
        assert!(mi > 0.0, "MI of identical sequences should be > 0");
    }

    #[test]
    fn test_mi_independent() {
        // Two sequences with no clear relationship
        let input: Vec<u8> = (0..64).cycle().take(256).collect();
        let output: Vec<u8> = (200..=255).cycle().take(256).collect();
        // Both have high entropy; MI will be non-zero due to structure but should be moderate
        let mi = mutual_information(&input, &output);
        assert!(mi >= 0.0);
    }

    // --- Bottleneck score ---

    #[test]
    fn test_bottleneck_perfect() {
        let data: Vec<u8> = (0..64).cycle().take(256).collect();
        let score = bottleneck_score(&data, &data, &data);
        assert!((score - 1.0).abs() < 0.01, "Expected ~1.0, got {}", score);
    }

    #[test]
    fn test_bottleneck_lossy() {
        let input: Vec<u8> = (0..64).cycle().take(256).collect();
        let intermediate = vec![0u8; 256];
        let output: Vec<u8> = (0..64).cycle().take(256).collect();
        let score = bottleneck_score(&input, &intermediate, &output);
        // Intermediate is flat, so it carries no information
        assert!(score < 0.5, "Expected low score, got {}", score);
    }

    #[test]
    fn test_bottleneck_empty() {
        let score = bottleneck_score(&[], &[1, 2, 3], &[4, 5, 6]);
        assert_eq!(score, 0.0);
    }
}
