# forge-conservation

Conservation ratio tracking for tile transforms. Implements the conservation ratio (CR) mathematics from the **SuperInstance spectral conservation theorems** (T1–T5), applied to tile pipelines.

## Theorems (T1–T5)

The conservation theorems define how information flows through transform pipelines:

- **T1 — Entropy Conservation**: The total Shannon entropy of a transform pipeline's output cannot exceed the entropy of its input. Information is preserved or lost, never created.
- **T2 — Conservation Ratio Invariance**: The conservation ratio `CR = min(H(out)/H(in), 1.0)` is multiplicative across pipeline stages. A pipeline's overall CR is the product of its stage CRs.
- **T3 — Loss Monotonicity**: Information loss through a pipeline is monotonically non-decreasing with each additional stage. `loss_total ≥ loss_any_single_stage`.
- **T4 — Divergence Bound**: The KL divergence between input and output byte distributions provides an upper bound on information loss per stage.
- **T5 — Bottleneck Theorem**: For a three-stage pipeline (input → intermediate → output), the bottleneck score constrains the recoverable mutual information: `I(X;Y) ≤ min(I(X;Z), I(Z;Y))` where Z is the intermediate representation.

## Conservation Ratio (CR)

The core metric:

```
CR = min(H(output) / H(input), 1.0)
```

Where H is Shannon entropy in bits. Perfect conservation = 1.0, total loss = 0.0.

## Usage

```rust
use forge_conservation::{ConservationTracker, conservation_ratio, shannon_entropy};

// Track a multi-stage pipeline
let mut tracker = ConservationTracker::new();

let input_data: Vec<u8> = /* ... */;
let compressed: Vec<u8> = /* ... */;
let reconstructed: Vec<u8> = /* ... */;

tracker.record_stage("compress", &input_data, &compressed);
tracker.record_stage("decompress", &compressed, &reconstructed);

let report = tracker.report();
println!("Overall CR: {:.4}", report.overall_cr);
println!("Worst stage: {} (index {})",
    report.stage_reports[report.worst_stage].stage_name,
    report.worst_stage
);

// Or use standalone functions
let cr = conservation_ratio(&input_data, &output_data);
let h = shannon_entropy(&data);
```

## Functions

| Function | Description |
|---|---|
| `shannon_entropy(data)` | Shannon entropy of byte distribution (bits) |
| `conservation_ratio(input, output)` | CR between two byte sequences |
| `verify_conservation(input, output, threshold)` | Check CR ≥ threshold |
| `kl_divergence(input, output)` | KL divergence between byte distributions |
| `mutual_information(input, output)` | Mutual information between sequences |
| `bottleneck_score(input, intermediate, output)` | Information bottleneck metric |

## Installation

```toml
[dependencies]
forge-conservation = "0.1"
```

## License

MIT
