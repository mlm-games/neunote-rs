use std::fs;
use std::path::Path;

/// Loaded CNN weights ready to pass to BasicPitchCNN::new()
pub struct CnnWeights {
    pub contour_w1: Vec<f32>,
    pub contour_b1: Vec<f32>,
    pub contour_w2: Vec<f32>,
    pub contour_b2: Vec<f32>,
    pub note_w1: Vec<f32>,
    pub note_b1: Vec<f32>,
    pub note_w2: Vec<f32>,
    pub note_b2: Vec<f32>,
    pub onset1_w: Vec<f32>,
    pub onset1_b: Vec<f32>,
    pub onset2_w: Vec<f32>,
    pub onset2_b: Vec<f32>,
}

/// Parse a single RTNeural-style JSON layer and extract weights + bias
fn parse_layer(weights: &serde_json::Value) -> (Vec<f32>, Vec<f32>) {
    // JSON format: weights[0] = weight tensor, weights[1] = bias vector
    let weight_tensor = &weights[0];
    let bias_tensor = &weights[1];

    // Flatten weight tensor: [t][kf][fi][fo] order
    let mut flat_weights = Vec::new();
    flatten_json(weight_tensor, &mut flat_weights);

    let flat_bias: Vec<f32> = bias_tensor
        .as_array()
        .map(|arr| arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect())
        .unwrap_or_default();

    (flat_weights, flat_bias)
}

fn flatten_json(val: &serde_json::Value, out: &mut Vec<f32>) {
    match val {
        serde_json::Value::Array(arr) => {
            for elem in arr {
                flatten_json(elem, out);
            }
        }
        serde_json::Value::Number(n) => {
            out.push(n.as_f64().unwrap_or(0.0) as f32);
        }
        _ => {}
    }
}

/// Load all 4 CNN model JSON files from a directory
pub fn load_cnn_weights(model_dir: &Path) -> Result<CnnWeights, String> {
    let load = |name: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
        let path = model_dir.join(format!("cnn_{}_model.json", name));
        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

        let layers = json["layers"]
            .as_array()
            .ok_or_else(|| format!("No layers array in {}", path.display()))?;

        // For multi-layer models, extract each layer's weights and concatenate
        let mut all_w = Vec::new();
        let mut all_b = Vec::new();
        for layer in layers {
            let w = &layer["weights"];
            let (layer_w, layer_b) = parse_layer(w);
            all_w.extend(layer_w);
            all_b.extend(layer_b);
        }

        Ok((all_w, all_b))
    };

    let (contour_w1, contour_b1) = load("contour")?;
    let (note_w1, note_b1) = load("note")?;
    let (onset1_w, onset1_b) = load("onset_1")?;
    let (onset2_w, onset2_b) = load("onset_2")?;

    // Split multi-layer models: contour has 2 layers, note has 2 layers
    // Layer 1 contour: 8 → 8, kernel (3, 39) = 3*39*8*8 = 7488 weights + 8 bias
    // Layer 2 contour: 8 → 1, kernel (5, 5) = 5*5*8*1 = 200 weights + 1 bias
    let c1_size = 3 * 39 * 8 * 8; // 7488
    let c2_size = 5 * 5 * 8; // 200

    if contour_w1.len() < c1_size + c2_size {
        return Err(format!(
            "Contour model: expected {} + {} = {} weights, got {}",
            c1_size,
            c2_size,
            c1_size + c2_size,
            contour_w1.len()
        ));
    }

    // Note model: layer 1: (1,32,264,7,7) = 7*7*1*32 = 1568 + 32 bias
    //             layer 2: (32,1,88,7,3) = 7*3*32*1 = 672 + 1 bias
    let n1_size = 7 * 7 * 32; // 1568
    let n2_size = 7 * 3 * 32; // 672

    if note_w1.len() < n1_size + n2_size {
        return Err(format!(
            "Note model: expected {} + {} = {} weights, got {}",
            n1_size,
            n2_size,
            n1_size + n2_size,
            note_w1.len()
        ));
    }

    // Onset 1: (8,32,264,5,5) = 5*5*8*32 = 6400 + 32 bias
    let o1_size = 5 * 5 * 8 * 32; // 6400

    // Onset 2: (33,1,88,3,3) = 3*3*33*1 = 297 + 1 bias
    let o2_size = 3 * 3 * 33; // 297

    Ok(CnnWeights {
        contour_w1: contour_w1[..c1_size].to_vec(),
        contour_b1: contour_b1[..8].to_vec(),
        contour_w2: contour_w1[c1_size..c1_size + c2_size].to_vec(),
        contour_b2: contour_b1[8..].to_vec(),
        note_w1: note_w1[..n1_size].to_vec(),
        note_b1: note_b1[..32].to_vec(),
        note_w2: note_w1[n1_size..n1_size + n2_size].to_vec(),
        note_b2: note_b1[32..].to_vec(),
        onset1_w: onset1_w[..o1_size].to_vec(),
        onset1_b: onset1_b.clone(),
        onset2_w: onset2_w[..o2_size].to_vec(),
        onset2_b: onset2_b.clone(),
    })
}
