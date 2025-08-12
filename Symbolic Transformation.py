import torch
import torch.nn as nn
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
import json

# ===== Config =====
FEATURE_PATH = Path(r"C:\Users\ADMIN\Desktop\sajin CT MRI\fused_features.pt")
MODEL_PATH = Path(r"C:\Users\ADMIN\Desktop\sajin CT MRI\cbm_model.pth")
OUTPUT_PROLOG_PATH = Path(r"C:\Users\ADMIN\Desktop\sajin CT MRI\tumor_concepts_facts.pl")
OUTPUT_JSON_PATH = Path(r"C:\Users\ADMIN\Desktop\sajin CT MRI\tumor_concepts_facts.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# ===== Model definition (must match training) =====
class CBMModel(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_concepts: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),     # Use LayerNorm as in training
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_concepts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ===== Concept mappings =====
# You can extend this dict for more concepts or multi-class concepts
CONCEPT_MAPS: Dict[str, Dict[int, str]] = {
    "shape": {0: "round", 1: "irregular"},
    "location": {0: "parietal_lobe", 1: "frontal_lobe"},
    "effect": {0: "compression", 1: "mass_effect"},
}

CONCEPT_ORDER = ["shape", "location", "effect"]  # must correspond to model output order

# ===== Load features =====
def load_features(feature_path: Path) -> torch.Tensor:
    data = torch.load(feature_path)
    features = data.get("features")
    if features is None:
        raise KeyError(f"'features' key not found in {feature_path}")
    return features.to(DEVICE)

# ===== Load model =====
def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = CBMModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ===== Predict concepts =====
def predict_concepts(model: nn.Module, features: torch.Tensor, threshold: float) -> torch.Tensor:
    with torch.no_grad():
        logits = model(features)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()
    return preds

# ===== Convert predictions to symbolic Prolog facts =====
def concepts_to_prolog(sample_id: int, pred_tensor: torch.Tensor) -> List[str]:
    facts = []
    for idx, concept_name in enumerate(CONCEPT_ORDER):
        concept_value_int = pred_tensor[idx].item()
        concept_value_str = CONCEPT_MAPS[concept_name].get(concept_value_int, "unknown")
        fact = f"has_{concept_name}(tumor_{sample_id}, {concept_value_str})."
        facts.append(fact)
    return facts

# ===== Main =====
def main():
    print(f"Loading features from: {FEATURE_PATH}")
    features = load_features(FEATURE_PATH)

    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, DEVICE)

    print(f"Predicting concepts for {features.size(0)} samples...")
    preds = predict_concepts(model, features, THRESHOLD)

    print("Converting predictions to Prolog facts...")
    all_facts = []
    json_facts = []
    for i, pred in enumerate(tqdm(preds, desc="Samples")):
        facts = concepts_to_prolog(i, pred)
        all_facts.extend(facts)
        # Also prepare JSON dict for easier machine processing later
        fact_dict = {f"has_{concept}": CONCEPT_MAPS[concept][pred[idx].item()]
                     for idx, concept in enumerate(CONCEPT_ORDER)}
        fact_dict["tumor_id"] = f"tumor_{i}"
        json_facts.append(fact_dict)

    # Save Prolog facts
    with open(OUTPUT_PROLOG_PATH, "w") as f:
        f.write("\n".join(all_facts))

    # Save JSON facts
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(json_facts, f, indent=2)

    print(f"✅ Prolog facts saved to: {OUTPUT_PROLOG_PATH}")
    print(f"✅ JSON facts saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
