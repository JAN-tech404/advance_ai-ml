# Multimodal Housing Price Prediction

Deep learning model that combines image and tabular data for house price prediction.

## Approach

- **Image Branch**: ResNet18 (pretrained) extracts features from synthetic house images
- **Tabular Branch**: MLP processes structured data (square feet, bedrooms, bathrooms, etc.)
- **Fusion**: Concatenated features pass through FC layers for price regression

## Data

- Synthetic dataset with 1,000 samples
- Features: square_feet, bedrooms, bathrooms, age, garage_size, lot_size, condition, neighborhood
- Generated house images based on property attributes

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Full (Multimodal) | $456,272 | $460,736 | 92.52% |
| Image Only | $492,752 | $497,142 | 100.00% |
| Tabular Only | $492,752 | $497,142 | 100.00% |

## Quick Start

```python
import torch
from housing_multimodal_ml import MultimodalHousingModel

# Load model
model = MultimodalHousingModel(tabular_input_size=11)
model.load_state_dict(torch.load('output/best_model.pth'))

# Predict
model.eval()
with torch.no_grad():
    price = model(image_tensor, tabular_tensor)
```

## Training

Run `housing_multimodal_ml.ipynb` to train the model yourself.

## Requirements

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn pillow
```

## Files

- `housing_multimodal_ml.ipynb` - Training notebook
- `code.txt` - Exported Python code
- `output/` - Model weights, plots, and generated images
