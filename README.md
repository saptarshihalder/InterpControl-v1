# 🔬 InterpControl

> **Mechanistic Interpretability Dashboard for Transformer Models**

A powerful tool for probing, steering, and understanding the internal representations of transformer language models. Built with FastAPI and TransformerLens.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Features

- **🔍 Linear Probing** - Train classifiers on model activations to understand internal representations
- **🎯 Activation Steering** - Directly manipulate model behavior by adding vectors to activation space
- **🧠 Dual-System Inference** - Automatic switching between fast (System 1) and slow (System 2) reasoning
- **📊 Real-time Visualization** - PCA projections of activation spaces
- **🌐 Web Interface** - Clean, modern dashboard for all operations
- **⚡ Easy to Use** - Works with Google Colab or local installations

## 🚀 Quick Start

### Option 1: Google Colab (Easiest)

```python
!git clone https://github.com/yourusername/interpcontrol.git
%cd interpcontrol
!python app.py
```

The UI will automatically embed in your notebook!

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/interpcontrol.git
cd interpcontrol

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open http://localhost:8000 in your browser.

### Option 3: Docker (Coming Soon)

```bash
docker run -p 8000:8000 yourusername/interpcontrol
```

## 📖 How It Works

### 1. **Truth Probing**

Train a linear classifier on model activations to detect factual correctness:

```python
from app import InterpController

controller = InterpController()
accuracy, confusion = controller.train_probe(layer=6)
print(f"Probe accuracy: {accuracy*100:.1f}%")
```

### 2. **Activation Steering**

Manipulate model outputs by adding direction vectors:

```python
output = controller.generate_steered(
    text="The capital of France is",
    layer=6,
    steering_strength=2.0  # Positive = more truthful
)
```

### 3. **Confidence-Based Routing**

Models automatically choose reasoning strategy:

- **High confidence (>65%)** → System 1: Direct generation
- **Low confidence** → System 2: Chain-of-thought reasoning

## 🎮 Usage Examples

### Basic Probing

```python
# Train probe on layer 6
controller.train_probe(layer=6)

# Check confidence
confidence = controller.get_confidence(
    "The Earth is flat", 
    layer=6
)
print(f"Confidence: {confidence:.2%}")  # Low confidence for false statement
```

### Steering Generation

```python
# Steer toward truthfulness
truthful_output = controller.generate_steered(
    "The capital of Spain is",
    layer=6,
    steering_strength=3.0
)

# Steer away from truthfulness
false_output = controller.generate_steered(
    "The capital of Spain is",
    layer=6,
    steering_strength=-3.0
)
```

### Visualization

```python
# Get 3D PCA projection of activations
pca_data = controller.get_pca_visualization(layer=6)
# Returns: [{"x": ..., "y": ..., "z": ..., "label": ...}, ...]
```

## 🏗️ Architecture

```
InterpControl
│
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── README.md              # This file
├── LICENSE                # MIT License
└── .gitignore            # Git ignore rules
```

### Tech Stack

- **TransformerLens** - Access to model internals
- **FastAPI** - Modern web framework
- **PyTorch** - Deep learning backend
- **scikit-learn** - Probe training
- **Tailwind CSS** - UI styling

## 🔧 Configuration

Environment variables (optional):

```bash
export MODEL_NAME=gpt2-small    # Model to use
export PORT=8000                # Server port
export HOST=0.0.0.0            # Server host
```

Or modify `CONFIG` dict in `app.py`:

```python
CONFIG = {
    'model_name': 'gpt2-small',
    'device': 'cuda',  # or 'cpu'
    'port': 8000
}
```

## 📊 Supported Models

Currently tested with:
- ✅ GPT-2 Small (124M params)
- ✅ GPT-2 Medium (355M params)
- ✅ GPT-2 Large (774M params)
- ⚠️  GPT-2 XL (1.5B params) - Requires >16GB RAM

Any model supported by TransformerLens should work!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/interpcontrol.git
cd interpcontrol

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
python app.py
```

## 📚 Research Background

This tool implements concepts from:

- **Linear Probing**: [Alain & Bengio, 2016](https://arxiv.org/abs/1610.01644)
- **Activation Steering**: [Turner et al., 2023](https://arxiv.org/abs/2308.10248)
- **Mechanistic Interpretability**: [Olah et al., 2020](https://distill.pub/2020/circuits/)

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'transformer_lens'**
```bash
pip install transformer-lens
```

**CUDA out of memory**
```python
CONFIG['device'] = 'cpu'  # Use CPU instead
```

**Port already in use**
```bash
export PORT=8001  # Use different port
```

**Python version incompatibility**
- Requires Python 3.10 or higher
- On older Python: Use legacy dependencies in `requirements.txt`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) by Neel Nanda
- The mechanistic interpretability community

---

<img width="1732" height="727" alt="image" src="https://github.com/user-attachments/assets/b60c8893-c39a-4745-8bb9-21075ab8c616" />


<img width="1730" height="482" alt="image" src="https://github.com/user-attachments/assets/ff5ba7d5-79cd-4109-b177-95adceb2f569" />

<img width="1743" height="722" alt="image" src="https://github.com/user-attachments/assets/68d67932-effe-4984-b283-7291610d6002" />

If you find this useful, please consider starring the repository!
