# InterpControl

Research console for mechanistic interpretability with GPT2-Medium. Train probes, visualize activations, and steer model behavior.

## Features

- 🎯 Train linear probes on any layer
- 📊 3D PCA visualization of activation spaces
- 🎚️ Steering vector control
- 🧠 Dual-process inference (System 1/System 2)
- 🔍 Real-time confidence monitoring

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/interpcontrol.git
cd interpcontrol
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Open your browser to `http://localhost:8000`

3. The interface will automatically train a probe on layer 14

4. Enter prompts and experiment with:
   - Different probe layers (12-16)
   - Steering coefficients (-5 to +5)
   - System 1 vs System 2 inference

## Project Structure
```
interpcontrol/
├── app.py              # FastAPI backend
├── templates/
│   └── index.html      # React frontend
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How It Works

1. **Probe Training**: Trains logistic regression classifiers on model activations to detect truthfulness
2. **Steering**: Applies learned direction vectors to influence model outputs
3. **Dual Processing**: Routes to System 2 (chain-of-thought) when confidence is low

## Requirements

- Python 3.8+
- 8GB+ RAM (for GPT2-Medium)
- CPU or CUDA-compatible GPU

## License

MIT
```

**.gitignore**
```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/
.venv
*.log
.DS_Store
.idea/
.vscode/
*.swp
*.swo
```

**Folder structure:**
```
interpcontrol/
├── app.py
├── templates/
│   └── index.html
├── requirements.txt
├── README.md
└── .gitignore
