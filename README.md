# InterpControl-v1

Research console for mechanistic interpretability with GPT2-Medium. Train probes, visualize activations, and steer model behavior.

## Features

- Train linear probes on any layer
- 3D PCA visualization of activation spaces
- Steering vector control (-5x to +5x)
- Dual-process inference (System 1/System 2)
- Real-time confidence monitoring
- Cyberpunk-themed UI

## Installation

Clone the repository:
git clone https://github.com/saptarshihalder/InterpControl-v1.git
cd InterpControl-v1

Install dependencies:
pip install numpy==1.23.5
pip install torch==2.1.0 transformers==4.35.0
pip install transformer-lens scikit-learn fastapi uvicorn

Run the application:
python InterpControl.py

Open browser:
http://localhost:8000

## Usage

Example Prompts:
- The capital of France is Paris
- The capital of France is London
- Water is composed of hydrogen and oxygen
- The Sun orbits around the Earth
- Two plus two equals four
- Two plus two equals five

Controls:
- Layer Selection: Choose layers 12-16 to train probes on different transformer layers
- Steering Slider: Range -5 to +5, adjusts bias toward true/false responses
- Initialize Probe: Click to retrain probe on selected layer
- Input Box: Type prompt and press Enter or click Run

## How It Works

Probe Training:
1. Extracts activations from specified transformer layer
2. Trains logistic regression classifier on 30 true/false statements
3. Learns direction in activation space corresponding to truthfulness
4. Visualizes learned representations via PCA in 3D space

Steering Vectors:
- Uses probe classifier weights as steering direction
- Adds scaled vector to residual stream during generation
- Positive values bias toward "true", negative toward "false"
- Zero means no intervention, pure model output

Dual-Process Inference:
- System 1 (Confidence > 65%): Fast, direct generation
- System 2 (Confidence <= 65%): Chain-of-thought analysis then conclusion
- Automatically routes based on probe confidence score

Visualization:
- 3D PCA projection of activation space (green = true, red = false)
- Real-time confidence bar for each inference
- System routing indicator shows which process was used

## Training Data

The probe trains on 30 statements across three categories:

Geography (8 true, 8 false):
- The capital of France is Paris (TRUE)
- The capital of France is London (FALSE)
- The capital of Germany is Berlin (TRUE)
- The capital of Germany is Paris (FALSE)
- The capital of Italy is Rome (TRUE)
- The capital of Italy is Madrid (FALSE)
- The capital of Spain is Madrid (TRUE)
- The capital of Spain is Berlin (FALSE)
- The capital of England is London (TRUE)
- The capital of England is Dublin (FALSE)
- The capital of Japan is Tokyo (TRUE)
- The capital of Japan is Seoul (FALSE)
- The capital of Australia is Canberra (TRUE)
- The capital of Australia is Sydney (FALSE)
- The capital of Canada is Ottawa (TRUE)
- The capital of Canada is Toronto (FALSE)

Science (4 true, 4 false):
- Water is composed of hydrogen and oxygen (TRUE)
- Water is composed of hydrogen and nitrogen (FALSE)
- The Earth orbits around the Sun (TRUE)
- The Sun orbits around the Earth (FALSE)
- Photosynthesis produces oxygen (TRUE)
- Photosynthesis produces carbon dioxide (FALSE)
- Humans have 46 chromosomes (TRUE)
- Humans have 23 chromosomes (FALSE)

Mathematics (3 true, 3 false):
- Two plus two equals four (TRUE)
- Two plus two equals five (FALSE)
- Ten divided by two equals five (TRUE)
- Ten divided by two equals four (FALSE)
- A square has four sides (TRUE)
- A square has five sides (FALSE)

## Technical Details

Model: GPT2-Medium (355M parameters, 24 layers)
Framework: TransformerLens for interpretability hooks
Backend: FastAPI with async support
Frontend: React 18 with Babel compilation
Visualization: Plotly 3D scatter plots
Probe: Logistic Regression via scikit-learn
Dimensionality Reduction: PCA (3 components)

Default Settings:
- Model: gpt2-medium
- Device: CPU (GPU optional)
- Port: 8000
- Default Layer: 14
- Max Tokens: 25 (steered/system1), 40 (analysis), 10 (conclusion)
- Temperature: 0.6 (steered/system1), 0.7 (analysis), 0.5 (conclusion)

## Architecture

File Structure:
InterpControl-v1/
├── InterpControl.py    
├── index.html          
└── README.md           

Code Organization:
InterpControl.py contains:
- InterpAgent class: Model loading, probe training, inference
- FastAPI routes: /train, /infer endpoints
- HTML template: Complete React UI embedded as string
- Server configuration: Uvicorn with threading

Key Methods:
- get_training_data(): Returns 30 labeled examples
- train_probe(layer): Trains classifier on activations
- get_confidence(text, layer): Returns truthfulness score
- generate_steered(text, layer, coef): Applies steering vector
- generate_system2(text): Chain-of-thought reasoning
- get_pca_data(layer): 3D projection for visualization

## API Endpoints

POST /train
Request: {"layer": 14}
Response: {"accuracy": 0.95, "pca": [...], "confusion": [tn, fp, fn, tp]}

POST /infer
Request: {"text": "...", "layer": 14, "steering_coef": 0}
Response: {"confidence": 0.85, "output": "...", "system": "System 1", "trace": "..."}

GET /
Returns: HTML interface

## Requirements

Python 3.8 or higher
4GB RAM minimum (8GB recommended)
Internet connection for first run (downloads model)
Modern web browser (Chrome, Firefox, Safari, Edge)

Dependencies:
- numpy==1.23.5
- torch==2.1.0
- transformers==4.35.0
- transformer-lens
- scikit-learn
- fastapi
- uvicorn

## Troubleshooting

Port Already In Use:
Edit THE_SETTINGS_DICT in InterpControl.py:
THE_SETTINGS_DICT = {'port': 8001}

Model Download Fails:
- Check internet connection
- Ensure 2GB free disk space
- Model downloads to ~/.cache/huggingface/

Out of Memory:
- Close other applications
- Reduce max_new_tokens in generate calls
- Use smaller model (gpt2-small)

Probe Training Fails:
- Check if model loaded successfully
- Verify layer number is valid (0-23 for gpt2-medium)
- Restart application

UI Not Loading:
- Check console for errors
- Verify port 8000 is accessible
- Try different browser
- Check backend logs

## Research Applications

Factual Knowledge Probing:
- Test where factual knowledge is stored in layers
- Compare early vs late layer representations
- Analyze confidence calibration

Activation Steering:
- Study effect of intervention strength
- Test generalization to new prompts
- Measure output distribution shifts

Interpretability Analysis:
- Visualize semantic clustering
- Identify truthfulness directions
- Compare layer-wise separability

Model Behavior:
- System 1 vs System 2 routing
- Confidence thresholds
- Generation quality under steering

## Advanced Usage

Custom Training Data:
Edit get_training_data() in InterpControl.py to add your own examples:

def get_training_data(self):
    return [
        ("Your true statement", 1),
        ("Your false statement", 0),
        # Add more...
    ]

Different Model:
Change model_name in THE_SETTINGS_DICT:
THE_SETTINGS_DICT = {'model_name': 'gpt2-small'}

Available models: gpt2-small, gpt2-medium, gpt2-large, gpt2-xl

GPU Support:
THE_SETTINGS_DICT = {'device': 'cuda'}

Adjust Generation Parameters:
In generate methods, modify:
- max_new_tokens: Length of generation
- temperature: Randomness (0.1-1.0)
- top_k: Sampling diversity

## Performance Notes

Model Loading: 5-15 seconds (first run downloads model)
Probe Training: 10-30 seconds per layer
Inference: 1-3 seconds per prompt
Steering: Same as inference (minimal overhead)
System 2: 2-5 seconds (runs multiple generations)

Optimization Tips:
- Train probes on needed layers only
- Reuse trained probes across sessions
- Batch similar prompts
- Use GPU for faster generation

## Known Limitations

- Probe accuracy limited by training data quality
- Steering may produce incoherent text at extreme values
- System 2 reasoning not always reliable
- PCA visualization loses information (high-dim to 3D)
- No probe persistence (retrains on restart)

## Future Enhancements

- Save/load trained probes
- Custom training data upload
- Multiple probe comparison
- Layer-wise activation visualization
- Batch inference mode
- Export results to CSV
- Model comparison tools
- Advanced steering methods

## Citation

If you use this code in your research:

@software{interpcontrol2024,
  author = {Saptarshi Halder},
  title = {InterpControl: Interactive Mechanistic Interpretability Console},
  year = {2024},
  url = {https://github.com/saptarshihalder/InterpControl-v1}
}

## References

TransformerLens: https://github.com/neelnanda-io/TransformerLens
Activation Steering: https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/
Mechanistic Interpretability: https://transformer-circuits.pub/

## License

MIT License

Copyright (c) 2024 Saptarshi Halder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

Guidelines:
1. Fork the repository
2. Create feature branch (git checkout -b feature/amazing)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing)
5. Open Pull Request

## Contact

GitHub: @saptarshihalder
Issues: https://github.com/saptarshihalder/InterpControl-v1/issues
Email: Create an issue for questions

## Acknowledgments

Built with TransformerLens by Neel Nanda
Thanks to the open-source ML community

---

Built for interpretability researchers and AI safety enthusiasts

<img width="1727" height="647" alt="image" src="https://github.com/user-attachments/assets/8c2c5bdb-5cc2-441c-9a96-4e2b0c1a22b4" />
<img width="1736" height="455" alt="image" src="https://github.com/user-attachments/assets/8202c920-96e0-4455-83e8-2aae5866a6d2" />
<img width="1715" height="703" alt="image" src="https://github.com/user-attachments/assets/31b40143-d8a7-46e2-bc9b-7be11f97df15" />

<img width="1449" height="590" alt="image" src="https://github.com/user-attachments/assets/b9ed051f-cd8f-437b-bb64-46124d8f17d1" />
on TruthfulQA Evaluation


