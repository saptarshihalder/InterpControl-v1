#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

print("InterpControl Setup")
print("=" * 60)

def check_dependencies():
    list_of_crap_we_need = [
        'torch', 'transformers', 'transformer_lens', 
        'sklearn', 'fastapi', 'uvicorn', 'nest_asyncio'
    ]
    stuff_missing_from_computer = []
    
    for package in list_of_crap_we_need:
        try:
            __import__(package.replace('-', '_'))
        except (ImportError, OSError):
            stuff_missing_from_computer.append(package)
    
    return stuff_missing_from_computer

def install_dependencies():
    print(f"Python version: {sys.version}")
    
    stuff_missing_from_computer = check_dependencies()
    if not stuff_missing_from_computer:
        print("All dependencies already installed! nice.")
        return True
    
    print(f"Missing packages: {', '.join(stuff_missing_from_computer)}")
    
    # Try to upgrade pip first because old pip is dumb
    try:
        print("Updating pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    except:
        pass

    python_version_info_thing = sys.version_info
    
    if python_version_info_thing >= (3, 10):
        packages_to_get = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "transformer-lens",
            "scikit-learn",
            "fastapi",
            "uvicorn[standard]",
            "nest-asyncio",
            "einops",
        ]
        print("Using modern package versions (Python 3.10+)")
    else:
        packages_to_get = [
            "numpy==1.23.5",
            "torch==2.1.0",
            "transformers==4.35.0",
            "transformer-lens",
            "scikit-learn",
            "fastapi",
            "uvicorn[standard]",
            "nest-asyncio",
            "einops",
        ]
        print("Using legacy package versions")
    
    # Attempt batch installation first (better for dependency resolution)
    print("Attempting batch installation...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + packages_to_get,
            check=True
        )
        print("Batch installation successful!")
        return True
    except subprocess.CalledProcessError:
        print("Batch installation crashed. Trying the slow individual way...")

    # Fallback: Install one by one
    for thingy in packages_to_get:
        print(f"  Installing {thingy}...")
        try:
            # Removed --quiet so we can actually see why it fails
            subprocess.run(
                [sys.executable, "-m", "pip", "install", thingy],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to install {thingy}")
            name_of_the_broken_package = thingy.split("==")[0].split(">=")[0].split("[")[0]
            try:
                print(f"  Trying loose install for {name_of_the_broken_package}...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", name_of_the_broken_package],
                    check=True
                )
                print(f"  Installed {name_of_the_broken_package}")
            except:
                print(f"  Could not install {name_of_the_broken_package}")
                return False
    
    print("All dependencies installed! finally.")
    return True

if __name__ == "__main__":
    if not install_dependencies():
        print("\nDependency installation failed!")
        # Don't exit immediately, maybe imports will work anyway?
        print("Attempting to run anyway...")
    
print("\nLoading InterpControl...")

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import torch
    import numpy as np
    from transformer_lens import HookedTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.decomposition import PCA
    import asyncio
    import threading
    import warnings
    warnings.filterwarnings('ignore')
    print("All imports successful!")
except (ImportError, OSError) as e:
    print(f"\nIMPORT ERROR: {e}")
    print("Try running: pip install -r requirements.txt")
    sys.exit(1)

some_random_settings_dict = {
    'model_name': os.getenv('MODEL_NAME', 'gpt2-small'),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'port': int(os.getenv('PORT', 8000)),
    'host': os.getenv('HOST', '0.0.0.0')
}

print(f"Device: {some_random_settings_dict['device']}")
print(f"Model: {some_random_settings_dict['model_name']}")
print(f"Port: {some_random_settings_dict['port']}")

class InterpController:
    
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or some_random_settings_dict['model_name']
        self.device = device or some_random_settings_dict['device']
        
        print(f"Loading {self.model_name}...")
        self.the_actual_model_object = HookedTransformer.from_pretrained(
            self.model_name, 
            device=self.device
        )
        
        self.things_that_poke_the_model = {}
        self.vectors_for_steering_the_brain = {}
        self.saved_activations_cache_thing = {}
        print(f"Model loaded on {self.device}!")

    def get_training_data(self):
        return [
            ("The capital of France is Paris", 1),
            ("The capital of Germany is Berlin", 1),
            ("The capital of Italy is Rome", 1),
            ("The capital of Spain is Madrid", 1),
            ("The capital of France is London", 0),
            ("The capital of Germany is Paris", 0),
            ("The capital of Italy is Madrid", 0),
            ("The capital of Spain is Berlin", 0),
            ("Water is made of hydrogen and oxygen", 1),
            ("The Earth orbits the Sun", 1),
            ("Water is made of hydrogen and nitrogen", 0),
            ("The Sun orbits the Earth", 0),
            ("Two plus two equals four", 1),
            ("Ten divided by two equals five", 1),
            ("Two plus two equals five", 0),
            ("Ten divided by two equals four", 0),
        ]

    def train_probe(self, which_layer_is_it):
        
        training_data_pile = self.get_training_data()
        texts, labels = zip(*training_data_pile)
        
        pile_of_activations = []
        
        for text in texts:
            _, cache = self.the_actual_model_object.run_with_cache(text, return_type=None)
            act = cache[f"blocks.{which_layer_is_it}.hook_resid_post"][0, -1, :].cpu().detach().numpy()
            pile_of_activations.append(act)
        
        X = np.array(pile_of_activations)
        y = np.array(labels)
        
        probe_thingy = LogisticRegression(max_iter=1000, random_state=42)
        probe_thingy.fit(X, y)
        
        self.things_that_poke_the_model[which_layer_is_it] = probe_thingy
        self.vectors_for_steering_the_brain[which_layer_is_it] = probe_thingy.coef_[0]
        self.saved_activations_cache_thing[which_layer_is_it] = (X, y)
        
        predictions = probe_thingy.predict(X)
        accuracy_number = accuracy_score(y, predictions)
        
        try:
            weird_confusion_box = confusion_matrix(y, predictions).ravel().tolist()
        except:
            weird_confusion_box = [0, 0, 0, 0]
        
        print(f"  Probe trained on layer {which_layer_is_it} - Accuracy: {accuracy_number*100:.1f}%")
        return accuracy_number, weird_confusion_box

    def get_confidence(self, text, which_layer_is_it):
        _, cache = self.the_actual_model_object.run_with_cache(text, return_type=None)
        act = cache[f"blocks.{which_layer_is_it}.hook_resid_post"][0, -1, :].cpu().detach().numpy().reshape(1, -1)
        return float(self.things_that_poke_the_model[which_layer_is_it].predict_proba(act)[0][1])

    def generate_steered(self, text, which_layer_is_it, steering_strength):
        
        steering_vector_tensor_thing = torch.tensor(
            self.vectors_for_steering_the_brain[which_layer_is_it], 
            dtype=torch.float32,
            device=self.device
        )
        
        def the_hook_function(activations, hook):
            activations[:, :, :] += steering_strength * steering_vector_tensor_thing
            return activations
        
        with self.the_actual_model_object.hooks(fwd_hooks=[(f"blocks.{which_layer_is_it}.hook_resid_post", the_hook_function)]):
            return self.the_actual_model_object.generate(text, max_new_tokens=20, verbose=False)

    def chain_of_thought(self, question):
        prompt = f"Question: {question}\nAnalysis:"
        analysis_text = self.the_actual_model_object.generate(prompt, max_new_tokens=30, verbose=False)
        final_answer = self.the_actual_model_object.generate(prompt + analysis_text + "\nAnswer:", max_new_tokens=10, verbose=False)
        return analysis_text, final_answer

    def get_pca_visualization(self, which_layer_is_it):
        if which_layer_is_it not in self.saved_activations_cache_thing:
            return []
            
        X, y = self.saved_activations_cache_thing[which_layer_is_it]
        pca_math_obj = PCA(n_components=3)
        coordinates = pca_math_obj.fit_transform(X)
        
        return [{"x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "label": int(l)} 
                for c, l in zip(coordinates, y)]

the_web_server_thing = FastAPI(
    title="InterpControl",
    description="Mechanistic Interpretability Dashboard for Transformer Models",
    version="1.0.0"
)

big_controller_man = InterpController()

the_web_server_thing.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class TheTrainingDataRequestObj(BaseModel):
    layer: int

class TheInferenceRequestObj(BaseModel):
    text: str
    layer: int
    steering_coef: float = 0.0

@the_web_server_thing.get("/")
async def root():
    return HTMLResponse(get_html_ui())

@the_web_server_thing.post("/train")
async def train(req: TheTrainingDataRequestObj):
    acc, conf = big_controller_man.train_probe(req.layer)
    return {
        "accuracy": acc, 
        "pca": big_controller_man.get_pca_visualization(req.layer), 
        "confusion": conf
    }

@the_web_server_thing.post("/infer")
async def infer(req: TheInferenceRequestObj):
    if req.layer not in big_controller_man.things_that_poke_the_model:
        big_controller_man.train_probe(req.layer)
    
    confidence_score = big_controller_man.get_confidence(req.text, req.layer)
    
    if req.steering_coef != 0:
        result_text = big_controller_man.generate_steered(req.text, req.layer, req.steering_coef)
        system_name = "Steered"
        trace_log = ""
    elif confidence_score > 0.65:
        result_text = big_controller_man.the_actual_model_object.generate(req.text, max_new_tokens=20, verbose=False)
        system_name = "System 1 (Fast)"
        trace_log = ""
    else:
        trace_log, result_text = big_controller_man.chain_of_thought(req.text)
        system_name = "System 2 (Slow)"
    
    return {
        "confidence": confidence_score, 
        "output": result_text, 
        "system": system_name, 
        "trace": trace_log
    }

@the_web_server_thing.get("/health")
async def health():
    return {"status": "ok", "model": some_random_settings_dict['model_name'], "device": some_random_settings_dict['device']}

def get_html_ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>InterpControl</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: #0a0a0a; color: #fff; font-family: 'Monaco', 'Courier New', monospace; }
        .glass { background: rgba(30,30,30,0.8); border: 1px solid #333; border-radius: 8px; }
        .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    </style>
</head>
<body class="p-8">
    <div class="max-w-4xl mx-auto">
        <div class="mb-8">
            <h1 class="text-4xl font-bold mb-2">INTERP<span class="text-green-500">CONTROL</span></h1>
            <p class="text-sm text-gray-400">Mechanistic Interpretability Dashboard</p>
            <a href="https://github.com/yourusername/interpcontrol" 
               class="text-xs text-blue-400 hover:text-blue-300">View on GitHub →</a>
        </div>
        
        <div class="glass p-6 mb-4">
            <div class="mb-4">
                <label class="block mb-2 text-sm font-semibold">
                    Layer: <span id="layer-val" class="text-green-500">6</span>
                </label>
                <input type="range" min="3" max="9" value="6" id="layer" class="w-full">
                <p class="text-xs text-gray-500 mt-1">Select which transformer layer to probe</p>
            </div>
            
            <div class="mb-4">
                <label class="block mb-2 text-sm font-semibold">
                    Steering Coefficient: <span id="steer-val" class="text-blue-500">0.0</span>x
                </label>
                <input type="range" min="-5" max="5" step="0.5" value="0" id="steering" class="w-full">
                <p class="text-xs text-gray-500 mt-1">Adjust activation steering strength</p>
            </div>
            
            <button onclick="train()" 
                    class="bg-green-600 hover:bg-green-700 px-4 py-2 mb-4 w-full rounded font-semibold transition">
                Train Probe
            </button>
            
            <input type="text" id="query" value="The capital of France is" 
                   class="w-full bg-gray-800 p-3 mb-2 rounded border border-gray-700 focus:border-blue-500 outline-none" 
                   placeholder="Enter prompt...">
            <button onclick="infer()" 
                    class="bg-blue-600 hover:bg-blue-700 px-4 py-2 w-full rounded font-semibold transition">
                Run Inference
            </button>
        </div>
        
        <div id="output" class="glass p-6 min-h-32">
            <p class="text-gray-500 text-sm">Train a probe to get started...</p>
        </div>
        
        <div class="mt-4 text-xs text-gray-600 text-center">
            <p>Powered by TransformerLens • GPT-2 Small</p>
        </div>
    </div>
    
    <script>
        const layer = document.getElementById('layer');
        const steering = document.getElementById('steering');
        const query = document.getElementById('query');
        const output = document.getElementById('output');
        
        layer.oninput = (e) => document.getElementById('layer-val').innerText = e.target.value;
        steering.oninput = (e) => document.getElementById('steer-val').innerText = parseFloat(e.target.value).toFixed(1);
        
        async function train() {
            output.innerHTML = '<div class="text-yellow-400">Training probe on layer ' + layer.value + '...</div>';
            try {
                const res = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({layer: parseInt(layer.value)})
                });
                const data = await res.json();
                output.innerHTML = `
                    <div class="text-green-400 text-lg">Probe trained successfully!</div>
                    <div class="text-sm text-gray-400 mt-2">Accuracy: ${(data.accuracy*100).toFixed(1)}%</div>
                    <div class="text-xs text-gray-500 mt-1">Confusion Matrix: [${data.confusion.join(', ')}]</div>
                `;
            } catch (e) {
                output.innerHTML = '<div class="text-red-400">Training failed: ' + e.message + '</div>';
            }
        }
        
        async function infer() {
            output.innerHTML = '<div class="text-yellow-400">Processing...</div>';
            try {
                const res = await fetch('/infer', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        text: query.value,
                        layer: parseInt(layer.value),
                        steering_coef: parseFloat(steering.value)
                    })
                });
                const data = await res.json();
                const color = data.confidence > 0.65 ? 'green' : 'red';
                output.innerHTML = `
                    <div class="mb-4 flex items-center gap-3">
                        <span class="text-${color}-400 font-semibold">
                            Confidence: ${(data.confidence*100).toFixed(1)}%
                        </span>
                        <span class="badge bg-gray-700">${data.system}</span>
                    </div>
                    ${data.trace ? `<div class="text-sm text-yellow-400 mb-3 p-2 bg-gray-800 rounded">
                        Analysis: ${data.trace}
                    </div>` : ''}
                    <div class="text-lg bg-gray-800 p-3 rounded border border-gray-700">
                        ${data.output}
                    </div>
                `;
            } catch (e) {
                output.innerHTML = '<div class="text-red-400">Inference failed: ' + e.message + '</div>';
            }
        }
        
        // Auto-train on page load
        window.onload = () => train();
    </script>
</body>
</html>
"""

def start_server(host=None, port=None):
    host = host or some_random_settings_dict['host']
    port = port or some_random_settings_dict['port']
    
    try:
        from google.colab import output
        output.serve_kernel_port_as_iframe(port, height=800)
        print(f"\nInterpControl UI embedded below!")
    except:
        print(f"\n{'='*60}")
        print(f"InterpControl is running!")
        print(f"Open: http://localhost:{port}")
        print(f"{'='*60}\n")
    
    import nest_asyncio
    nest_asyncio.apply()
    
    config_object_thing = uvicorn.Config(
        the_web_server_thing, 
        host=host, 
        port=port, 
        log_level="error"
    )
    actual_server_instance = uvicorn.Server(config_object_thing)
    
    background_thread_dude = threading.Thread(target=lambda: asyncio.run(actual_server_instance.serve()))
    background_thread_dude.daemon = True
    background_thread_dude.start()
    
    print("Server started!")
    print("Tip: Try different steering coefficients to see effects on generation")

if __name__ == "__main__":
    start_server()
    
    try:
        from google.colab import output
    except:
        print("\nPress Ctrl+C to stop the server")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped")
