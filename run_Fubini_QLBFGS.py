import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from qiskit_aer import noise

# 커스텀 모듈 Import
from optimizers.fubini_init_qlbfgs import FubiniInitQLBFGS
from models.pennylane_qiskit_model import PennyLaneQiskitModel

def get_depolarizing_noise(prob=0.01):
    noise_model = noise.NoiseModel()
    error_1 = noise.depolarizing_error(prob, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz', 'h'])
    error_2 = noise.depolarizing_error(prob * 10, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    return noise_model

def plot_results(history, save_path, n_qubits):
    energies = [h[0] for h in history]
    plt.figure(figsize=(10, 5)) # 그래프 크기 통일
    plt.plot(energies, 'g-o', label='Fubini-SP-BFGS Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(f'Fubini-Init Optimization ({n_qubits} Qubits)')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")

def run_experiment(config):
    # 1. Config Parsing
    n_qubits = config["physics"]["n_qubits"]
    layers = config["physics"]["layers"]
    
    shots = config["simulation"]["shots"]
    noise_prob = config["simulation"]["noise_prob"]
    seed = config["simulation"]["seed"]
    
    max_iter = config["optimizer"]["max_iter"]
    tol = config["optimizer"]["tol"]
    hist_size = config["optimizer"]["history_size"]
    # [추가] Fubini 전용 파라미터
    spsa_delta = config["optimizer"].get("spsa_delta", 0.01) # 없으면 기본값 0.01
    
    save_dir = config["output"]["save_dir"]
    plot_path = os.path.join(save_dir, config["output"]["plot_filename"])
    param_path = os.path.join(save_dir, config["output"]["param_filename"])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"=== Running Fubini-Init SP-BFGS ({n_qubits} Qubits) ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # 2. Model Init
    my_noise_model = get_depolarizing_noise(prob=noise_prob)
    vqe_model = PennyLaneQiskitModel(
        n_qubits=n_qubits, layers=layers, shots=shots, noise_model=my_noise_model
    )
    
    # 3. Optimizer Init (Version 1)
    # Config에서 읽은 값들을 주입
    optimizer = FubiniInitQLBFGS(
        model=vqe_model, 
        history_size=hist_size, 
        tol=tol,
        spsa_delta=spsa_delta 
    )
    
    # 4. Run
    np.random.seed(seed)
    initial_params = np.random.uniform(0, 2*np.pi, size=layers * n_qubits)
    
    print("\nStarting Optimization Loop...")
    best_params, history = optimizer.optimize(initial_params, max_iter=max_iter)
    
    # 5. Save Results
    plot_results(history, plot_path, n_qubits)
    
    final_energy = history[-1][0]
    print(f"\nExperiment Finished!")
    print(f"Final Energy: {final_energy:.6f}")
    
    np.save(param_path, best_params)
    print(f"Best parameters saved to: {param_path}")
    
    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fubini-Init QLBFGS Experiment")
    
    # 기본값을 새로 만든 JSON 파일로 지정
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/Fubini_QLBFGS.json", 
        help="Path to the configuration JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as f:
            config_data = json.load(f)
        run_experiment(config_data)
        
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file {args.config}")