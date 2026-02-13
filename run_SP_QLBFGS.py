import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
from qiskit_aer import noise

# 모듈 Import
from optimizers.sp_qlbfgs import SP_QLBFGS
from models.pennylane_qiskit_model import PennyLaneQiskitModel

# --- [유틸리티] 디렉토리 및 경로 관리 ---
def setup_directories(config):
    base_dir = config["output"]["base_dir"]
    if config["simulation"]["noise_prob"] > 0.0:
        noise_type = f"p{config["simulation"]["noise_prob"]}"
    else:
        noise_type = "ideal"
    exp_name = (
        f"{config["optimizer"]["name"]}_"
        f"{config["physics"]["ansatz"]}"
        f"({config["physics"]["n_qubits"]},{config["physics"]["layers"]})_"
        f"{config["physics"]["structure"]}_"
        f"J{int(config["physics"]["J"])}h{config["physics"]["h"]}_"
        f"{noise_type}"
    )
    
    # 3개의 서브 폴더 정의
    dirs = {
        "params": os.path.join(base_dir, "best"),
        "history": os.path.join(base_dir, "history"),
        "figures": os.path.join(base_dir, "figures")
    }
    
    # 폴더가 없으면 생성
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    # 저장될 파일 경로 생성
    paths = {
        "params": os.path.join(dirs["params"], f"{exp_name}_params.npy"),
        "history": os.path.join(dirs["history"], f"{exp_name}_history.npy"),
        "plot": os.path.join(dirs["figures"], f"{exp_name}_plot.png")
    }
    
    return paths, exp_name

def get_depolarizing_noise(prob=0.01):
    noise_model = noise.NoiseModel()
    error_1 = noise.depolarizing_error(prob, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz', 'h'])
    error_2 = noise.depolarizing_error(prob * 10, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    return noise_model

def plot_results(history, save_path, n_qubits):
    energies = [h[0] for h in history]
    grads = [h[1] for h in history]

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(energies, 'm-o', label='Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title(f'Convergence ({n_qubits} Qubits)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(grads, 'k--', label='Grad Norm')
    plt.xlabel('Iteration')
    plt.ylabel('|Gradient|')
    plt.title('Gradient Descent')
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")

def run_experiment(config):
    # 1. Config & Path Setup
    paths, exp_name = setup_directories(config)
    
    n_qubits = config["physics"]["n_qubits"]
    layers = config["physics"]["layers"]
    noise_prob = config["simulation"]["noise_prob"]
    seed = config["simulation"]["seed"]
    
    max_iter = config["optimizer"]["max_iter"]
    tol = config["optimizer"]["tol"]
    hist_size = config["optimizer"]["history_size"]
    spsa_eps = config["optimizer"].get("spsa_epsilon", 0.01)
    
    # 저장 옵션 확인
    do_save_history = config["output"].get("save_history", True)
    do_save_plot = config["output"].get("save_plot", True)

    print(f"=== Running Experiment: {exp_name} ===")
    
    # 2. Noise Model Setup
    if noise_prob > 0.0:
        my_noise_model = get_depolarizing_noise(prob=noise_prob)
        print(f"Noise Model Active: Depolarizing({noise_prob})")
    else:
        my_noise_model = None
        print("Noise Model Deactivated: Ideal Simulation")

    # 3. Model Init
    vqe_model = PennyLaneQiskitModel(
        config=config, noise_model=my_noise_model
    )

    # 4. Optimizer Init
    optimizer = SP_QLBFGS(
        model=vqe_model, 
        history_size=hist_size, 
        tol=tol,
        spsa_epsilon=spsa_eps
    )

    # 5. Run Optimization
    np.random.seed(seed)
    # initial_params = np.random.uniform(0, 2*np.pi, size=layers * n_qubits)
    initial_params = np.random.uniform(0, 2*np.pi, size=(layers + 1) * n_qubits)
    
    print("\nStarting Optimization...")
    best_params, history = optimizer.optimize(initial_params, max_iter=max_iter)
    
    final_energy = history[-1][0]
    print(f"\nExperiment Finished! Final Energy: {final_energy:.6f}")

    # 6. Save Results (Organized)
    # (1) Best Parameters (항상 저장)
    np.save(paths["params"], best_params)
    print(f"Best Params saved to: {paths['params']}")
    
    # (2) History (옵션)
    if do_save_history:
        np.save(paths["history"], np.array(history))
        print(f"History saved to: {paths['history']}")
        
    # (3) Plot (옵션)
    if do_save_plot:
        plot_results(history, paths["plot"], n_qubits)

    return best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/SP_QLBFGS.json")
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as f:
            config_data = json.load(f)
        run_experiment(config_data)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")