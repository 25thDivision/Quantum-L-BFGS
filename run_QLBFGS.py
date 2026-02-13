import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import noise

# 커스텀 모듈 import
from models.pennylane_qiskit_model import PennyLaneQiskitModel
from optimizers.qlbfgs import QLBFGS

def get_depolarizing_noise(prob=0.01):
    """
    노이즈 모델 생성 함수
    """
    noise_model = noise.NoiseModel()
    # 1-qubit gate error
    error_1 = noise.depolarizing_error(prob, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz', 'h'])
    # 2-qubit gate error
    error_2 = noise.depolarizing_error(prob * 10, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    return noise_model

def plot_results(history, save_path, n_qubits):
    """
    학습 이력(history)을 받아 수렴 그래프를 그리고 저장하는 함수
    """
    energies = [h[0] for h in history]
    grads = [h[1] for h in history]

    plt.figure(figsize=(10, 5))
    
    # 1. 에너지 그래프
    plt.subplot(1, 2, 1)
    plt.plot(energies, 'b-o', label='Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(f'Convergence ({n_qubits} Qubits)')
    plt.grid(True)
    plt.legend()

    # 2. 기울기(Gradient Norm) 그래프
    plt.subplot(1, 2, 2)
    plt.plot(grads, 'r-s', label='Grad Norm')
    plt.xlabel('Iteration')
    plt.ylabel('|Gradient|')
    plt.title('Gradient Magnitude')
    plt.grid(True)
    plt.legend()

    # 저장 및 닫기 (메모리 누수 방지)
    plt.savefig(save_path)
    plt.close() 
    print(f"Convergence plot saved to: {save_path}")

def run_experiment(config):
    """
    설정(Config)을 입력받아 전체 실험을 수행하는 메인 함수
    """
    # 1. 설정 로딩 (Parsing)
    n_qubits = config["physics"]["n_qubits"]
    layers = config["physics"]["layers"]
    
    shots = config["simulation"]["shots"]
    noise_prob = config["simulation"]["noise_prob"]
    seed = config["simulation"]["seed"]
    
    max_iter = config["optimizer"]["max_iter"]
    tol = config["optimizer"]["tol"]
    hist_size = config["optimizer"]["history_size"]
    
    save_dir = config["output"]["save_dir"]
    plot_path = os.path.join(save_dir, config["output"]["plot_filename"])
    param_path = os.path.join(save_dir, config["output"]["param_filename"])

    # 저장 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"=== Running Experiment: QLBFGS ({n_qubits} Qubits) ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # 2. 모델 및 노이즈 초기화
    my_noise_model = get_depolarizing_noise(prob=noise_prob)
    
    print("\nInitializing Quantum Model...")
    vqe_model = PennyLaneQiskitModel(
        n_qubits=n_qubits, 
        layers=layers, 
        shots=shots, 
        noise_model=my_noise_model
    )

    # 3. 최적화기 설정
    optimizer = QLBFGS(
        model=vqe_model, 
        history_size=hist_size, 
        tol=tol
    )

    # 4. 파라미터 초기화
    np.random.seed(seed)
    total_params = layers * n_qubits
    initial_params = np.random.uniform(0, 2*np.pi, size=total_params)

    # 5. 최적화 실행
    print("\nStarting Optimization Loop...")
    best_params, history = optimizer.optimize(initial_params, max_iter=max_iter)

    # 6. 결과 처리
    print("\nExperiment Finished!")
    final_energy = history[-1][0]
    print(f"Final Energy: {final_energy:.6f}")

    # 파라미터 저장
    np.save(param_path, best_params)
    print(f"Best parameters saved to: {param_path}")

    # [수정됨] 그래프 그리기 함수 호출 (코드가 깔끔해짐)
    plot_results(history, plot_path, n_qubits)
    
    return best_params, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QLBFGS Quantum Experiment")
    
    # [수정됨] 기본값을 configs/QLBFGS.json (대문자)으로 변경
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/QLBFGS.json", 
        help="Path to the configuration JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as f:
            config_data = json.load(f)
        
        run_experiment(config_data)
        
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        print("Please check if the file exists or check the filename case (QLBFGS vs qlbfgs).")
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file {args.config}")