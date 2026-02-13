import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from .base_model import QuantumBase

class PennyLaneQiskitModel(QuantumBase):
    def __init__(self, config, noise_model=None):
        self.n_qubits = config["physics"]["n_qubits"]
        self.layers = config["physics"]["layers"]
        self.J = config["physics"].get("J", 1.0)
        self.h = config["physics"].get("h", 2.0)
        self.shots = config["simulation"]["shots"]
        self.structure = config["physics"].get("structure", "full")
        
        
        if noise_model is None:
            self.dev = qml.device("lightning.qubit", wires=self.n_qubits)
            diff_method = "adjoint"
        else:
            # Statevector 시뮬레이터 사용 (Fidelity 계산 효율성 및 시뮬레이션 속도)
            self.dev = qml.device(
                "qiskit.aer", 
                wires=self.n_qubits, 
                noise_model=noise_model,
                method="statevector",
                max_parallel_experiments=16,
                max_parallel_threads=1
            )
            diff_method = "parameter-shift"

        # ---------------------------------------------------------
        # 1. Hamiltonian 정의 (Ising Model)
        # ---------------------------------------------------------
        coeffs = []
        obs = []
        
        # =========================================================
        # [A] Interaction Term (상호작용)
        # =========================================================

        # [Option 1] Linear (선형 / Open Boundary)
        # 구조: 0-1, 1-2, 2-3 ... (양 끝 끊어짐) -> 논문 12큐빗(Linear) 실험용
        if self.structure == 'linear':
            for i in range(self.n_qubits - 1):
                coeffs.append(-self.J)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

        # [Option 2] Reverse Linear (역선형)
        # 구조: 3-2, 2-1, 1-0 ... (Linear와 물리적으로 같지만 순서만 반대)
        elif self.structure == 'reverse_linear':
            for i in range(self.n_qubits - 1, 0, -1):
                coeffs.append(-self.J)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(i - 1))

        # [Option 3] Circular (원형 / Periodic Boundary)
        # 구조: 0-1, 1-2, ... (N-1)-0 (반지 모양) -> 아까 4큐빗 대박 실험용
        elif self.structure == 'circular':
            for i in range(self.n_qubits):
                coeffs.append(-self.J)
                obs.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % self.n_qubits))

        # [Option 4] Full (완전 연결 / All-to-All)
        # 구조: 모든 큐빗끼리 서로 연결 (가장 복잡한 문제)
        else:
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    coeffs.append(-self.J)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

        # =========================================================
        # [B] Transverse Field Term (자기장)
        # =========================================================
        # 0. 자기장 세기
        for i in range(self.n_qubits):
            coeffs.append(-self.h)
            obs.append(qml.PauliX(i))

        self.hamiltonian = qml.Hamiltonian(coeffs, obs)

        # 1. Cost Function (Energy)
        @qml.qnode(self.dev, diff_method=diff_method, shots=self.shots)
        def circuit(params):
            self._ansatz(params)
            return qml.expval(self.hamiltonian)
        self.cost_fn = circuit

        # 2. State Function (Fidelity용)
        @qml.qnode(self.dev, interface="numpy")
        def state_circuit(params):
            self._ansatz(params)
            return qml.state()
        self.state_fn = state_circuit

    def _ansatz(self, params):
        """
        Qiskit RealAmplitudes (Reps=Layers) 스타일 Ansatz
        원하는 얽힘 구조(Entanglement)의 주석을 풀어서 사용하세요.
        """
        idx = 0
        for l in range(self.layers):
            # ==========================================
            # 1. Rotation Layer (공통)
            # ==========================================
            for q in range(self.n_qubits):
                if idx < len(params):
                    qml.RY(params[idx], wires=q)
                    idx += 1
            
            # ==========================================
            # 2. Entanglement Layer
            # ==========================================
            
            # [Option A] Linear (선형)
            # 구조: 0-1, 1-2, 2-3 ... (양 끝 연결 안 됨)
            if self.structure == 'linear':
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # [Option B] Reverse Linear (역선형)
            # 구조: 3-2, 2-1, 1-0 ... (거꾸로 연결)
            elif self.structure == 'reverse_linear':
                for i in range(self.n_qubits - 1, 0, -1):
                    qml.CNOT(wires=[i, i - 1])

            # [Option C] Circular (원형/Periodic)
            # 구조: 0-1, 1-2, 2-3, 3-0 (반지 모양)
            elif self.structure == 'circular':
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            # [Option D] Full (완전 연결)
            # 구조: 모든 큐빗끼리 서로 연결 (가장 복잡하고 강력함)
            else:
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])

        # ==========================================
        # 3. Final Rotation Layer (선택 사항)
        # ==========================================
        # Qiskit RealAmplitudes는 마지막에 Rot를 한 번 더 하지만,
        # 파라미터 개수를 맞추기 위해 보통 생략하거나 주석 처리함.
        for q in range(self.n_qubits):
            if idx < len(params):
                qml.RY(params[idx], wires=q)
                idx += 1

    def get_energy(self, params):
        return float(self.cost_fn(params))

    def get_gradient(self, params):
        p_params = pnp.array(params, requires_grad=True)
        grad_fn = qml.grad(self.cost_fn)
        gradients = grad_fn(p_params)
        return np.array(gradients, dtype=float)

    def get_fidelity(self, params1, params2):
        """
        두 상태 간의 Fidelity 계산: |<psi(p1)|psi(p2)>|^2
        Metric 근사를 위해 사용됨.
        """
        state1 = self.state_fn(params1)
        state2 = self.state_fn(params2)
        # 내적의 절대값 제곱
        overlap = np.vdot(state1, state2)
        return float(np.abs(overlap)**2)
    
    def get_metric_diag(self, params: np.ndarray) -> np.ndarray:
        """
        Returns Identity metric diagonal.
        SP-BFGS는 내부적으로 Metric을 계산하므로 여기서는 기본값만 반환하여
        인스턴스화 에러를 방지합니다.
        """
        return np.ones_like(params)