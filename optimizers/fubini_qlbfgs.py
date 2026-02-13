import numpy as np
from collections import deque
from models.base_model import QuantumBase

class FubiniQLBFGS:
    def __init__(self, model: QuantumBase, history_size=10, tol=1e-4, spsa_delta=0.01):
        """
        [Version 1] Fubini-Study Initialization based SP-BFGS Optimizer.
        
        Args:
            model: QuantumModel (Energy & PSR Gradient)
            history_size: L-BFGS Memory limit (m)
            tol: Convergence tolerance
            spsa_delta: SPSA perturbation size (for initialization)
        """
        self.model = model
        self.m = history_size
        self.tol = tol
        self.delta = spsa_delta
        
        # History Deques (Standard L-BFGS)
        self.s_history = deque(maxlen=history_size)
        self.y_history = deque(maxlen=history_size)
        self.rho_history = deque(maxlen=history_size)
        
        # [핵심] Step 0에서 구한 Metric 대각 성분을 저장할 변수
        self.H0_diag = None 

    def _initialize_metric_spsa(self, params):
        """
        [Step 0] 4-Shot QN-SPSA to estimate Diagonal Metric.
        Fubini-Study Metric의 대각 성분을 근사하여 Warm Start를 유도함.
        """
        dim = len(params)
        
        # 1. Perturbation Vector (Bernoulli +/- 1)
        # 모든 파라미터를 동시에 흔듭니다.
        delta_vec = np.random.choice([-1, 1], size=dim) * self.delta
        
        # 2. 4-Shot Measurement
        # 지점 1: 현재 위치 (params)
        # E(theta + delta), E(theta - delta)
        ep1 = self.model.get_energy(params + delta_vec)
        em1 = self.model.get_energy(params - delta_vec)
        # SPSA Gradient approx at theta
        g1 = (ep1 - em1) / (2 * self.delta) * np.sign(delta_vec) # Element-wise
        
        # 지점 2: 아주 조금 이동한 위치 (params + step)
        # 곡률을 보기 위해 섭동 방향으로 살짝 이동
        probe_step = 0.01 * delta_vec
        params_2 = params + probe_step
        
        # E(theta' + delta), E(theta' - delta)
        ep2 = self.model.get_energy(params_2 + delta_vec)
        em2 = self.model.get_energy(params_2 - delta_vec)
        # SPSA Gradient approx at theta'
        g2 = (ep2 - em2) / (2 * self.delta) * np.sign(delta_vec)
        
        # 3. Diagonal Hessian/Metric Estimation (Secant Equation)
        # y = H * s  =>  (g2 - g1) ~ H_diag * probe_step
        y_diff = g2 - g1
        s_diff = probe_step
        
        # 0으로 나누기 방지 및 절대값 (Metric은 양수여야 함)
        # Regularization: + 1e-3 (너무 작은 값이 나오면 발산하므로 보정)
        diag_metric = np.abs(y_diff / (s_diff + 1e-10)) + 1e-3
        
        # Hessian Inverse Approximation (H_0 = M^{-1})
        H0_diag = 1.0 / diag_metric
        
        return H0_diag

    def _get_direction(self, grad):
        """
        L-BFGS Two-loop recursion.
        초기 Hessian B_0를 SPSA로 구한 H0_diag로 설정하여 계산.
        """
        q = grad.copy()
        alphas = []
        
        # 1. Backward Pass
        for s, y, rho in reversed(list(zip(self.s_history, self.y_history, self.rho_history))):
            alpha = rho * np.dot(s, q)
            q -= alpha * y
            alphas.append(alpha)
            
        # 2. Scaling / Initialization (B_0)
        # History가 없으면(초반) SPSA 초기값을 사용.
        # History가 쌓여도 "Base Geometry"로 SPSA 값을 사용할지,
        # 아니면 표준 L-BFGS 처럼 gamma*I 를 쓸지 결정해야 함.
        # 여기서는 "Front-loading" 철학에 따라, History가 비었을 때 SPSA 값을 확실히 사용.
        
        if not self.s_history:
             # [Warm Start] SPSA Metric 정보 사용
             r = self.H0_diag * q
        else:
            # [Standard L-BFGS Scaling]
            # 수렴 단계에서는 최근 곡률 정보(s, y)가 더 정확하므로 스칼라 스케일링으로 전환
            last_s = self.s_history[-1]
            last_y = self.y_history[-1]
            gamma = np.dot(last_s, last_y) / (np.dot(last_y, last_y) + 1e-10)
            r = gamma * q
        
        # 3. Forward Pass
        for (s, y, rho), alpha in zip(list(zip(self.s_history, self.y_history, self.rho_history)), reversed(alphas)):
            beta = rho * np.dot(y, r)
            r += s * (alpha - beta)
            
        return -r # Descent direction

    def _line_search(self, params, direction, current_energy, current_grad):
        """Backtracking Line Search"""
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        max_ls_iter = 5
        
        dir_deriv = np.dot(current_grad, direction)
        
        if dir_deriv > 0: # 방향 보정
            direction = -direction
            dir_deriv = -dir_deriv
            
        for i in range(max_ls_iter):
            cand_params = params + alpha * direction
            cand_energy = self.model.get_energy(cand_params)
            
            if cand_energy <= current_energy + c1 * alpha * dir_deriv:
                return alpha, cand_params, cand_energy
            
            alpha *= rho
            
        return alpha, params + alpha * direction, self.model.get_energy(params + alpha * direction)

    def optimize(self, initial_params, max_iter=50):
        params = np.array(initial_params, dtype=float)
        
        # --- [Step 0] Initialization (Front-loading) ---
        print(">>> Step 0: Initializing Metric via QN-SPSA (4 Shots)...")
        self.H0_diag = self._initialize_metric_spsa(params)
        print(f"    Metric Init Complete. Mean Scale: {np.mean(self.H0_diag):.4f}")
        # -----------------------------------------------
        
        # Initial Gradient (PSR: 2n shots)
        energy = self.model.get_energy(params)
        grad = self.model.get_gradient(params)
        
        print(f"Initial Energy: {energy:.6f}")
        history = [(energy, np.linalg.norm(grad))]
        
        best_params = params.copy()
        best_energy = energy

        for k in range(max_iter):
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tol:
                print(f"Converged at iter {k}")
                break
            
            # 1. Direction (using SPSA init or BFGS history)
            p_k = self._get_direction(grad)
            
            # 2. Line Search
            alpha, params_new, energy_new = self._line_search(params, p_k, energy, grad)
            
            # 3. Record & Update
            if energy_new < best_energy:
                best_energy = energy_new
                best_params = params_new.copy()
                print(f"  > New Best: {best_energy:.6f}")
                
            s_k = params_new - params
            
            if np.linalg.norm(s_k) < 1e-9:
                print("Step too small.")
                break
                
            grad_new = self.model.get_gradient(params_new)
            y_k = grad_new - grad
            
            # 4. History Update (Curvature Check)
            ys = np.dot(y_k, s_k)
            if ys > 1e-10:
                rho = 1.0 / ys
                self.s_history.append(s_k)
                self.y_history.append(y_k)
                self.rho_history.append(rho)
            
            params = params_new
            energy = energy_new
            grad = grad_new
            
            history.append((energy, grad_norm))
            print(f"Iter {k+1}: E={energy:.6f}, |g|={grad_norm:.4f}, Step={alpha:.4f}")
            
        return best_params, history