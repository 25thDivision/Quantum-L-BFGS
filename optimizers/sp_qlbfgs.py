import numpy as np
from collections import deque
from .qlbfgs import QLBFGS

class SP_QLBFGS(QLBFGS):
    def __init__(self, model, history_size=10, tol=1e-3, spsa_epsilon=0.01):
        """
        SP-BFGS: Preconditioned L-BFGS with Approximate Fubini-Study Metric.
        """
        super().__init__(model, history_size, tol)
        self.epsilon = spsa_epsilon # SPSA Perturbation size

    def _compute_fubini_scaling(self, params):
        """
        QN-SPSA (4-shot)를 사용하여 Local Curvature(Metric)를 추정.
        Diagonal Preconditioner로 사용할 스칼라 혹은 벡터를 반환.
        """
        dim = len(params)
        # 1. Random Direction (Bernoulli +/- 1)
        d = np.random.choice([-1, 1], size=dim)
        s = self.epsilon
        
        # 2. 4-Point Fidelity Measurement
        # (params를 중심으로 한 4개 지점에서의 Overlap 측정)
        # 위치: theta + 2sd, theta + sd, theta - sd, theta - 2sd
        # (여기서는 곡률 계산을 위해 2차 미분 근사식 사용)
        
        p_p = params + s * d
        p_m = params - s * d
        
        # Fidelity 측정 (중심 params와의 Overlap)
        f_p = self.model.get_fidelity(p_p, params)
        f_m = self.model.get_fidelity(p_m, params)
        
        # 3. Curvature Estimation (Metric Approximation)
        # Metric g ≈ -0.5 * d^2(F)/dtheta^2
        # Finite Difference: (2 - f_p - f_m) / s^2 (since f(0)=1)
        # 이 값은 "이 방향(d)으로의 저항"을 의미함.
        
        curvature_scalar = (2.0 - f_p - f_m) / (s**2 + 1e-10)
        curvature_scalar = np.abs(curvature_scalar) # Metric은 양수여야 함
        
        # Regularization (너무 작은 값 방지)
        curvature_scalar = max(curvature_scalar, 1e-3)
        
        # Scaling Factor = 1 / Metric (Hessian Inverse의 역할)
        # 곡률이 크면(Metric 大) 조금만 이동(Scaling 小)
        scaling_factor = 1.0 / curvature_scalar
        
        return scaling_factor

    def _get_direction(self, grad, scaling_factor=1.0):
        """
        L-BFGS Two-loop recursion with Fubini Preconditioning
        """
        q = grad.copy()
        alphas = []
        
        # Loop 1: Backward (Gradient -> Unscaled Search Direction)
        for s, y, rho in reversed(list(zip(self.s_history, self.y_history, self.rho_history))):
            alpha = rho * np.dot(s, q)
            q -= alpha * y
            alphas.append(alpha)
            
        # --- [Key Modification: Preconditioning] ---
        # H_0 = scaling_factor * I
        # Fubini Metric 정보를 반영하여 초기 Hessian 추정치를 조정
        r = scaling_factor * q 
        # -------------------------------------------
        
        # Loop 2: Forward (Recover Search Direction)
        for (s, y, rho), alpha in zip(list(zip(self.s_history, self.y_history, self.rho_history)), reversed(alphas)):
            beta = rho * np.dot(y, r)
            r += s * (alpha - beta)
            
        return -r # Descent direction

    def optimize(self, initial_params, max_iter=50):
        params = np.array(initial_params, dtype=float)
        
        # 초기 계산
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
                
            # 1. Compute Fubini Scaling (Cost: 2~4 shots)
            # 매 스텝 현재 위치의 곡률을 파악
            scaling = self._compute_fubini_scaling(params)
            
            # 2. Get Direction with Preconditioning
            p_k = self._get_direction(grad, scaling_factor=scaling)
            
            # 3. Line Search
            alpha, params_new, energy_new = self._line_search(params, p_k, energy, grad)
            
            # 4. Updates
            if energy_new < best_energy:
                best_energy = energy_new
                best_params = params_new.copy()
                print(f"  > New Best: {best_energy:.6f}")
                
            s_k = params_new - params
            grad_new = self.model.get_gradient(params_new)
            y_k = grad_new - grad
            
            if np.linalg.norm(s_k) > 1e-9 and np.dot(y_k, s_k) > 1e-10:
                rho = 1.0 / np.dot(y_k, s_k)
                self.s_history.append(s_k)
                self.y_history.append(y_k)
                self.rho_history.append(rho)
            
            params = params_new
            energy = energy_new
            grad = grad_new
            
            history.append((energy, grad_norm))
            print(f"Iter {k+1}: E={energy:.6f}, |g|={grad_norm:.4f}, Scale={scaling:.4f}")
            
        return best_params, history