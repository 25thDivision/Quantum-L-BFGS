import numpy as np
from collections import deque
from models.base_model import QuantumBase

class QLBFGS:
    def __init__(self, model: QuantumBase, history_size=10, tol=1e-4):
        """
        QITE-Inspired Geometry-Aware VQE Optimizer (SP-BFGS Hybrid).
        
        Args:
            model: QuantumBase 모델
            history_size: L-BFGS 메모리 크기 (m)
            tol: 수렴 기준 (Gradient Norm)
        """
        self.model = model
        self.m = history_size
        self.tol = tol
        
        # History Deques
        self.s_history = deque(maxlen=history_size)
        self.y_history = deque(maxlen=history_size)
        self.rho_history = deque(maxlen=history_size)
        
        # 초기 Hessian Scale (QN-SPSA로 설정 예정)
        self.H_diag_scale = 1.0

    def _spsa_curvature_probe(self, params, delta=0.01):
        """
        [Step 0] QN-SPSA 방식의 4-Shot 곡률 추정 (Warm Start).
        전체 Metric을 구하는 건 불가능하므로, 평균적인 Scaling Factor를 구함.
        """
        dim = len(params)
        
        # 1. 랜덤 섭동 벡터 (Bernoulli +/- 1)
        pert = np.random.choice([-1, 1], size=dim)
        
        # 2. 두 지점에서의 SPSA Gradient 근사 (각 2 shot, 총 4 shot)
        # 지점 A: params
        ep_p = self.model.get_energy(params + delta * pert)
        ep_m = self.model.get_energy(params - delta * pert)
        grad_spsa_1 = (ep_p - ep_m) / (2 * delta)
        
        # 지점 B: params + 0.1 * pert (조금 이동한 곳)
        step = 0.1
        params_shifted = params + step * pert
        ep_p2 = self.model.get_energy(params_shifted + delta * pert)
        ep_m2 = self.model.get_energy(params_shifted - delta * pert)
        grad_spsa_2 = (ep_p2 - ep_m2) / (2 * delta)
        
        # 3. 곡률(Hessian Diagonal Scale) 추정 (Secant equation 유사)
        # y = H * s  =>  (grad2 - grad1) ~ H * (step * pert)
        y_scalar = grad_spsa_2 - grad_spsa_1
        s_scalar = step # pert의 크기는 sqrt(dim)이지만 스칼라로 단순화
        
        # H ~ y / s
        if abs(s_scalar) > 1e-8 and abs(y_scalar) > 1e-8:
            curvature = abs(y_scalar / s_scalar)
            # Inverse Hessian Scale 이므로 1/curvature
            return 1.0 / (curvature + 1e-5)
        else:
            return 1.0 # 실패 시 기본값

    def _get_direction(self, grad):
        """
        L-BFGS Two-loop recursion으로 Search Direction (p_k = -B_k^{-1} g_k) 계산
        """
        q = grad.copy()
        alphas = []
        
        # Backward Pass
        for s, y, rho in reversed(list(zip(self.s_history, self.y_history, self.rho_history))):
            alpha = rho * np.dot(s, q)
            q -= alpha * y
            alphas.append(alpha)
            
        # Scaling (User's Hybrid Strategy: SPSA 초기값 or History 기반)
        if self.s_history:
            # L-BFGS 정석: 최근 스텝 정보로 스케일링
            last_s = self.s_history[-1]
            last_y = self.y_history[-1]
            gamma = np.dot(last_s, last_y) / (np.dot(last_y, last_y) + 1e-10)
        else:
            # 초기 스텝: SPSA로 구한 Warm Start 값 사용
            gamma = self.H_diag_scale
            
        r = gamma * q
        
        # Forward Pass
        for (s, y, rho), alpha in zip(list(zip(self.s_history, self.y_history, self.rho_history)), reversed(alphas)):
            beta = rho * np.dot(y, r)
            r += s * (alpha - beta)
            
        return -r # Descent direction

    def _line_search(self, params, direction, current_energy, current_grad):
        """
        Adaptive Step Size를 위한 Backtracking Line Search.
        (Armijo Condition: 충분히 에너지가 떨어지는가?)
        """
        alpha = 1.0 # 초기 보폭
        c1 = 1e-4   # 감소 조건 상수
        rho = 0.5   # 감쇠 비율
        max_ls_iter = 5 # 샷 절약을 위해 최대 5번만 시도
        
        # Directional derivative (기울기와 방향의 내적)
        dir_deriv = np.dot(current_grad, direction)
        
        # 만약 방향 자체가 상승 방향(오류)이라면 강제로 반전
        if dir_deriv > 0:
            direction = -direction
            dir_deriv = -dir_deriv
            
        for i in range(max_ls_iter):
            # 후보 위치
            cand_params = params + alpha * direction
            cand_energy = self.model.get_energy(cand_params) # 1 Shot Cost
            
            # Armijo Condition Check
            # E_new <= E_old + c1 * alpha * (grad * dir)
            if cand_energy <= current_energy + c1 * alpha * dir_deriv:
                return alpha, cand_params, cand_energy
            
            # 실패 시 보폭 반토막
            alpha *= rho
            
        # Line Search 실패 시: 아주 작은 보폭으로라도 이동하거나 제자리 유지
        # 여기서는 아주 작은 보폭으로 강제 이동 (Stochastic 특성 고려)
        return alpha, params + alpha * direction, self.model.get_energy(params + alpha * direction)

    def optimize(self, initial_params, max_iter=50):
        params = np.array(initial_params, dtype=float)
        
        # [Step 0] Warm Start: QN-SPSA로 초기 곡률 스케일 측정 (4 Shot)
        print("Initializing with QN-SPSA (4-shot probe)...")
        self.H_diag_scale = self._spsa_curvature_probe(params)
        print(f"Initial Metric Scale (Gamma): {self.H_diag_scale:.4f}")
        
        # 초기 에너지 및 기울기 (PSR 2n Shot)
        energy = self.model.get_energy(params)
        grad = self.model.get_gradient(params)
        
        print(f"Initial Energy: {energy:.6f}")
        history = [(energy, np.linalg.norm(grad))]
        
        best_energy = energy
        best_params = params.copy()
        
        for k in range(max_iter):
            # 1. 수렴 체크
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.tol:
                print(f"Converged at iter {k}")
                break
                
            # 2. 방향 결정 (SP-BFGS + QITE Metric)
            # p_k = - (M_approx)^(-1) * grad
            p_k = self._get_direction(grad)
            
            # 3. Line Search (Adaptive Step Size)
            # QITE의 고정된 dt 대신 최적의 alpha를 찾음
            alpha, params_new, energy_new = self._line_search(params, p_k, energy, grad)
            
            # 4. 업데이트 및 다음 스텝 준비
            s_k = params_new - params
            
            # 만약 이동이 거의 없으면 수렴으로 간주
            if np.linalg.norm(s_k) < 1e-8:
                print("Step size too small. Stopping.")
                break
                
            # 새로운 기울기 (비싼 연산)
            grad_new = self.model.get_gradient(params_new)
            y_k = grad_new - grad
            
            # 5. SP-BFGS History Update (Damping Strategy)
            # 곡률 조건 (Curvature Condition): s^T y > 0
            ys = np.dot(y_k, s_k)
            
            if ys > 1e-10:
                rho = 1.0 / ys
                self.s_history.append(s_k)
                self.y_history.append(y_k)
                self.rho_history.append(rho)
            else:
                # Damped update or Skip
                # 여기서는 노이즈 내성을 위해 과감히 Skip
                pass # print(f"Iter {k}: Skipped BFGS update (ys={ys:.2e})")

            # 변수 갱신
            params = params_new
            energy = energy_new
            grad = grad_new
            
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
                print(f"  > New Best Found: {best_energy:.6f}")
            
            history.append((energy, np.linalg.norm(grad)))
            print(f"Iter {k+1}: Energy={energy:.6f}, GradNorm={grad_norm:.4f}, Step={alpha:.4f}")
            
        return best_params, history