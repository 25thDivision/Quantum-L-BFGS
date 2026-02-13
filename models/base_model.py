from abc import ABC, abstractmethod
import numpy as np

class QuantumBase(ABC):
    """
    모든 양자 모델(PennyLane, Qiskit 등)이 반드시 지켜야 할 규격(Interface)입니다.
    최적화기는 이 클래스의 메서드만 호출합니다.
    """

    @abstractmethod
    def get_energy(self, params: np.ndarray) -> float:
        """
        주어진 파라미터(params)에 대한 에너지(Cost Function 값)를 반환합니다.
        
        Args:
            params (np.ndarray): 최적화할 파라미터 벡터 (1D array)
            
        Returns:
            float: 측정된 에너지 값 (예: <H>)
        """
        pass

    @abstractmethod
    def get_gradient(self, params: np.ndarray) -> np.ndarray:
        """
        주어진 파라미터에 대한 기울기(Gradient) 벡터를 반환합니다.
        반드시 Parameter Shift Rule (또는 이에 준하는 하드웨어 호환 방식)을 사용해야 합니다.
        
        Returns:
            np.ndarray: 기울기 벡터 (1D array, params와 같은 크기)
        """
        pass

    @abstractmethod
    def get_metric_diag(self, params: np.ndarray) -> np.ndarray:
        """
        (선택 사항) 파라미터 공간의 Metric(곡률) 대각 성분을 반환합니다.
        SP-BFGS의 초기 Hessian(H0) 설정에 사용됩니다. (QN-SPSA 활용 등)
        """
        pass