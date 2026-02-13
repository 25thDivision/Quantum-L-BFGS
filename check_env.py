import pennylane as qml
import numpy as np

try:
    # 1. PennyLane 버전 찍어보기
    print(f"PennyLane Version: {qml.__version__}")

    # 2. Qiskit Aer 백엔드 불러오기 테스트
    # 여기서 에러가 나면 pennylane-qiskit 설치가 잘못된 것
    dev = qml.device("qiskit.aer", wires=2)
    print("✅ 성공: Qiskit Aer 백엔드가 정상적으로 로드되었습니다.")

    # 3. 간단한 회로 실행 테스트
    @qml.qnode(dev)
    def circuit(theta):
        qml.RX(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    result = circuit(0.5)
    print(f"✅ 실행 테스트 완료: 결과값 = {result}")

except Exception as e:
    print("\n❌ 설치 오류 발생:")
    print(e)