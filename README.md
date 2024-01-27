# Federated_Learning_Simulator_with_LA_Multipath
## About The Project
이 프로젝트는 연합 학습과 다중 경로 통신을 결합하여 학습 성능을 향상시키는 것을 목표로 한다. 주변 기기로 명명된 Learning Agent를 도입하여 클라이언트보다 성능이 우수한 기기에서 학습을 진행하고, 다중 경로 통신을 통해 모델을 효율적으로 분산 전송한다. 자세한 내용은 논문 `Learning agent 활용 및 다중 경로 기반 모델 분할 전송을 통한 연합학습 개선`에서 확인할 수 있다. 
### Key Features
- Learning Agent: 클라이언트가 신뢰할 수 있는 주변 기기로 클라이언트 대신 학습을 진행하여 학습 성능을 향상할 수 있음
- Multipaths: 기존의 단일 경로 통신이 아닌 다중 경로 통신을 활용하여 모델을 효율적으로 분산 전송
## Overview
본 시뮬레이터는 Learning agent와 다중경로 모두 사용하지 않는 경우부터 모두 사용하는 경우까지의 4가지 상황에 대한 연합학습을 시뮬레이션 할 수 있다. 
## Requirements
Python 3.9 version

use `requirements.txt`
## Project Files Description 
1. python file
2. yaml
   - `device.yaml`
   - `communication.yaml`
   - `model.yaml`
   - `dataset.yaml`
   - `path_setting`
   - `custom_setting`
       - setting
       - server_spec
       - edge_device_spec
  
## Run
1. Single path
   ```
   python main.py
   ```
2. Multi paths
   ```
   python main.py --use_multipaths
   ```
## Results
### 1. Configuration situation
### 2. output
