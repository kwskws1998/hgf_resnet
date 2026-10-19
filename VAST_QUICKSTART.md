# Vast.ai에서 첫 실험 실행하기

준비할 머신은 **RTX 4090 1장, CPU 4 vCPU 이상, RAM 16GB 이상,
디스크 50GB 정도**입니다. 이는 이 프로젝트를 위한 권장 자원 설정이며,
실제 GPU 메모리와 시간은 실행 초기에 측정합니다. Vast.ai의 PyTorch
템플릿과 Python 3.10 이상을 사용하세요. 이 코드는 Linux/WSL에서
백그라운드 실행하도록 작성했습니다.

[Vast.ai 시작 안내](https://docs.vast.ai/guides/get-started/quickstart)

## 업로드와 설치·실행

최신 `hgf_resnet.zip`을 인스턴스의 `/workspace`에 업로드하고 압축을 풉니다.
SSH/SFTP 또는 템플릿에서 제공하는 파일 업로드 기능을 사용할 수 있습니다.

```console
python3 -m zipfile -e /workspace/hgf_resnet.zip /workspace
```

터미널의 작업 디렉터리를 `/workspace/hgf_resnet`으로 옮긴 다음, 아래
명령 **하나**를 실행합니다.

```console
python3 setup_environment.py --launch-pilot --run-output runs/first_pilot
```

이 명령은 프로젝트 안에 `.venv`를 만들고, 검증에 사용한 버전과 같은
PyTorch 2.12.1 / torchvision 0.27.1의 CUDA 12.6 빌드를 설치합니다.
프로젝트 테스트와 실제 CUDA 연산 검사를 통과한 경우에만 실험을 시작합니다.
드라이버나 운영체제 패키지를 설치하지는 않습니다.

[공식 PyTorch 설치 조합](https://pytorch.org/get-started/previous-versions/)

**처음 환경 설치는 터미널에서 진행됩니다.** 설치 후
`Background worker started`가 출력되면 실험이 별도 세션에서 실행 중입니다.
그 뒤에는 SSH 연결을 닫아도 실험이 계속됩니다. 인스턴스나 컨테이너 자체를
중지하면 계산은 중단되며, 아래 재개 명령을 사용해야 합니다.

Vast 템플릿의 기존 CUDA PyTorch 환경을 그대로 사용하려면 다음 명령을
대신 사용할 수 있습니다. 기존 버전을 유지하며 GPU 검사와 테스트는 동일하게
실행합니다.

```console
python3 setup_environment.py --use-existing-env --launch-pilot --run-output runs/first_pilot
```

이 옵션을 사용했을 때는 아래의 `.venv/bin/python` 대신 설치 때 사용한
`python3`를 사용하세요. 두 설치 명령을 연속해서 실행할 필요는 없습니다.

## 자리를 비운 동안 실행되는 최소 범위

1. 실제 batch 설정으로 ResNet 학습·adapter 업데이트가 GPU에서 되는지 검사.
2. CIFAR-10 다운로드와 분할 검증.
3. 초기 ResNet-18을 source 30,000장으로 **100 epochs** 학습.
4. 같은 source 체크포인트로 **1 seed, 방법당 200 updates, batch 8** 비교.

방법 순서는 frozen → AdamW → adaptive Q/R → hierarchical Q/R → zero-Q →
fixed-Q입니다. 먼저 핵심 비교 결과를 얻고 나머지 비교를 이어갑니다.
30회 반복 실험이나 하이퍼파라미터 탐색은 자동으로 시작하지 않습니다.

2시간은 강제 종료 시각이 아닙니다. 초기 학습이 오래 걸리면 그 이후에도
계속하며, 정해진 실험이 끝나면 프로세스가 종료됩니다. 시간 채우기용 대기나
무한 반복은 없습니다. 초기 환경 설치, 데이터 다운로드, CPU/디스크 속도에
따라 귀환 시점에 완료되는 범위가 달라집니다.

## 진행 상황 확인

```console
.venv/bin/python run_experiments.py --status --output runs/first_pilot
```

이 명령은 GPU 작업을 추가하지 않습니다. 현재 단계, 각 방법의 상태,
완료한 source epoch와 최근 validation 결과를 표시합니다.

- `runs/first_pilot/run.json`: 전체 실행 상태와 실제 명령.
- `runs/first_pilot/logs/`: 각 단계의 표준 출력·오류.
- `runs/first_pilot/preflight_attempt01/benchmark.json`: GPU 메모리와 속도.
- `runs/first_pilot/source_attempt01/source.pt`: validation 기준 최선의 모델.
- `runs/first_pilot/source_attempt01/last.pt`: 직전 완료 epoch의 전체 학습 상태.
- `runs/first_pilot/comparison.json`: 완료된 방법들의 비교 결과.

재시도 시 `attempt02`처럼 새 폴더가 생길 수 있으므로 실제 경로는
`run.json`을 기준으로 확인하세요. 비교 방법 하나가 실패해도 독립적인 다른
방법은 계속 실행합니다. GPU 검사나 초기 학습이 실패하면 이후 비교는
진행하지 않습니다. 전체 성공은 `run.json`의 `status: completed`로 확인합니다.

## 중단 후 이어서 실행

```console
.venv/bin/python run_experiments.py --resume --detach --output runs/first_pilot
```

동일한 환경과 코드에서 실행하세요. 완료된 단계는 결과 해시를 확인하고
건너뜁니다. 초기 모델은 마지막 **완료 epoch**부터 이어갑니다. 중간에 끊긴
epoch는 다시 실행합니다. AdamW/Kalman 적응 도중 끊긴 방법은 새로운
attempt 폴더에서 같은 source 모델로 처음부터 재실행합니다.

첫 epoch가 끝나기 전 중단되었다면 저장된 학습 경계가 없으므로 source도
새 attempt에서 다시 시작합니다. `source.pt`가 있다는 이유만으로 초기
학습 100 epochs가 완료된 것으로 처리하지 않습니다.

실행 중인 작업에 같은 명령을 다시 보내면 잠금으로 중복 GPU 작업을 막습니다.
코드·설정·라이브러리 버전 변경을 검사하며, source 학습을 재개할 때는
데이터가 같은지도 확인합니다.

## 결과 회수와 대여 종료

최소한 `runs/first_pilot`, `environment_setup.json`,
`environment_freeze.txt`를 다운로드하세요. 코드 종료는 Vast.ai 인스턴스
정지와 별개입니다. 결과를 확인한 뒤 Vast에서 **Stop**하면 GPU 과금이
멈추고 저장공간 비용은 계속됩니다. **Destroy**는 데이터를 삭제하므로
결과를 먼저 회수해야 합니다.

[Vast.ai 인스턴스 관리](https://docs.vast.ai/guides/instances/manage-instances)

## 현재 검증 범위

환경 설치의 dry run, CPU 통합 테스트, 실제 ResNet을 사용하는 CPU 사전
검사를 확인했습니다. 중단·재개한 CPU 학습 결과도 중단 없는 결과와 일치합니다.
이 문서를 준비한 로컬 환경에는 CUDA가 없으므로, Vast.ai 설치·GPU 학습·
24GB 메모리 사용량은 대여 머신에서 위 명령이 실행되어야 확인됩니다.
