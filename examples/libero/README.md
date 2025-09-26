# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

---

# LIBERO 벤치마크 (한국어)

이 예제는 LIBERO 벤치마크를 실행합니다: https://github.com/Lifelong-Robot-Learning/LIBERO

참고: 이 디렉토리의 requirements.txt를 업데이트할 때, `uv pip compile` 명령어에 `--extra-index-url https://download.pytorch.org/whl/cu113` 플래그를 추가해야 합니다.

이 예제는 git 서브모듈이 초기화되어 있어야 합니다. 다음을 실행하는 것을 잊지 마세요:

```bash
git submodule update --init --recursive
```

## 도커 사용 (권장)

```bash
# X11 서버에 대한 접근 권한 부여:
sudo xhost +local:docker

# 기본 체크포인트와 태스크 스위트로 실행:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# Mujoco에 glx를 대신 사용하려면 (egl 오류가 있는 경우 사용):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

추가 `SERVER_ARGS`를 제공하여 로드된 체크포인트를 사용자 정의할 수 있고 (`scripts/serve_policy.py` 참조), 추가 `CLIENT_ARGS`를 제공하여 LIBERO 태스크 스위트를 사용자 정의할 수 있습니다 (`examples/libero/main.py` 참조).
예를 들어:

```bash
# 사용자 정의 체크포인트를 로드하려면 (최상위 openpi/ 디렉토리에 위치):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# libero_10 태스크 스위트를 실행하려면:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## 도커 미사용 (권장하지 않음)

터미널 창 1:

```bash
# 가상 환경 생성
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 시뮬레이션 실행
python examples/libero/main.py

# Mujoco에 glx를 대신 사용하려면 (egl 오류가 있는 경우 사용):
MUJOCO_GL=glx python examples/libero/main.py
```

터미널 창 2:

```bash
# 서버 실행
uv run scripts/serve_policy.py --env LIBERO
```

## 결과

다음 수치를 재현하고 싶다면, `gs://openpi-assets/checkpoints/pi05_libero/`에 있는 체크포인트를 평가할 수 있습니다. 이 체크포인트는 `pi05_libero` 설정으로 openpi에서 훈련되었습니다.

| 모델 | Libero Spatial | Libero Object | Libero Goal | Libero 10 | 평균 |
|---|---|---|---|---|---|
| π0.5 @ 30k (미세 조정) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |
