# Run Aloha Sim

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM
```

---

# Aloha Sim 실행 (한국어)

## 도커 사용

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## 도커 미사용

터미널 창 1:

```bash
# 가상 환경 생성
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# 시뮬레이션 실행
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

참고: EGL 오류가 발생하는 경우 다음 종속성을 설치해야 할 수 있습니다:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

터미널 창 2:

```bash
# 서버 실행
uv run scripts/serve_policy.py --env ALOHA_SIM
```
