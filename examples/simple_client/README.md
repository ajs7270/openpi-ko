# Simple Client

A minimal client that sends observations to the server and prints the inference rate.

You can specify which runtime environment to use using the `--env` flag. You can see the available options by running:

```bash
uv run examples/simple_client/main.py --help
```

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
uv run examples/simple_client/main.py --env DROID
```

Terminal window 2:

```bash
uv run scripts/serve_policy.py --env DROID
```

---

# 간단한 클라이언트 (한국어)

서버에 관측치를 보내고 추론 속도를 출력하는 최소한의 클라이언트입니다.

`--env` 플래그를 사용하여 사용할 런타임 환경을 지정할 수 있습니다. 사용 가능한 옵션은 다음을 실행하여 볼 수 있습니다:

```bash
uv run examples/simple_client/main.py --help
```

## 도커 사용

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## 도커 미사용

터미널 창 1:

```bash
uv run examples/simple_client/main.py --env DROID
```

터미널 창 2:

```bash
uv run scripts/serve_policy.py --env DROID
```
