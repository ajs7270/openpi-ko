
# Running openpi models remotely

We provide utilities for running openpi models remotely. This is useful for running inference on more powerful GPUs off-robot, and also helps keep the robot and policy environments separate (and e.g. avoid dependency hell with robot software).

## Starting a remote policy server

To start a remote policy server, you can simply run the following command:

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

The `env` argument specifies which $\pi_0$ checkpoint should be loaded. Under the hood, this script will execute a command like the following, which you can use to start a policy server, e.g. for checkpoints you trained yourself (here an example for the DROID environment):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

This will start a policy server that will serve the policy specified by the `config` and `dir` arguments. The policy will be served on the specified port (default: 8000).

## Querying the remote policy server from your robot code

We provide a client utility with minimal dependencies that you can easily embed into any robot codebase.

First, install the `openpi-client` package in your robot environment:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

Then, you can use the client to query the remote policy server from your robot code. Here's an example of how to do this:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

Here, the `host` and `port` arguments specify the IP address and port of the remote policy server. You can also specify these as command-line arguments to your robot code, or hard-code them in your robot codebase. The `observation` is a dictionary of observations and the prompt, following the specification of the policy inputs for the policy you are serving. We have concrete examples of how to construct this dictionary for different environments in the [simple client example](examples/simple_client/main.py).

---

# openpi 모델 원격 실행 (한국어)

openpi 모델을 원격으로 실행하기 위한 유틸리티를 제공합니다. 이는 로봇 외부의 더 강력한 GPU에서 추론을 실행하는 데 유용하며, 로봇과 정책 환경을 분리하고 (예: 로봇 소프트웨어와의 의존성 문제를 피하는 데) 도움이 됩니다.

## 원격 정책 서버 시작하기

원격 정책 서버를 시작하려면 다음 명령을 실행하면 됩니다:

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

`env` 인수는 로드할 $\pi_0$ 체크포인트를 지정합니다. 내부적으로 이 스크립트는 다음과 같은 명령을 실행하며, 이를 사용하여 직접 훈련한 체크포인트 등에 대한 정책 서버를 시작할 수 있습니다 (여기서는 DROID 환경 예시):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

이렇게 하면 `config` 및 `dir` 인수로 지정된 정책을 제공하는 정책 서버가 시작됩니다. 정책은 지정된 포트(기본값: 8000)에서 제공됩니다.

## 로봇 코드에서 원격 정책 서버 쿼리하기

모든 로봇 코드베이스에 쉽게 포함할 수 있는 최소한의 의존성을 가진 클라이언트 유틸리티를 제공합니다.

먼저 로봇 환경에 `openpi-client` 패키지를 설치합니다:

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

그런 다음 클라이언트를 사용하여 로봇 코드에서 원격 정책 서버를 쿼리할 수 있습니다. 다음은 그 예시입니다:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# 에피소드 루프 외부에서 정책 클라이언트를 초기화합니다.
# 정책 서버의 호스트와 포트를 가리킵니다 (localhost와 8000이 기본값).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # 에피소드 루프 내부에서 관측을 구성합니다.
    # 대역폭/지연 시간을 최소화하기 위해 클라이언트 측에서 이미지를 리사이즈합니다. 항상 이미지를 uint8 형식으로 반환합니다.
    # 훈련 루틴과 일치하도록 이미지 리사이즈 + uint8 변환 유틸리티를 제공합니다.
    # 사전 훈련된 pi0 모델의 일반적인 resize_size는 224입니다.
    # 고유 수용성 `state`는 정규화되지 않은 상태로 전달될 수 있으며, 정규화는 서버 측에서 처리됩니다.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # 현재 관측으로 정책 서버를 호출합니다.
    # 이는 (action_horizon, action_dim) 모양의 액션 청크를 반환합니다.
    # 일반적으로 N 스텝마다 정책을 호출하고 나머지 스텝에서는
    # 예측된 액션 청크의 스텝을 오픈 루프로 실행하면 됩니다.
    action_chunk = client.infer(observation)["actions"]

    # 환경에서 액션을 실행합니다.
    ...

```

여기서 `host` 및 `port` 인수는 원격 정책 서버의 IP 주소와 포트를 지정합니다. 이를 로봇 코드에 대한 명령줄 인수로 지정하거나 로봇 코드베이스에 하드코딩할 수도 있습니다. `observation`은 제공하는 정책의 정책 입력 사양에 따라 관측 및 프롬프트의 사전입니다. [간단한 클라이언트 예제](examples/simple_client/main.py)에서 다양한 환경에 대해 이 사전을 구성하는 방법에 대한 구체적인 예가 있습니다.
