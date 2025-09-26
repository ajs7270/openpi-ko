### Docker Setup

All of the examples in this repo provide instructions for being run normally, and also using Docker. Although not required, the Docker option is recommended as this will simplify software installation, produce a more stable environment, and also allow you to avoid installing ROS and cluttering your machine, for examples which depend on ROS.

- Basic Docker installation instructions are [here](https://docs.docker.com/engine/install/).
- Docker must be installed in [rootless mode](https://docs.docker.com/engine/security/rootless/).
- To use your GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- The version of docker installed with `snap` is incompatible with the NVIDIA container toolkit, preventing it from accessing `libnvidia-ml.so` ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)). The snap version can be uninstalled with `sudo snap remove docker`.
- Docker Desktop is also incompatible with the NVIDIA runtime ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)). Docker Desktop can be uninstalled with `sudo apt remove docker-desktop`.


If starting from scratch and your host machine is Ubuntu 22.04, you can use accomplish all of the above with the convenience scripts `scripts/docker/install_docker_ubuntu22.sh` and `scripts/docker/install_nvidia_container_toolkit.sh`.

Build the Docker image and start the container with the following command:
```bash
docker compose -f scripts/docker/compose.yml up --build
```

To build and run the Docker image for a specific example, use the following command:
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
where `<example_name>` is the name of the example you want to run.

During the first run of any example, Docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.

---

### 도커 설정 (한국어)

이 저장소의 모든 예제는 일반적인 방식과 도커를 사용하는 방식 모두에 대한 실행 지침을 제공합니다. 필수는 아니지만, 도커 옵션은 소프트웨어 설치를 단순화하고, 더 안정적인 환경을 만들며, ROS에 의존하는 예제의 경우 ROS를 설치하고 기계를 복잡하게 만드는 것을 피할 수 있으므로 권장됩니다.

- 기본 도커 설치 지침은 [여기](https://docs.docker.com/engine/install/)에 있습니다.
- 도커는 [rootless 모드](https://docs.docker.com/engine/security/rootless/)로 설치해야 합니다.
- GPU를 사용하려면 [NVIDIA 컨테이너 툴킷](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)도 설치해야 합니다.
- `snap`으로 설치된 도커 버전은 NVIDIA 컨테이너 툴킷과 호환되지 않아 `libnvidia-ml.so`에 접근할 수 없습니다 ([이슈](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)). snap 버전은 `sudo snap remove docker`로 제거할 수 있습니다.
- Docker Desktop 또한 NVIDIA 런타임과 호환되지 않습니다 ([이슈](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)). Docker Desktop은 `sudo apt remove docker-desktop`으로 제거할 수 있습니다.

처음부터 시작하고 호스트 머신이 Ubuntu 22.04인 경우, 편의 스크립트인 `scripts/docker/install_docker_ubuntu22.sh`와 `scripts/docker/install_nvidia_container_toolkit.sh`를 사용하여 위의 모든 작업을 수행할 수 있습니다.

다음 명령으로 도커 이미지를 빌드하고 컨테이너를 시작하세요:
```bash
docker compose -f scripts/docker/compose.yml up --build
```

특정 예제에 대한 도커 이미지를 빌드하고 실행하려면 다음 명령을 사용하세요:
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
여기서 `<example_name>`은 실행하려는 예제의 이름입니다.

어떤 예제든 처음 실행하는 동안 도커는 이미지를 빌드합니다. 이 작업이 진행되는 동안 커피 한 잔의 여유를 가지세요. 이미지가 캐시되므로 후속 실행은 더 빠를 것입니다.