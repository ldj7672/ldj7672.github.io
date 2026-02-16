---
title: "GPU 클러스터 통신: NVLink, InfiniBand, NCCL,..."
date: 2025-10-01
categories:
  - AI Reports
tags:
  - GPU Cluster
  - ML Engineering
  - Distributed Training
  - NCCL
  - InfiniBand
  - NVLink
category: engineering
---

대규모 모델 학습에서 'GPU가 충분히 빠른데 왜 학습이 안 빨라지지?'라는 질문의 답은 대부분 **통신**에 있다. 최신 GPU(H100, A100)는 TFLOPS 수준에서 압도적이지만, 분산 학습의 실질적인 throughput은 **GPU 간 데이터 이동 효율**에 의해 결정된다. 이 글은 GPU 클러스터 통신을 구성하는 핵심 인프라를 ML Engineer 관점에서 다뤄보려 한다.

## 핵심 목표

- **하드웨어 토폴로지**: NVLink, NVSwitch, PCIe의 물리적 구조와 대역폭 특성
- **네트워크 계층**: InfiniBand vs Ethernet, RDMA, GPUDirect의 동작 원리와 성능 차이
- **통신 라이브러리**: NCCL의 내부 알고리즘과 최적화 전략
- **실무 도구**: 벤치마킹, 프로파일링, 트러블슈팅 방법론
- **성능 최적화**: 토폴로지별 병렬화 전략과 통신 패턴 설계

이 글은 단순한 개념 설명이 아니라, **실제 클러스터 환경에서 통신 병목을 진단하고 최적화할 수 있는 엔지니어링 지식**을 제공하는 것을 목표로 한다.


## 1. 통신 계층 구조: Intra-node vs Inter-node

GPU 클러스터 통신을 이해하는 첫 번째 원칙은 **물리적 범위(scope)에 따른 계층 구분**이다.

### 1.1 Intra-node Communication (노드 내부)

- **정의**: 단일 서버 내 여러 GPU 간 통신
- **인터커넥트**: NVLink, NVSwitch, PCIe
- **대역폭**: 300-900 GB/s (NVLink 기준)
- **지연**: ~1-5 μs
- **사용 사례**: Tensor Parallelism, 노드 내 데이터 병렬

### 1.2 Inter-node Communication (노드 간)

- **정의**: 서로 다른 서버 간 통신
- **네트워크**: InfiniBand, Ethernet (network fabric)
- **대역폭**: 200-400 Gb/s (InfiniBand HDR/NDR)
- **지연**: ~2-10 μs
- **사용 사례**: Data Parallelism, Pipeline Parallelism

### 1.3 왜 이 구분이 중요한가

**대역폭 비율** (NVIDIA 및 Mellanox 공식 스펙 기준)
- NVLink (노드 내): 600 GB/s (NVLink 3.0, A100 기준, NVIDIA A100 Technical Overview)
- InfiniBand HDR (노드 간): 200 Gb/s = 25 GB/s (Mellanox HDR 스펙)
- **비율**: 24:1 차이

> **출처**: NVIDIA A100 Tensor Core GPU Architecture Whitepaper, Mellanox InfiniBand HDR 제품 사양서

**실무 영향**
- Tensor Parallelism처럼 통신이 **매우 빈번한** 병렬화는 노드 내부에 제한
- Data Parallelism처럼 **주기적이지만 덜 빈번한** 통신은 노드 간 분산 가능
- Pipeline Parallelism은 point-to-point 통신이므로 노드 간에서도 효율적


## 2. NVLink & NVSwitch: Intra-node Interconnect 상세

### 2.1 NVLink 기술 개요

**NVLink**는 NVIDIA GPU 간 직접 연결을 위한 고대역폭 인터커넥트로, PCIe의 병목을 우회한다.

#### 세대별 대역폭 발전

| 세대 | GPU | 링크당 대역폭 | 총 대역폭 (양방향) | 지연 |
|------|-----|---------------|-------------------|------|
| NVLink 1.0 | P100 | 20 GB/s | 160 GB/s | ~3 μs |
| NVLink 2.0 | V100 | 25 GB/s | 300 GB/s | ~2 μs |
| NVLink 3.0 | A100 | 25 GB/s | 600 GB/s (12 links) | ~1.5 μs |
| NVLink 4.0 | H100 | 50 GB/s | 900 GB/s (18 links) | ~1 μs |

**PCIe 대비 장점**
- PCIe 4.0 x16: ~32 GB/s (양방향)
- NVLink 4.0: **900 GB/s** (28배 차이)

### 2.2 NVSwitch: NVLink의 물리적 한계를 극복하는 스위치 패브릭

**NVLink만으로는 불충분한 이유**

NVLink는 강력하지만, GPU 수가 많아지면 **물리적 연결 한계**에 부딪힌다.

**문제 1: 링크 수 부족**
- 8 GPU를 모두 직접(pairwise) 연결하려면: C(8,2) = **28개 링크** 필요
- 하지만 A100 GPU는 **12개 NVLink 포트**만 제공
- 결과: 일부 GPU 쌍은 직접 연결 불가 → 중간 GPU를 경유해야 함

**문제 2: 비대칭 토폴로지**
- 직접 연결된 GPU 쌍: 1-hop (빠름)
- 간접 연결된 GPU 쌍: 2-hop 이상 (느림)
- Collective communication(AllReduce 등)에서 **경로 길이 불균등** → 성능 저하

**NVSwitch: 필수 솔루션**

NVSwitch는 이런 물리적 한계를 해결하기 위한 **외부 스위치 패브릭**이다.
- 각 GPU는 자신의 12개(또는 18개) NVLink를 모두 NVSwitch에 연결
- NVSwitch가 **크로스바 스위치(crossbar switch)** 역할을 수행
- 결과: 모든 GPU 쌍이 **1-hop으로 균등하게 통신** (full mesh topology)
- **비대칭 제거** → collective communication 성능 극대화

**정리**: NVLink는 고속 링크, NVSwitch는 이 링크들을 **최적 토폴로지로 구성하는 필수 인프라**다.

#### NVSwitch 세대

| 세대 | 포트 수 | 포트당 대역폭 | 총 처리량 | 사용 시스템 |
|------|---------|--------------|-----------|------------|
| NVSwitch 1.0 | 18 | 25 GB/s | 900 GB/s | DGX A100 |
| NVSwitch 2.0 | 18 | 50 GB/s | 1.8 TB/s | DGX H100 |
| NVSwitch 3.0 | 64 | 50 GB/s | 14.4 TB/s | DGX B200 (예상) |

**용어 설명: DGX vs HGX**

- **DGX (Deep Learning Supercomputer)**
  - NVIDIA의 **완제품 서버 시스템** (턴키 솔루션)
  - GPU + NVSwitch + 네트워킹 + 스토리지를 통합한 엔터프라이즈급 제품
  - 예: DGX A100 (8×A100 + 6×NVSwitch), DGX H100 (8×H100 + 4×NVSwitch)
  - 특징: NVIDIA가 직접 판매, 소프트웨어 스택 최적화, 엔터프라이즈 지원

- **HGX (High-Performance GPU eXtension)**
  - GPU + NVSwitch가 통합된 **베이스보드/모듈** (OEM용)
  - Dell, HPE, Lenovo 같은 서버 제조사가 자사 서버에 탑재
  - 예: HGX A100 (4 또는 8 GPU), HGX H100
  - 특징: 서버 벤더가 자체 시스템에 통합, 커스터마이징 가능

**DGX H100 예시**
- 8×H100 GPU
- 4×NVSwitch 2.0
- 각 GPU는 18개 NVLink 4.0 링크 모두 사용
- **All-to-all 대역폭**: 7.2 TB/s (aggregate)

### 2.3 PCIe: 범용 I/O 인터커넥트

**PCIe (PCI Express)란?**

PCIe는 **범용 고속 I/O 버스**로, CPU와 주변 장치(GPU, NIC, SSD 등)를 연결하는 표준 인터페이스다.
- **역할**: CPU ↔ GPU, GPU ↔ NIC, GPU ↔ 스토리지 연결
- **특징**: 직렬 point-to-point 연결, 레인(lane) 단위로 확장 가능
- **세대별 대역폭** (x16 기준, 양방향):
  - PCIe 3.0: ~32 GB/s
  - PCIe 4.0: ~64 GB/s
  - PCIe 5.0: ~128 GB/s

**NVLink가 있어도 PCIe는 필수**

NVLink는 GPU 간 통신에 특화되어 있지만, PCIe는 여전히 필요하다.

**PCIe의 역할**
- **CPU ↔ GPU 통신**: 호스트 메모리 접근, 커널 런치, 제어 명령
- **NIC ↔ GPU 통신**: GPUDirect RDMA가 없는 경우 NIC가 PCIe를 통해 GPU 메모리 접근
- **I/O 장치 연결**: 스토리지(NVMe SSD), 네트워크 카드, 기타 주변 장치

**PCIe 토폴로지 고려사항**
- **NUMA affinity**: GPU가 연결된 PCIe root complex가 어느 CPU 소켓에 속하는지
- **PLX switch**: 여러 GPU가 PCIe 스위치를 공유하면 대역폭 경쟁
- **P2P 경로**: GPU 간 PCIe peer-to-peer 가능 여부 (`nvidia-smi topo -m`으로 확인)


## 3. InfiniBand & Network Topology: Inter-node Interconnect

### 3.1 InfiniBand

대부분의 large-scale 모델 학습(수백~수천 GPU)은 InfiniBand 클러스터에서 수행된다. 대규모 학습에서는 **통신 오버헤드가 학습 시간을 지배**하기 때문이다. 수백 GPU가 동시에 gradient를 동기화할 때, 네트워크 지연과 대역폭이 병목이 되면 GPU는 계산을 끝내고도 통신을 기다리며 idle 상태가 된다.

**InfiniBand의 핵심 장점**
- **저지연**: 1-2 μs (Ethernet/RoCE 5-10 μs 대비 3-5배 빠름)
- **안정적 성능**: 하드웨어 레벨 혼잡 제어, jitter 최소화
- **RDMA 네이티브**: CPU 오버헤드 거의 없음, 커널 우회
- **검증된 생태계**: NCCL, MPI 등 수년간 최적화 축적

**InfiniBand 세대**

| 세대 | 대역폭 | 지연 | 주요 사용처 |
|------|--------|------|-----------|
| FDR | 56 Gb/s | ~2 μs | 구형 HPC |
| EDR | 100 Gb/s | ~1.5 μs | V100 클러스터 |
| HDR | 200 Gb/s | ~1 μs | A100 클러스터 |
| NDR | 400 Gb/s | ~0.6 μs | H100 클러스터 |
| XDR (예정) | 800 Gb/s | - | 차세대 |

**Ethernet (RoCE)의 한계**

- **지연**: 5-10 μs (RoCE v2) → InfiniBand 대비 3-5배 느림
- **안정성**: PFC 미세 튜닝 필요, 설정 복잡
- **Jitter**: 트래픽 혼재 시 성능 변동 심함
- **대규모 확장 어려움**: 수백 노드 이상에서 성능 예측 불안정

### 3.2 Network Topology: Fat-Tree

Fat-Tree는 대규모 GPU 클러스터에서 가장 널리 사용되는 네트워크 토폴로지다.

**구조 설명**
- **Leaf (ToR - Top of Rack)**: 각 랙의 GPU 노드들을 직접 연결
- **Spine (Aggregation)**: 여러 Leaf 스위치를 연결
- **Core**: 여러 Spine 스위치를 연결 (매우 큰 클러스터에만 필요)

**특징**
- **Oversubscription ratio**: 하위 링크 대역폭 ÷ 상위 링크 대역폭
- **이상적**: 1:1 (non-blocking) → 비용 높음
- **현실적**: 2:1 ~ 4:1 oversubscription
- **영향**: 동일 rack 내부 통신은 빠르지만, cross-rack은 느려짐

**예시: 128 노드 클러스터 (1024 GPUs)**
- Leaf: 16대 (각 8 노드 연결)
- Spine: 8대
- Oversubscription: 2:1 (Leaf→Spine 대역폭이 Leaf→Node의 1/2)


## 4. RDMA & GPUDirect: 통신 최적화의 핵심

### 4.1 RDMA (Remote Direct Memory Access)

**RDMA**란 NIC가 CPU를 거치지 않고 메모리에 직접 읽고 쓸 수 있는 기술이다. 일반적인 네트워크 통신에서는 CPU가 모든 데이터를 복사하고 처리한다 (GPU Memory → CPU Memory → Kernel buffer → NIC). 이 과정에서 CPU 오버헤드와 메모리 복사 비용이 크다.

RDMA는 **NIC가 메모리에 직접 접근**하여 CPU를 완전히 우회한다. 네트워크 카드가 메모리와 직접 통신하므로 CPU는 통신 과정에 관여하지 않는다.

**핵심 효과**
- **지연 감소**: 1-2 μs (일반 통신 대비 50-70% 단축)
- **CPU 오버헤드 제거**: CPU는 통신에 관여하지 않음
- **대역폭 효율**: 메모리 복사 없음, 버스 경합 최소화

InfiniBand는 네이티브 RDMA를 지원하며, Ethernet은 RoCE (RDMA over Converged Ethernet)를 통해 구현한다.

### 4.2 GPUDirect RDMA

**GPUDirect RDMA**는 NIC가 GPU 메모리에 직접 접근하여 데이터를 읽고 쓸 수 있는 기술이다. 일반적인 RDMA는 `GPU Memory → CPU Memory → NIC`로 CPU 메모리를 거쳐야 한다. GPUDirect RDMA는 **NIC가 GPU 메모리에 직접 접근**하여 (`GPU Memory → NIC`) CPU 메모리를 완전히 우회한다.

**효과**
- **zero-copy**: CPU 메모리 복사 완전 제거
- **지연 50-80% 단축**: 대규모 학습에서 체감 가능
- **메모리 대역폭 절약**: PCIe 버스 효율 향상


## 5. NCCL: Collective Communication Library

### 5.1 NCCL이란

**NCCL (NVIDIA Collective Communications Library)**: GPU 간 집단 통신을 최적화하는 라이브러리로, PyTorch DDP, FSDP, DeepSpeed 등 분산 학습 프레임워크가 내부적으로 사용한다.

분산 학습에서 수백 GPU가 동시에 gradient를 동기화하거나 파라미터를 교환할 때, NCCL은 하드웨어 토폴로지(NVLink, InfiniBand 등)를 자동으로 감지하고 최적의 통신 경로와 알고리즘을 선택한다. 즉, **"GPU들이 어떻게 연결되어 있는지"를 파악하고 "가장 빠른 통신 방법"을 자동으로 찾아준다**.

**핵심 기능**
- **Topology-aware**: NVLink, PCIe, InfiniBand 구조를 자동 감지하고 최적 경로 계산
- **알고리즘 자동 선택**: Ring (대역폭 우선), Tree (지연 우선) 등을 메시지 크기에 따라 선택
- **통신 패턴 최적화**: AllReduce, AllGather, ReduceScatter 등 분산 학습 핵심 연산 지원

### 5.2 주요 통신 패턴 (Collective Operations)

**AllReduce**: 모든 GPU의 텐서를 합산/평균하여 동일한 결과를 모두에게 분배
```
GPU 0: [1, 2, 3]     →  [10, 14, 18] (sum)
GPU 1: [4, 5, 6]     →  [10, 14, 18]
GPU 2: [5, 7, 9]     →  [10, 14, 18]
```
- Data Parallelism의 gradient 동기화에 사용
- 통신량: `2 × message_size` (Ring 알고리즘)

**ReduceScatter**: AllReduce의 전반부, 합산 후 분산
```
GPU 0: [1, 2, 3, 4]  →  [10] (chunk 0의 합)
GPU 1: [4, 5, 6, 7]  →  [14] (chunk 1의 합)
```
- ZeRO, FSDP의 gradient 동기화에 사용
- 통신량: `message_size`

**AllGather**: 각 GPU의 조각을 모아서 전체를 모두에게 분배
```
GPU 0: [10]  →  [10, 14, 18, 20]
GPU 1: [14]  →  [10, 14, 18, 20]
```
- ZeRO, FSDP의 파라미터 재구성에 사용
- 통신량: `message_size`

**Broadcast**: 한 GPU의 데이터를 모든 GPU에 전송
- 옵티마이저 상태 공유, 체크포인트 로딩에 사용

### 5.3 NCCL 성능 튜닝

**중요 환경 변수**

```bash
# 기본 설정
export NCCL_DEBUG=INFO              # 디버그 로그
export NCCL_DEBUG_SUBSYS=ALL        # 상세 서브시스템 로그

# 토폴로지 최적화
export NCCL_IB_HCA=mlx5_0,mlx5_1    # 사용할 IB 어댑터 지정
export NCCL_IB_GID_INDEX=3          # RoCE v2 사용 시
export NCCL_NET_GDR_LEVEL=5         # GPUDirect RDMA 강제 활성화
export NCCL_P2P_LEVEL=SYS           # P2P 통신 레벨 (NVL, PIX, SYS)

# 알고리즘 선택
export NCCL_ALGO=Tree,Ring          # 알고리즘 우선순위
export NCCL_PROTO=LL,LL128,Simple   # 프로토콜 우선순위

# 네트워크 최적화
export NCCL_IB_TIMEOUT=22           # IB 타임아웃 증가 (불안정한 네트워크)
export NCCL_IB_RETRY_CNT=7          # 재시도 횟수

# 멀티 NIC 환경
export NCCL_SOCKET_IFNAME=ib0,ib1   # 네트워크 인터페이스 지정
export NCCL_IB_DISABLE=0            # InfiniBand 활성화
export NCCL_NET="IB"                # 네트워크 백엔드

# SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)
export NCCL_COLLNET_ENABLE=1        # 스위치 내 aggregation
```

**벤치마킹: nccl-tests**

nccl-tests는 GPU 클러스터의 통신 성능을 순수하게 측정하는 벤치마크 도구다. 주로 **새 클러스터 구축 시 네트워크가 제대로 설정되었는지 검증**하거나, **학습이 예상보다 느릴 때 통신 병목을 확인**하기 위해 사용한다. 네트워크나 드라이버 업데이트 후 성능 회귀가 없는지 확인하는 용도로도 쓰인다. 

**실행 방법** (멀티 노드 환경 필요)
```bash
# 설치
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# AllReduce 벤치마크 (2개 노드, 각 8 GPU 예시)
mpirun -np 16 --npernode 8 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_IB_HCA=mlx5_0,mlx5_1 \
  ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1

# 결과 해석
#       size         count      type   redop    root     time   algbw   busbw
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)
    8388608       2097152     float     sum      -1   1242.3    6.75   12.66
   16777216       4194304     float     sum      -1   2156.4    7.78   14.59
```
결과에서 **algbw**는 실제 전송된 데이터 기준 대역폭이고, **busbw**는 링크 활용 기준 대역폭으로 하드웨어 스펙과 비교한다. 목표는 busbw가 하드웨어 대역폭의 80-90%에 도달하는 것이다 (예: InfiniBand HDR 200 Gb/s → 목표 160-180 Gb/s).

**프로파일링**

프로파일링은 실제 학습 코드에서 통신 병목을 진단할 때 사용한다. GPU utilization은 높은데 throughput이 낮거나, 노드 수를 늘려도 성능이 선형으로 증가하지 않을 때 통신이 병목인지 확인하기 위해 실행한다.

*Nsight Systems* (전체 시스템 프로파일)
```bash
nsys profile -t cuda,nvtx,mpi --mpi-impl=openmpi \
  -o profile.qdrep \
  python -m torch.distributed.launch --nproc_per_node=8 train.py
```
GPU idle 구간을 확인하여 통신 대기 시간을 측정하고, computation과 communication의 중첩 비율을 분석한다.

*PyTorch Profiler* (Python 레벨 프로파일)
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:  # warm-up
            prof.step()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if step >= 10:
            break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```
`ncclKernel*` 패턴의 시간이 전체에서 차지하는 비율을 확인한다. 통신 시간이 10-20% 이상이면 병목을 의심해야 한다.


## 7. 실무 트러블슈팅

### 7.1 통신 병목 진단

분산 학습에서 GPU utilization이 높은데도 throughput이 낮다면, 대부분 통신 또는 I/O가 병목이다. 통신 병목을 진단하는 가장 직접적인 방법은 **NCCL 로그를 활성화**하여 GPU가 실제로 어떤 통신 패턴을 사용하는지 확인하는 것이다.

```bash
# NCCL 디버그 로그 활성화
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV python train.py 2>&1 | grep "Ring\|Tree\|Using"
```

로그에서 "Using GPUDirect RDMA"가 보이지 않으면 통신 성능이 크게 저하되므로 인프라 팀과 확인이 필요하다. 또한 Ring 알고리즘이 아닌 Tree 알고리즘이 큰 메시지에 사용되고 있다면 토폴로지 인식에 문제가 있을 수 있다.

### 7.2 노드 확장 시 성능 저하

노드 수를 늘렸는데 throughput이 선형으로 증가하지 않는다면 **노드 간 통신이 병목**이다. 이때는 nccl-tests로 순수 통신 성능을 측정하여 네트워크 인프라에 문제가 있는지 확인한다.

```bash
# 실제 AllReduce 대역폭 측정
mpirun -np 16 ./all_reduce_perf -b 8 -e 2G -f 2 -g 1
```

busbw가 하드웨어 대역폭의 80% 이하라면 네트워크 설정, oversubscription, 또는 다른 작업과의 간섭을 의심해야 한다. InfiniBand 환경이라면 `ibstat`와 `ibdiagnet`으로 fabric 상태를 확인할 수 있다.

### 7.3 Straggler 노드 디버깅

특정 노드만 느려서 전체 학습이 지연되는 경우(straggler)가 있다. 이는 **하드웨어 이상**(GPU 성능 저하, NIC 문제, 케이블 불량)이거나 **토폴로지 불균형** 때문일 수 있다. 간단한 Python 스크립트로 각 GPU의 AllReduce 성능을 직접 측정할 수 있다.

```python
import torch
import torch.distributed as dist
dist.init_process_group('nccl')
rank = dist.get_rank()
x = torch.randn(100_000_000, device='cuda')
import time
t0 = time.time()
for _ in range(100):
    dist.all_reduce(x)
    torch.cuda.synchronize()
print(f'Rank {rank}: {time.time()-t0:.2f}s')
```

모든 rank의 시간이 비슷해야 하는데, 특정 rank만 느리다면 `nvidia-smi -q -d PERFORMANCE,CLOCK`로 GPU 상태를 확인하고 인프라 팀과 협업하여 하드웨어를 점검해야 한다.

### 7.4 GPUDirect RDMA 확인

최신 클러스터는 대부분 GPUDirect RDMA가 활성화되어 있지만, 가끔 커널 모듈이나 설정 문제로 비활성화되는 경우가 있다. 이를 확인하는 방법은 다음과 같다.

```bash
# 1. 커널 모듈 확인
lsmod | grep nvidia_peermem

# 2. NCCL 로그에서 "Using GPUDirect RDMA" 확인
NCCL_DEBUG=INFO python train.py 2>&1 | grep "GPUDirect"

# 3. nvidia-smi topo로 NIC-GPU 연결 확인
nvidia-smi topo -m
# NIC와 GPU 간에 "SYS" 대신 "NODE" 또는 "PHB"가 표시되어야 함
```

만약 GPUDirect가 비활성화되어 있다면 통신 성능이 50% 이상 저하될 수 있으므로 반드시 수정해야 한다.


## 참고 자료

**NVIDIA 공식 문서**
- NVIDIA NVLink and NVSwitch  
  https://www.nvidia.com/en-us/data-center/nvlink/
- NVIDIA NCCL User Guide  
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- GPUDirect RDMA Documentation  
  https://docs.nvidia.com/cuda/gpudirect-rdma/

**Networking**
- NVIDIA Networking Documentation (InfiniBand)  
  https://docs.nvidia.com/networking/

**Distributed Training**
- PyTorch Distributed Training Tutorial  
  https://pytorch.org/tutorials/beginner/dist_overview.html
- PyTorch FSDP Documentation  
  https://pytorch.org/docs/stable/fsdp.html

**Tool**
- nccl-tests (통신 성능 벤치마크)  
  https://github.com/NVIDIA/nccl-tests
- NVIDIA Nsight Systems (프로파일러)  
  https://developer.nvidia.com/nsight-systems

---

GPU 클러스터의 통신 인프라는 **계층적 구조**로 이해해야 한다. 노드 내부는 NVLink/NVSwitch, 노드 간은 InfiniBand, 그리고 이 모든 것을 최적화하는 NCCL이 분산 학습의 뼈대를 이룬다. 분산 학습의 성능은 병렬화 전략 이전에, **어떤 통신을 어떤 링크 계층에서 처리하느냐**로 결정된다.
