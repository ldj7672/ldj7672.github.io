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

대규모 모델 학습에서 "GPU가 충분히 빠른데 왜 학습이 안 빨라지지?"라는 질문의 답은 대부분 **통신**에 있다. 최신 GPU(H100, A100)는 TFLOPS 수준에서 압도적이지만, 분산 학습의 실질적인 throughput은 **GPU 간 데이터 이동 효율**에 의해 결정된다. 이 글은 GPU 클러스터 통신을 구성하는 핵심 인프라를 ML Engineer 관점에서 다뤄보려 한다.

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

**정의**: 단일 서버 내 여러 GPU 간 통신
- **인터커넥트**: NVLink, NVSwitch, PCIe
- **대역폭**: 300-900 GB/s (NVLink 기준)
- **지연**: ~1-5 μs
- **사용 사례**: Tensor Parallelism, 노드 내 데이터 병렬

### 1.2 Inter-node Communication (노드 간)

**정의**: 서로 다른 서버 간 통신
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

### 3.1 InfiniBand vs Ethernet

#### InfiniBand 특징

**장점**
- **저지연**: 1-2 μs (RDMA 기준)
- **고대역폭**: HDR (200 Gb/s), NDR (400 Gb/s)
- **RDMA 네이티브**: 커널 우회, CPU 오버헤드 최소화
- **Reliable transport**: 하드웨어 레벨 재전송/순서 보장

**InfiniBand 세대**
| 세대 | 대역폭 | 지연 | 주요 사용처 |
|------|--------|------|-----------|
| FDR | 56 Gb/s | ~2 μs | 구형 HPC |
| EDR | 100 Gb/s | ~1.5 μs | V100 클러스터 |
| HDR | 200 Gb/s | ~1 μs | A100 클러스터 |
| NDR | 400 Gb/s | ~0.6 μs | H100 클러스터 |
| XDR (예정) | 800 Gb/s | - | 차세대 |

#### Ethernet (RoCE)

**장점**
- **범용성**: 기존 네트워크 인프라 활용 가능
- **비용**: InfiniBand 대비 저렴
- **RoCE v2**: RDMA over Converged Ethernet (UDP 기반)

**단점**
- **지연**: 5-10 μs (RoCE v2)
- **안정성**: PFC(Priority Flow Control) 미세 튜닝 필요
- **Jitter**: 트래픽 혼재 시 성능 변동

### 3.2 Network Topology

#### Fat-Tree (가장 흔한 구조)

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

#### 실무 고려사항

**Topology-aware placement**
- **Intra-rack 우선**: Data Parallelism 그룹을 같은 rack에 배치
- **Rail-optimized IB**: 각 노드가 여러 IB 포트를 가질 때, 트래픽을 rail별로 분산
- **NCCL_TOPO_FILE**: 수동 토폴로지 정의로 NCCL 최적화

```bash
# 예시: 토폴로지 확인
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    CPU Affinity
GPU0     X      NV12    NV12    NV12    SYS     SYS     0-23
GPU1    NV12     X      NV12    NV12    SYS     SYS     0-23
NIC0    SYS     SYS     SYS     SYS      X      PIX     
```
- `NV#`: NVLink connection (숫자는 연결 개수)
- `SYS`: PCIe + NUMA crossing
- `PIX`: PCIe switches


## 4. RDMA & GPUDirect: Zero-Copy Communication

### 4.1 RDMA (Remote Direct Memory Access)

**전통적인 네트워크 통신 경로**
```
GPU Memory → CPU Memory (cudaMemcpy) 
→ Kernel buffer (socket API) 
→ NIC buffer 
→ Network
```
- **CPU 개입**: 모든 단계에서 CPU가 데이터 복사
- **지연 증가**: 복사 오버헤드 + 컨텍스트 스위치
- **대역폭 낭비**: 메모리 버스 경합

**RDMA 경로**
```
Memory → NIC (DMA) → Network
```
- **커널 우회**: 유저 공간에서 NIC 직접 제어
- **CPU 오프로드**: CPU는 통신에 관여하지 않음
- **낮은 지연**: 1-2 μs

#### RDMA 구현

**InfiniBand**: 네이티브 RDMA
- Verbs API (`libibverbs`)
- Queue Pair (QP) 기반 통신
- Completion Queue (CQ)로 비동기 처리

**RoCE (RDMA over Converged Ethernet)**
- InfiniBand Verbs를 Ethernet에 매핑
- v2는 UDP/IP 기반 (라우팅 가능)
- PFC (Priority Flow Control) 필수

### 4.2 GPUDirect RDMA

**문제**: RDMA만으로는 여전히 CPU 메모리 경유
```
GPU Memory → CPU Memory (PCIe) → NIC (RDMA) → Network
```

**GPUDirect RDMA**
```
GPU Memory → NIC → Network (PCIe P2P)
```
- **NIC가 GPU 메모리에 직접 DMA**
- **zero-copy**: CPU 메모리 우회
- **지연 감소**: 50-80% 단축

#### 구현 요구사항

1. **하드웨어**
   - NVIDIA GPU (Kepler 이상)
   - GPUDirect 지원 NIC (Mellanox/NVIDIA, Broadcom 등)
   - PCIe P2P 가능한 root complex

2. **소프트웨어**
   - 커널 모듈: `nvidia_peermem` (NVIDIA GPU + Mellanox NIC)
   - CUDA IPC 활성화
   - NCCL 2.0+

3. **검증**
```bash
# GPUDirect RDMA 활성화 확인
$ cat /sys/kernel/mm/memory_peers/nv_mem/version
$ lsmod | grep nvidia_peermem

# NCCL에서 GPUDirect 사용 확인
$ NCCL_DEBUG=INFO python train.py
# 로그에서 "Using GPUDirect RDMA" 확인
```

### 4.3 GPUDirect Storage

**추가 기술**: NVMe SSD → GPU 직접 전송
- **용도**: 대규모 데이터 로딩 (I/O bound 학습)
- **효과**: 데이터 로딩 시간 40-50% 단축
- **제약**: 특정 NVMe 컨트롤러 필요 (NVIDIA Magnum IO)


## 5. NCCL: Collective Communication Library

### 5.1 NCCL 아키텍처

**NCCL (NVIDIA Collective Communications Library)**는 GPU 분산 통신의 사실상 표준이다.

#### 핵심 기능

1. **Topology-aware optimization**:
   - NVLink, PCIe, InfiniBand 토폴로지 자동 감지
   - 최적 통신 경로 계산 (graph algorithm)
   - Ring, Tree, CollNet 알고리즘 혼합 사용

2. **알고리즘 선택**:
   - **Ring algorithm**: 대역폭 최적화 (큰 메시지)
   - **Tree algorithm**: 지연 최적화 (작은 메시지)
   - **CollNet**: 스위치 기반 acceleration (Sharp, AWS EFA)

3. **프로토콜 계층**:
   - **LL (Low Latency)**: 작은 메시지용 (< 32 KB)
   - **LL128**: 중간 크기 (128-bit ops)
   - **Simple**: 큰 메시지용, 대역폭 우선

### 5.2 Collective Operations 상세

#### AllReduce

**목적**: 모든 GPU의 텐서를 합산/평균하여 동일한 결과를 모두에게 분배
```
GPU 0: [1, 2, 3]     →  [10, 14, 18] (sum)
GPU 1: [4, 5, 6]     →  [10, 14, 18]
GPU 2: [5, 7, 9]     →  [10, 14, 18]
```

**알고리즘**
1. **Ring AllReduce** (Baidu, 2017)
   - 각 GPU가 데이터를 N-1개 청크로 분할
   - N-1 step의 reduce-scatter + N-1 step의 allgather
   - **통신량**: `2(N-1)/N × message_size` ≈ `2 × message_size` (N이 클 때)
   - **대역폭 활용**: 거의 100% (링크 대역폭 포화)

2. **Tree AllReduce**
   - Binary tree 구조로 reduce → broadcast
   - **통신량**: `2 × log(N) × message_size`
   - **지연**: `2 × log(N) × latency`
   - **사용**: 작은 메시지 (< 128 KB)

**성능 모델**
```
T_allreduce = α + β × M
```
- α (latency): 알고리즘 의존 (log(N) for tree, constant for ring)
- β (bandwidth cost): `2M/B` (ring), `2M×log(N)/B` (tree)
- M: 메시지 크기
- B: 대역폭

#### ReduceScatter & AllGather

**ReduceScatter**: AllReduce의 전반부
```
GPU 0: [1, 2, 3, 4]  →  [10] (chunk 0의 합)
GPU 1: [4, 5, 6, 7]  →  [14] (chunk 1의 합)
GPU 2: [5, 7, 9, 8]  →  [18] (chunk 2의 합)
GPU 3: [0, 0, 0, 9]  →  [20] (chunk 3의 합)
```
- **통신량**: `(N-1)/N × message_size` ≈ `message_size`
- **사용**: ZeRO, FSDP의 gradient 동기화

**AllGather**: ReduceScatter의 역연산
```
GPU 0: [10]  →  [10, 14, 18, 20]
GPU 1: [14]  →  [10, 14, 18, 20]
GPU 2: [18]  →  [10, 14, 18, 20]
GPU 3: [20]  →  [10, 14, 18, 20]
```
- **통신량**: `(N-1)/N × message_size` ≈ `message_size`
- **사용**: ZeRO, FSDP의 파라미터 재구성

#### Broadcast & Reduce

**Broadcast**: 1:N 전송
- Root GPU → 모든 GPU
- 통신량: `message_size` (root에서 발신)
- 사용: 옵티마이저 상태 공유, 체크포인트 로딩

**Reduce**: N:1 전송
- 모든 GPU → Root GPU (결과 합산)
- 통신량: `message_size` (root에서 수신)
- 사용: 메트릭 집계

### 5.3 NCCL 성능 튜닝

#### 중요 환경 변수

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

#### 벤치마킹: nccl-tests

```bash
# 설치
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1

# AllReduce 벤치마크
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
- **algbw (algorithmic bandwidth)**: 실제 전송된 데이터 기준
- **busbw (bus bandwidth)**: 링크 활용 기준 (= algbw × (N-1)/N × 2 for allreduce)
- **목표**: busbw가 하드웨어 대역폭의 80-90%에 도달

#### 프로파일링

**Nsight Systems**
```bash
nsys profile -t cuda,nvtx,mpi --mpi-impl=openmpi \
  -o profile.qdrep \
  python -m torch.distributed.launch --nproc_per_node=8 train.py
```

**분석 포인트**
- **NCCL kernel 시간**: `ncclKernel*` 패턴
- **Gap 분석**: GPU idle 구간 → 통신 대기
- **Overlap 효율**: computation과 communication 중첩 비율

**PyTorch Profiler**
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


## 6. 통신 패턴과 병렬화 전략

### 6.1 Data Parallelism (DP/DDP)

**통신 패턴**
- **Forward**: 통신 없음
- **Backward**: AllReduce(gradients) 한 번

**통신량**
```
Communication_DP = 2 × Model_Size
```
- 모든 파라미터의 gradient를 AllReduce
- Ring AllReduce: 2× factor

**토폴로지 고려**
- **노드 간 분산 가능**: AllReduce는 비교적 노드 간에서도 효율적
- **대규모 확장**: 수백~수천 GPU까지 scaling (GPT-3: 1024 GPUs)
- **병목**: gradient 크기가 클 때 (큰 모델), 네트워크 대역폭

### 6.2 Tensor Parallelism (TP)

**통신 패턴**
- **Forward**: AllReduce 또는 AllGather (레이어당 2회)
- **Backward**: AllReduce 또는 ReduceScatter (레이어당 2회)

**통신량** (Megatron-LM 스타일)
```
Communication_TP = 4 × Activation_Size × Num_Layers
```
- 매 레이어마다 통신 발생

**토폴로지 고려**
- **노드 내부 제한 필수**: NVLink 필수 (노드 간 TP는 비현실적)
- **전형적 구성**: TP=8 (한 노드의 8 GPU)
- **병목**: 활성화 크기 (배치, 시퀀스 길이), 통신 빈도

### 6.3 Pipeline Parallelism (PP)

**통신 패턴**
- **Forward**: point-to-point send (activation)
- **Backward**: point-to-point send (gradient)

**통신량**
```
Communication_PP = 2 × Activation_Size × Microbatch_Count
```
- P2P이므로 collective 대비 통신량 적음

**토폴로지 고려**
- **노드 간 가능**: P2P는 대역폭보다 지연에 민감하지만, InfiniBand로 충분
- **전형적 구성**: stage를 노드 경계에 맞춰 배치
- **병목**: bubble time (GPU idle), 마이크로배치 스케줄링

### 6.4 3D Parallelism (DP + TP + PP)

**대규모 모델의 표준** (예: GPT-3, PaLM, Llama-3 405B):
```
TP = 8   (노드 내부, NVLink)
PP = 16  (stage를 노드 경계에 맞춤)
DP = 16  (데이터 병렬 복제)
---
Total = 8 × 16 × 16 = 2048 GPUs
```

**통신 특성**
- **TP**: 빈번, 작은 메시지, 노드 내부
- **PP**: 중간 빈도, 큰 메시지, 노드 간 가능
- **DP**: 낮은 빈도, 매우 큰 메시지, 노드 간

**최적화 원칙**
1. **TP를 노드 내부로**: NVLink 대역폭 활용
2. **PP를 노드 경계에 정렬**: 노드 간 통신 최소화
3. **DP를 최외각에**: 노드 간 통신이지만 빈도가 낮아 InfiniBand로 처리 가능

### 6.5 ZeRO (Zero Redundancy Optimizer)

**통신 패턴** (ZeRO-3 기준)
- **Forward**: AllGather(parameters)
- **Backward**: ReduceScatter(gradients)
- **Optimizer**: AllGather(optimizer states) - 필요 시

**통신량**
```
Communication_ZeRO3 = 3 × Model_Size (per iteration)
```
- AllGather: 1× Model_Size (forward)
- AllGather: 1× Model_Size (backward)
- ReduceScatter: 1× Model_Size (backward)

**통신 vs 메모리 트레이드오프**
- **ZeRO-0 (baseline)**: 통신 최소, 메모리 최대
- **ZeRO-1**: optimizer states 샤딩, 통신 약간 증가
- **ZeRO-2**: + gradients 샤딩, 통신 중간
- **ZeRO-3**: + parameters 샤딩, 통신 최대, 메모리 최소

**토폴로지 고려**
- **노드 간 가능하지만 비용 높음**: AllGather가 빈번
- **최적**: 노드 내부 ZeRO-3 + 노드 간 DP
- **PyTorch FSDP**: ZeRO-3 구현, hybrid sharding 지원


## 7. 실무 트러블슈팅

### 7.1 일반적인 병목 진단

#### 증상 1: GPU Utilization 낮음 + Step Time 높음

**원인**
- 통신 병목 (GPU가 통신 대기)
- I/O 병목 (데이터 로딩 느림)

**진단**
```bash
# GPU 상태 모니터링
nvidia-smi dmon -s ucmt

# PyTorch profiler
python -m torch.utils.bottleneck train.py

# NCCL 로그 분석
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV python train.py 2>&1 | grep "Ring\|Tree\|Using"
```

#### 증상 2: 노드 수 증가 시 Scaling 효율 하락

**원인**
- 노드 간 통신 병목
- 네트워크 토폴로지 문제 (oversubscription)

**진단**
```bash
# nccl-tests로 실제 대역폭 측정
mpirun -np 16 ./all_reduce_perf -b 8 -e 2G -f 2 -g 1

# 네트워크 상태 확인
ibstat                    # InfiniBand 상태
ibdiagnet                 # 전체 fabric 토폴로지
ib_write_bw               # 대역폭 테스트
```

**해결**
- 토폴로지 최적화 (rack-aware placement)
- NCCL 환경 변수 튜닝
- 네트워크 혼잡 제거 (다른 작업 분리)

#### 증상 3: 특정 노드만 느림 (straggler)

**원인**
- 하드웨어 이상 (GPU, NIC, 케이블)
- 토폴로지 불균형

**진단**
```bash
# 각 GPU별 throughput 확인
python -c "
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
"

# 하드웨어 상태
nvidia-smi -q -d PERFORMANCE,CLOCK  # GPU 상태
ethtool ib0                          # NIC 상태
```

### 7.2 GPUDirect RDMA 활성화 확인

```bash
# 1. 커널 모듈 확인
lsmod | grep nvidia_peermem
# 출력 없으면:
modprobe nvidia_peermem

# 2. NCCL 로그 확인
NCCL_DEBUG=INFO python train.py 2>&1 | grep "GPUDirect"
# 원하는 출력: "Using GPUDirect RDMA"

# 3. nvidia-smi topo 확인
nvidia-smi topo -m
# NIC와 GPU 간에 "SYS" 아닌 "NODE" 또는 "PHB" 표시가 좋음

# 4. NCCL 강제 활성화
export NCCL_NET_GDR_LEVEL=5  # 5 = 강제 활성화
export NCCL_NET_GDR_READ=1   # read 연산도 GPUDirect 사용
```

### 7.3 멀티 NIC 최적화

**문제**: 노드에 여러 NIC가 있을 때 NCCL이 하나만 사용할 수 있음

**해결**
```bash
# 모든 IB 어댑터 사용
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

# rail-optimized: 각 GPU가 특정 NIC에 affinity
# 예: GPU 0-1 → mlx5_0, GPU 2-3 → mlx5_1, ...
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_IB_HCA=mlx5_0

# 또는 NCCL_TOPO_FILE로 수동 지정
# (고급 사용자용)
```


## 8. 성능 모델링: 통신 비용 예측

### 8.1 Roofline Model (통신 포함)

**전통적 Roofline**: Compute vs Memory Bandwidth
```
Achievable TFLOPS = min(Peak TFLOPS, Operational Intensity × Memory BW)
```

**분산 학습 확장**:
```
Achievable Throughput = min(
    Compute Capacity,
    Memory Bandwidth,
    Network Bandwidth / Communication Volume
)
```

### 8.2 통신 비용 계산 예시

**시나리오**: GPT-3 (175B parameters) 학습
- **모델 크기**: 175B parameters × 2 bytes (fp16) = 350 GB
- **배치 크기**: 1536 (global), 마이크로배치 = 1
- **병렬화**: TP=8, PP=16, DP=16 (총 2048 GPUs)

#### TP 통신 (노드 내부)

```
Communication_TP = 4 × Activation_Size × Num_Layers
Activation_Size = Batch × Seq × Hidden × TP_degree
                = 1536/16/16 × 2048 × 12288 / 8
                ≈ 150 MB (per layer)
Layers = 96
Total = 4 × 150 MB × 96 = 57.6 GB

Time = 57.6 GB / (600 GB/s × 0.9) ≈ 0.106 sec
```

#### DP 통신 (노드 간)

```
Communication_DP = 2 × Model_Size / DP_degree
                 = 2 × 350 GB / 16
                 = 43.75 GB (per DP rank)

Time = 43.75 GB / (25 GB/s × 0.8) ≈ 2.19 sec
```

#### PP 통신 (노드 간)

```
Communication_PP = 2 × Activation_Size × Microbatches
                 ≈ 2 × 150 MB × 8 (microbatches)
                 = 2.4 GB

Time = 2.4 GB / 25 GB/s ≈ 0.096 sec
```

**총 통신 시간**: ~2.4 sec (DP가 지배적)

**Compute 시간** (예상): ~8 sec (forward + backward)

**통신 비율**: 2.4 / (8 + 2.4) ≈ **23%**

→ 통신이 전체 시간의 1/4을 차지, 최적화 필요

### 8.3 Scaling Efficiency 예측

**Weak Scaling** (배치 크기를 GPU 수에 비례하여 증가):
```
Efficiency = T_1 / T_N
```
- 이상적: 100% (통신이 없으면)
- 현실: 70-90% (통신 오버헤드)

**Strong Scaling** (배치 크기 고정):
```
Efficiency = T_1 / (N × T_N)
```
- 통신이 배치 크기에 비례하지 않으므로 효율 급격히 하락
- 예: 배치가 작아지면 computation 감소 > 통신은 동일

**임계점**
- **TP**: 8-16 GPUs (노드 내부 제한)
- **PP**: 수십 노드 (bubble time 증가)
- **DP**: 수천 GPUs (네트워크가 충분하면)


## 9. 최신 기술 동향

### 9.1 NVLink-C2C (Chip-to-Chip)

**Grace Hopper Superchip**
- CPU (Grace) ↔ GPU (Hopper) 간 **900 GB/s** NVLink
- CPU 메모리와 GPU 메모리 간 coherent access
- **영향**: CPU-GPU 데이터 이동 병목 제거 (특이 큰 feature 전처리)

### 9.2 InfiniBand XDR (800 Gb/s)

**예정**: 2026-2027
- **4배 대역폭** vs HDR
- **영향**: 노드 간 통신 병목 완화, DP scaling 개선

### 9.3 SHARP (Scalable Hierarchical Aggregation)

**개념**: InfiniBand 스위치 내부에서 reduction 수행
```
전통적: GPU → NIC → Switch → NIC → GPU (N steps)
SHARP:  GPU → NIC → Switch (reduction) → NIC → GPU
```
- **지연 감소**: 50-60%
- **대역폭 향상**: aggregation이 스위치에서 일어나므로 uplink 트래픽 감소
- **제약**: NVIDIA Quantum-2 스위치 필요, NCCL 2.12+

### 9.4 UltraEthernet Consortium

**목표**: Ethernet을 AI/ML 클러스터에 최적화
- **RDMA over Ethernet** 표준화
- **Congestion control** 개선 (DCQCN, HPCC)
- **목표 지연**: < 1 μs (InfiniBand와 동등)

---

결론적으로 GPU 클러스터의 통신 인프라는 **계층적 구조**로 이해해야 한다.

1. **Intra-node**: NVLink/NVSwitch (600-900 GB/s, < 2 μs)
2. **Inter-node**: InfiniBand/Ethernet (25-50 GB/s, 1-10 μs)
3. **Software**: NCCL (topology-aware collective optimization)

**핵심 원칙**
- **빈번한 통신은 노드 내부로** (TP → NVLink)
- **덜 빈번한 통신은 노드 간으로** (DP → InfiniBand)
- **토폴로지를 먼저 이해, 병렬화는 그 다음** (topology → strategy)

분산 학습의 성능은 "무슨 병렬화 전략을 쓰느냐" 이전에, **"어떤 통신을 어떤 링크 계층에서 처리하느냐"** 로 결정된다. 이 글이 제공한 기술적 기초를 바탕으로, 다음 단계는 구체적인 병렬화 전략(DP/TP/PP/ZeRO)의 조합과 대규모 클러스터에서의 실전 최적화로 나아갈 수 있다.

---

## 참고 자료

**Papers**
- Baidu Research (2017), "Bringing HPC Techniques to Deep Learning"
- Rajbhandari et al. (2020), "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- Shoeybi et al. (2019), "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"

**Documentation**
- NVIDIA NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- NVIDIA Collective Communications Library (NCCL) Developer Guide
- Mellanox RDMA Aware Networks Programming User Manual

**Tools**
- nccl-tests: https://github.com/NVIDIA/nccl-tests
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- PyTorch Distributed Training Guide: https://pytorch.org/tutorials/beginner/dist_overview.html
