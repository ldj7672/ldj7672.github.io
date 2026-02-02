---
title: "머신러닝과 컴퓨터비전을 위한 '선형 대수'"
date: 2024-12-23
categories:
  - Fundamentals
tags:
  - Machine Learning
  - Linear Algebra
---

# 1. 선형대수 기초  

선형대수는 머신러닝과 컴퓨터 비전에서 데이터를 표현하고 처리하는 데 필수적인 핵심 이론이다. 이 장에서는 선형대수의 기초 개념과 연산을 정리하며, 각 개념이 AI/ML에서 어떻게 활용되는지 간단히 살펴본다.


## 1.1. 스칼라, 벡터, 행렬, 텐서  
### 1.1.1. 스칼라(Scalar)
**스칼라(Scalar)**는 하나의 숫자를 나타내는 가장 기본적인 데이터 단위로, **크기(magnitude)만을 가지며 방향은 없다**. 머신러닝에서는 단일 변수, 손실 함수 값, 또는 특정 계산 결과를 스칼라로 표현하는 경우가 많다. 예를 들어, $a = 5$와 같은 형태로 나타낼 수 있다.


### 1.1.2. 벡터(Vector)
**벡터(Vector)는 숫자의 1차원 배열로, 크기와 방향을 동시에 표현할 수 있는 데이터** 단위이다. 머신러닝에서는 데이터 포인트, 입력 특징(feature), 또는 모델의 가중치(weight)를 벡터로 나타낸다. 벡터는 **행 벡터(row vector)**나 **열 벡터(column vector)**로 표현되며, 연산의 맥락에 따라 사용 방식이 달라질 수 있다. 예를 들어, 행 벡터는 $\mathbf{v} = [v_1, v_2, \dots, v_n]$으로, 열 벡터는 $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$으로 나타낼 수 있다. 머신러닝에서는 하나의 데이터 포인트(특징 벡터)가 보통 $n$차원의 벡터로 표현되는 경우가 많다.  
   

### 1.1.3. 행렬(Matrix)
**행렬(Matrix)는 숫자의 2차원 배열로, 행(row)과 열(column)로 구성**된다. 머신러닝과 딥러닝에서는 데이터를 저장하거나 선형 변환(Linear Transformation)을 수행하는 데 필수적으로 사용된다. 하나의 행렬은 여러 데이터 포인트를 저장할 수 있으며, 딥러닝 모델에서는 가중치(weight)를 행렬로 나타내어 입력과 곱해 다음 계층으로 전달된다. 또한, 행렬 연산은 데이터를 회전, 이동, 또는 스케일링하는 선형 변환을 표현하는 데 사용된다. 예를 들어, $\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$는 2행 3열의 행렬을 나타내며, 이는 $2$개의 데이터 포인트와 각각 $3$개의 특징을 포함한다고 볼 수 있다.

### 1.1.4. 텐서(Tensor)
**텐서(Tensor)는 숫자의 다차원 배열**로, **벡터와 행렬의 확장된 형태**이다. 머신러닝과 딥러닝에서는 고차원 데이터를 표현하는 데 필수적으로 사용된다. 텐서는 차원에 따라 다양한 형태를 가진다. 예를 들어, 1차원 배열은 벡터(1D 텐서), 2차원 배열은 행렬(2D 텐서), 3차원 이상의 배열은 고차원 텐서로 표현된다. 특히, RGB 이미지는 $H \times W \times 3$ 형태의 3차원 텐서로 나타내며, 딥러닝에서는 텐서를 활용해 입력 데이터, 가중치, 중간 계산 값을 효율적으로 표현한다.

예를 들어, 4차원 텐서 $\mathcal{T} \in \mathbb{R}^{32 \times 224 \times 224 \times 3}$은 배치 크기 32인 $224 \times 224$ RGB 이미지 데이터셋을 나타낸다. 이러한 텐서는 TensorFlow, PyTorch와 같은 딥러닝 프레임워크에서 연산 효율성을 높이는 데 사용된다.



## 1.2. 벡터와 행렬 연산  

### 1.2.1. 벡터 덧셈과 스칼라 곱
- 벡터 덧셈: $\mathbf{a} + \mathbf{b} = [a_1 + b_1, \dots, a_n + b_n]$ 
- 스칼라 곱: $c \cdot \mathbf{v} = [cv_1, cv_2, \dots, cv_n]$  
- 머신러닝에서는 가중치 업데이트나 방향 조정을 표현할 때 사용된다.

### 1.2.2. 행렬 곱(Matrix Multiplication)
- $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}$, 여기서 $\mathbf{A} \in \mathbb{R}^{m \times n}, \mathbf{B} \in \mathbb{R}^{n \times p}$  
- 딥러닝 신경망에서 입력과 가중치를 곱하거나 변환할 때 사용된다.

### 1.2.3. 전치 행렬(Transpose)
- **전치 행렬**은 행렬 $\mathbf{A}$의 행과 열을 교환한 행렬로, $\mathbf{A}^T$로 표기한다.  
  - e.g. $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$이면, $\mathbf{A}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$.  
- 전치 연산은 **행렬 곱셈의 순서를 바꾸는 데 유용**합니다.  
  - e.g. $(\mathbf{A} \cdot \mathbf{B})^T = \mathbf{B}^T \cdot \mathbf{A}^T$
- **활용 예시**  
  - **벡터 내적(Dot Product) 표현**: $\mathbf{x} \cdot \mathbf{y}$는 벡터 $\mathbf{x}$를 전치 행렬로 변환하여 $\mathbf{x}^T \mathbf{y}$ 형태로 계산할 수 있다.  
  - **이미지 데이터 변환**: 컴퓨터 비전에서 전치 행렬은 **이미지의 차원을 전환**하거나, **컬러 채널(RGB)**의 위치를 변경하는 데 활용된다.  



## 1.3. 선형 변환  

선형 변환은 **벡터 공간에서 벡터를 다른 벡터로 변환하는 함수**를 의미한다. 이는 **벡터와 행렬의 곱셈으로 표현**되며, 수식으로는 $\mathbf{y} = \mathbf{A} \cdot \mathbf{x}$와 같이 나타낼 수 있다. 여기서 $\mathbf{A}$는 선형 변환을 나타내는 행렬이고, $\mathbf{x}$는 변환 이전의 벡터, $\mathbf{y}$는 변환 이후의 벡터이다.  

선형 변환은 **회전, 이동, 확대/축소와 같은 기하학적 변환에서 자주 사용**된다. 예를 들어, 컴퓨터 비전에서는 이미지를 확대하거나 축소하고, 특정 각도로 회전시키는 작업을 수행할 때 선형 변환을 활용한다. 이러한 변환은 행렬 연산으로 효율적으로 처리할 수 있어, 딥러닝과 같은 대규모 데이터 처리에서도 필수적인 개념으로 자리 잡고 있다.  



## 1.4. 역행렬과 정방 행렬  

### 1.4.1. 정방 행렬(Square Matrix)
**정방 행렬(Square Matrix)**는 행(row)과 열(column)의 개수가 같은 $n \times n$ 형태의 행렬을 말한다. 예를 들어, $\mathbf{A} \in \mathbb{R}^{n \times n}$은 $n$개의 행과 $n$개의 열을 가진 정방 행렬이다. 정방 행렬은 선형대수에서 매우 중요한 역할을 하며, 역행렬 계산, 행렬식(Determinant) 계산, 그리고 고유값 및 고유벡터를 구하는 다양한 연산의 기초가 된다.

### 1.4.2. 역행렬(Inverse Matrix)
**역행렬(Inverse Matrix)**는 행렬 연산에서 "역원"의 역할을 하는 행렬로, $\mathbf{A}^{-1}$로 표기된다. 역행렬은 주어진 행렬 $\mathbf{A}$에 대해, 다음 조건을 만족한다.
$$
\mathbf{A}^{-1} \cdot \mathbf{A} = \mathbf{I}, \quad \mathbf{A} \cdot \mathbf{A}^{-1} = \mathbf{I}
$$  
여기서 $\mathbf{I}$는 단위 행렬(Identity Matrix)로, 대각선 성분이 모두 1이고 나머지 성분은 0인 $n \times n$ 크기의 행렬이다. 역행렬을 가진 행렬은 **가역 행렬(Invertible Matrix)**이라고 하며, 역행렬은 연립 방정식 풀이나 선형 회귀에서 가중치 계산에 사용된다.

$
\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix}$일 때, $\mathbf{A}^{-1} = \frac{1}{\text{det}(\mathbf{A})} \begin{bmatrix} 4 & -1 \\ -3 & 2 \end{bmatrix} 
$


### 1.4.3. 행렬식(Determinant)
$$
\text{det}\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$  
행렬식은 정방 행렬 $\mathbf{A}$의 특성을 나타내는 값으로, $\text{det}(\mathbf{A})$로 표기한다. 행렬식은 다음과 같은 역할을 한다.
	
1. **가역성 판별**: $\text{det}(\mathbf{A}) \neq 0$이면 $\mathbf{A}$는 가역 행렬(역행렬을 가짐)이다.  
2. **선형 변환의 크기**: 행렬식은 선형 변환에서 면적(2D), 부피(3D)와 같은 크기의 변화를 나타낸다.  
3. **특수 행렬 판별**: $\text{det}(\mathbf{A}) = 0$이면 $\mathbf{A}$는 특이 행렬(Singular Matrix)로, 역행렬을 가지지 않는다.  

$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 3 & 4 \end{bmatrix}$일 때, $\text{det}(\mathbf{A}) = (2)(4) - (1)(3) = 8 - 3 = 5$  


## 1.5. 특이값 분해  
특이값 분해(SVD)는 행렬 $\mathbf{A}$를 세 개의 행렬 곱으로 분해하는 방법으로, 데이터의 구조를 분석하고, 차원을 축소하거나 정보를 압축하는 데 사용된다. 
  $$  
  \mathbf{A} = \mathbf{U} \Sigma \mathbf{V}^T  
  $$  
  - $\mathbf{U}$: $\mathbf{A}$의 열 공간(column space)을 나타내는 직교 행렬(orthogonal matrix)이다.  
  - $\Sigma$: 대각 행렬(diagonal matrix)로, 대각선에 특이값(singular values)이 위치한다.  
  - $\mathbf{V}^T$: $\mathbf{A}$의 행 공간(row space)을 나타내는 직교 행렬의 전치이다.  

**활용** 
- **차원 축소**: PCA(주성분 분석)는 SVD를 기반으로 고차원 데이터를 저차원으로 축소하여 주요 특징만을 보존한다.  
- **이미지 압축**: 이미지를 행렬로 표현한 뒤, 특이값 분해를 통해 정보 손실을 최소화하며 압축한다.  
- **노이즈 제거**: SVD를 사용해 데이터의 노이즈를 제거하고, 본질적인 정보를 추출한다.   


## 1.6. 고유값와 고유벡터
고유값과 고유벡터는 선형 변환을 나타내는 행렬의 중요한 성질을 표현한다. 고유값과 고유벡터는 데이터의 주요 성분을 찾고, 선형 변환의 성질을 분석하는 데 유용하다.  

$$  
\mathbf{A} \mathbf{v} = \lambda \mathbf{v}  
$$   
- $\mathbf{A}$: 선형 변환을 나타내는 행렬이다.  
- $\mathbf{v}$: 고유벡터로, 선형 변환 후에도 방향이 변하지 않는 벡터이다.  
- $\lambda$: 고유값으로, 선형 변환 후 벡터의 크기(스케일링)를 나타낸다.  

**활용**  
  
- **PCA(주성분 분석)**: 데이터의 분산을 최대화하는 방향(주성분)을 찾기 위해 고유값과 고유벡터를 사용한다. 고유값이 클수록 데이터의 주요 성분을 설명하는데 더 중요한 축이 된다.  
- **동적 시스템 분석**: 고유값은 선형 시스템의 안정성이나 동작 특성을 분석하는 데 사용된다.  
- **그래프 분석**: 그래프의 라플라시안(Laplacian) 행렬에서 고유값과 고유벡터는 네트워크의 클러스터링과 같은 문제에 활용된다.  


## 1.7. Norm 
Norm은 벡터의 크기 또는 길이를 측정하는 방법으로, 머신러닝과 최적화에서 매우 자주 사용된다. 주요 Norm의 종류는 다음과 같다.
  
### 1.7.1. L1 Norm
$$  
\|\mathbf{x}\|_1 = \sum |x_i|  
$$  

- 벡터 성분의 절대값 합으로 계산된다.  
- e.g. $\mathbf{x} = [1, -2, 3]$이라면, $\|\mathbf{x}\|_1 = |1| + |-2| + |3| = 6$  

### 1.7.2. L2 Norm
$$  
\|\mathbf{x}\|_2 = \sqrt{\sum x_i^2}  
$$  
    
- (유클리드 거리): 벡터 성분의 제곱합에 대한 제곱근으로 계산된다.  
- e.g. $\mathbf{x} = [1, -2, 3]$이라면, $\|\mathbf{x}\|_2 = \sqrt{1^2 + (-2)^2 + 3^2} = \sqrt{14}$  

**활용**  
- **Normalization**: 데이터를 정규화할 때 Norm을 사용하여 벡터의 크기를 1로 조정한다.  
- **Loss 함수 설계**: 머신러닝에서 $\ell_1$ Norm은 라쏘(Lasso), $\ell_2$ Norm은 릿지(Ridge) 회귀와 같은 정규화 기법에서 활용된다.  
- **최적화 문제**: 모델 학습에서 Norm은 가중치의 크기를 제어하고 과적합을 방지한다.  


## 1.8. Projection 
프로젝션은 한 벡터를 다른 벡터나 공간 위로 투영하는 연산이다.  
$$  
\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}  
$$  

- $\mathbf{a}$는 투영할 벡터, $\mathbf{b}$는 투영 대상 벡터이다. 결과는 $\mathbf{b}$ 방향으로 투영된 $\mathbf{a}$의 성분을 나타낸다.  

**활용**  
- **컴퓨터 비전**: 3D 데이터를 2D 평면으로 투영하여 이미지나 영상을 생성할 때 사용된다. (e.g. 카메라에서 물체의 3D 좌표를 2D 이미지 평면으로 변환)
- **회귀 분석**: 선형 회귀에서 입력 데이터 $\mathbf{X}$를 출력 $\mathbf{y}$에 투영하여, 잔차(residual)를 최소화하는 최적의 모델을 찾는다.  
- **데이터 분석**: 고차원 데이터를 특정 축으로 투영하여, 데이터를 이해하거나 시각화하는 데 사용된다.  

---  
# 2. 머신러닝에서의 선형대수 활용  

**머신러닝의 여러 알고리즘과 모델 학습 과정은 선형대수의 원리를 기반으로 설계**되어 있다. 선형 회귀와 같은 고전적인 알고리즘부터 딥러닝 모델까지, 선형대수는 데이터의 표현, 학습, 그리고 최적화 과정에서 핵심적인 역할을 한다. 이 섹션에서는 머신러닝에서 선형대수가 어떻게 활용되는지 구체적으로 살펴본다.


## 2.1. Linear Regression  

**선형 회귀는 입력 데이터와 출력 데이터 간의 선형 관계를 학습하는 알고리즘**이다. 수학적으로 선형 회귀는 다음과 같이 표현된다. 

$$  
\mathbf{y} = \mathbf{X} \cdot \mathbf{w} + \mathbf{b}  
$$  

- $\mathbf{X}$: 입력 데이터 행렬($m \times n$). $m$은 데이터 샘플 수, $n$은 특징(feature) 수.  
- $\mathbf{w}$: 가중치(weight) 벡터($n \times 1$).  
- $\mathbf{b}$: 편향(bias) 또는 절편(intercept).  
- $\mathbf{y}$: 예측된 출력값 벡터($m \times 1$).  

Loss 함수(평균 제곱 오차, MSE)를 최소화하기 위해 선형대수 연산을 활용하여 최적 가중치를 다음과 같이 계산할 수 있다.  
$$  
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}  
$$  
이 방정식은 선형대수의 **전치 행렬(Transpose)**, **행렬 곱(Matrix Multiplication)**, **역행렬(Inverse Matrix)**을 사용하여 유도된다. 이러한 폐쇄형 해(closed-form solution)는 선형 회귀 모델을 효율적으로 푸는 데 유용하며, 특히 소규모 데이터셋에서는 계산 효율성을 높이는 데 사용된다.


## 2.2. 딥러닝과 행렬 연산  

**딥러닝 모델은 선형대수를 기반으로 설계된 구조로, 입력과 가중치를 행렬로 표현하고 이를 곱하는 과정을 통해 예측값을 생성**한다. 각 계층의 출력은 다음과 같은 식으로 계산된다.  
$$  
\mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)} \cdot \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})  
$$  

- $\mathbf{W}^{(l)}$: $l$번째 계층의 가중치 행렬.  
- $\mathbf{h}^{(l)}$: $l$번째 계층의 출력 벡터(활성화 함수 $\sigma$를 포함).  
- $\mathbf{b}^{(l)}$: 편향 벡터.  

**딥러닝의 학습 과정에서 중요한 역전파(backpropagation)는 손실 함수의 그라디언트를 계산해 가중치를 업데이트하는 과정으로, 선형대수 연산이 필수적**이다. 예를 들어, 가중치 $\mathbf{W}$의 그라디언트는 다음과 같이 계산된다.  
$$  
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{h}^{(l-1)} \cdot \delta^{(l)}  
$$  
여기서 $\delta^{(l)}$는 $l$번째 계층의 에러를 나타낸다.

딥러닝에서 선형대수는 입력 데이터와 가중치의 행렬 곱, 전치 행렬을 활용한 그라디언트 계산, 그리고 고차원 데이터를 효율적으로 처리하는 텐서 연산 등 다양한 연산에 필수적으로 사용된다.


## 2.3. 차원 축소와 PCA  

**PCA(Principal Component Analysis)는 데이터의 분산을 최대화하는 저차원 공간(주성분)을 찾는 차원 축소 기법**으로, 선형대수의 고유값 분해(Eigen Decomposition)와 특이값 분해(Singular Value Decomposition, SVD)를 기반으로 동작한다.  

먼저 입력 데이터 $\mathbf{X}$의 공분산 행렬을 계산한다.  
$$  
\mathbf{C} = \frac{1}{m} \mathbf{X}^T \mathbf{X}  
$$  
그다음, $\mathbf{C}$의 고유값 $\lambda$와 고유벡터 $\mathbf{v}$를 계산하여, 고유값이 큰 순서대로 주성분을 선택합니다. 이를 통해 데이터 $\mathbf{X}$를 저차원 공간에 투영합니다.  
$$  
\mathbf{Z} = \mathbf{X} \cdot \mathbf{V}_k  
$$  
여기서 $\mathbf{V}_k$는 상위 $k$개의 주성분으로 구성된 행렬이다.

PCA는 차원 축소와 노이즈 제거에 주로 사용된다. 고차원 데이터를 2차원 또는 3차원으로 축소하여 시각화하거나, 분산이 작은 주성분을 제거해 데이터의 핵심 정보를 유지하면서도 노이즈를 제거할 수 있다.


## 2.4. Regularization  

**머신러닝에서 정규화(Regularization)는 모델의 복잡도를 제어하여 과적합(overfitting)을 방지하는 데 사용**된다. 정규화는 손실 함수에 추가 항을 포함하여 가중치 $\mathbf{w}$의 크기를 제한한다.  

$$  
L = \|\mathbf{y} - \mathbf{X} \cdot \mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_2^2  
$$  

**L2 정규화(Ridge Regression)**는 위와 같이 정의된다. $\lambda$는 규제 강도를 조절하는 하이퍼파라미터로, $\ell_2$ 노름을 사용하여 큰 가중치에 페널티를 부과한다.  

$$  
L = \|\mathbf{y} - \mathbf{X} \cdot \mathbf{w}\|_2^2 + \lambda \|\mathbf{w}\|_1  
$$  

**L1 정규화(Lasso Regression)**는 위와 같이 정의되며, $\ell_1$ 노름을 사용하여 희소성(sparsity)을 유도한다.  

정규화는 과적합을 방지하고 중요하지 않은 특징(feature)을 제거하여 간결한 모델을 설계하는 데 유용하다. L1 정규화는 희소성을 유도해 가중치의 많은 항목을 0으로 만들 수 있으며, L2 정규화는 큰 가중치를 줄임으로써 모델의 안정성을 높인다.

---

# 3. 컴퓨터 비전에서의 선형대수 활용  
컴퓨터비전 분야에서도 선형대수는 없어선 안될 필수 이론이다. **이미지는 픽셀 값으로 이루어진 행렬로 표현**되며, **선형대수는 이러한 데이터를 변환, 분석, 그리고 이해하는 데 핵심적인 역할**을 한다. 

## 3.1. 이미지와 행렬 표현  

컴퓨터 비전에서 **이미지(Image)**는 픽셀 값의 배열로 표현된다.  
- **흑백 이미지**는 단일 채널의 2차원 행렬 $\mathbf{I} \in \mathbb{R}^{H \times W}$로 나타낸다.  
- **컬러 이미지**는 RGB 채널을 가지는 3차원 텐서 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$로 표현된다.  

예를 들어, $256 \times 256$ 크기의 컬러 이미지는 다음과 같은 구조를 가진다.  
$$  
\mathbf{I} = \begin{bmatrix}  
[r_{11}, g_{11}, b_{11}] & \dots & [r_{1W}, g_{1W}, b_{1W}] \\  
\vdots & \ddots & \vdots \\  
[r_{H1}, g_{H1}, b_{H1}] & \dots & [r_{HW}, g_{HW}, b_{HW}] \\  
\end{bmatrix}  
$$  

이처럼 **이미지를 행렬로 표현하면, 선형대수의 연산을 통해 데이터를 변환하거나 분석할 수** 있다. 대표적으로 이미지의 이동, 회전, 확대/축소와 같은 변환이 있다.


## 3.2. Geometric Transform 

**기하학적 변환(Geometric Transform)은 이미지를 이동, 회전, 확대/축소하거나, 3D 데이터를 2D로 투영하는 작업을 수행**한다. 이러한 변환은 변형의 정도에 따라 변형이 적은 것(선형적 변환)에서 많은 것(비선형적 변환) 순으로 나눌 수 있다. 모든 변환은 **선형대수의 행렬 연산으로 정의**되고 처리된다.


### 3.2.1. Euclidean Transform 

**Euclidean Transform은 이미지를 이동(Translation)하거나 회전(Rotation)하는 변환**으로, 거리를 보존하는 특징이 있다.  

$$  
\mathbf{p}' = \mathbf{R} \cdot \mathbf{p} + \mathbf{t}  
$$  

$$  
\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}  
$$

- $\mathbf{R}$: $2 \times 2$ 회전 행렬
- $\mathbf{t}$: 이동 벡터
- $\mathbf{p}$, $\mathbf{p}'$: 변환 전후의 좌표



### 3.2.2. Similarity Transform

**Similarity Transform 은 유클리드 변환에 크기 변경(Scaling)을 추가한 변환**으로, 선형적인 형태를 유지한다. 각도를 보존하지만 크기는 변할 수 있으며, 이미지 확대/축소를 수행하면서도 원본의 비율(aspect ratio)는 유지한다.

$$  
\mathbf{p}' = s \cdot \mathbf{R} \cdot \mathbf{p} + \mathbf{t}  
$$  

$s$는 스케일 팩터(Scale Factor)를 나타낸다.  


### 3.2.3. Affine Transform

**Affine Transform은 Similarity Transform에 전단(Shear)을 추가한 변환**이다. **직진성과 평행성이 보존**되며, 이동, 회전, 확대/축소뿐만 아니라 shear를 허용한다.

$$  
\mathbf{p}' = \mathbf{A} \cdot \mathbf{p} + \mathbf{t}  
$$  
- $\mathbf{A}$: $2 \times 2$ 변환 행렬
- $\mathbf{t}$: 이동 벡터
- $\mathbf{p}$와 $\mathbf{p}'$: 변환 전후의 좌표 


### 3.2.4. Perspective Transform

**Perspective Transform은 이미지의 원근법(perspective)을 반영하여 변환**한다. 이는 **3D 공간에서 평면으로의 투영 관계**를 나타낸다. 직선은 직선으로 유지되지만 **평행성은 보존되지 않는다.**

$$  
\mathbf{x}' = \mathbf{H} \cdot \mathbf{x}  
$$  
- $\mathbf{H}$: $3 \times 3$ 호모그래피(Homography) 행렬
- $\mathbf{x}$: 변환 전의 호모지니어스 좌표
- $\mathbf{x}'$: 변환 후의 좌표
  


## 3.3. Convolution (컨볼루션)  

컨볼루션은 CNN(Convolutional Neural Networks)의 핵심 연산으로, 필터를 사용해 이미지에서 특정 특징(에지, 코너 등)을 추출합니다. 수학적으로, **컨볼루션 연산은 입력 행렬(이미지)과 필터(커널) 행렬 간의 내적(dot product)을 여러 위치에서 계산하는 방식**으로 정의됩니다.
$$  
\mathbf{y}_{ij} = \sum_{m=1}^{k} \sum_{n=1}^{k} \mathbf{I}_{(i+m-1)(j+n-1)} \cdot \mathbf{K}_{mn}  
$$  
- $\mathbf{I}$: 입력 이미지 행렬.  
- $\mathbf{K}$: 컨볼루션 필터(커널) 행렬.  
- $\mathbf{y}_{ij}$: 출력 행렬의 요소.  

컨볼루션은 선형대수의 **내적(dot product)**과 **행렬 연산(matrix operations)**을 기반으로 동작한다. 컨볼루션 연산은 입력 이미지 행렬 $\mathbf{I}$에서 작은 부분 영역(패치)을 선택하고, 이 패치와 필터 행렬 $\mathbf{K}$ 간의 내적을 계산하여 새로운 행렬을 생성한다. 이를 여러 위치에서 반복적으로 수행함으로써 입력 데이터의 특징을 추출한다.

또한, 컨볼루션은 입력 데이터를 선형적으로 변환하여 특정 패턴이나 특징(예: 에지, 코너 등)을 탐지하는 과정으로 이해할 수 있다. 필터 행렬 $\mathbf{K}$는 특정 특징을 탐지하도록 설계되거나 학습되며, 입력 데이터에 선형 변환을 적용해 해당 특징의 존재 여부를 나타내는 값을 출력한다.


## 3.4. Edge Detection
에지 검출은 이미지에서 물체의 경계선을 찾는 작업으로, 픽셀 값의 변화율(Gradient)을 계산하여 수행된다. 대표적인 에지 검출 알고리즘으로 Sobel 필터와 Canny Edge Detection이 있다.  
 
$$  
G = \sqrt{G_x^2 + G_y^2}  
$$  
- $G_x$: 입력 이미지와 수평 필터의 컨볼루션 결과
- $G_y$: 입력 이미지와 수직 필터의 컨볼루션 결과


Sobel 필터는 수평($G_x$)과 수직($G_y$) 방향의 **기울기(Gradient)**를 계산한다. 입력 이미지 행렬 $\mathbf{I}$와 필터 행렬 $\mathbf{K}$ 간의 컨볼루션 연산을 통해 수평($G_x$)과 수직($G_y$) 방향의 Gradient를 계산한다. 이 과정에서 각 방향의 Gradient 크기와 방향을 계산하며, 결과적으로 이미지의 에지를 검출한다. 특히, 계산된 Gradient의 크기와 방향은 벡터 공간에서 **Vector Norm**을 사용해 표현된다. 이를 통해 에지의 강도와 방향성을 이해할 수 있다.

**Canny Edge Detection**은 Sobel 필터를 확장한 에지 검출 알고리즘으로, 여러 선형대수적 연산 단계를 포함한다. 첫 번째 단계에서는 **Sobel 필터와 유사한 방식으로 각 픽셀의 Gradient를 계산해 이미지의 기울기 정보**를 얻는다. 이후, Non-Maximum Suppression 단계에서는 **계산된 Gradient 크기를 행렬로 저장한 뒤, 각 픽셀의 값을 주변 픽셀들과 비교하여 에지의 최대값만 남기고 나머지는 제거**한다. 마지막으로, **이중 임계값(Double Threshold) 단계를 통해 Gradient 값이 특정 임계값 범위 내에 있는 경우를 에지로 분류하여 에지와 비에지를 효과적으로 구분**한다. 이 모든 과정은 선형대수 연산을 기반으로 에지를 보다 정확하게 추출할 수 있도록 설계되었다.


## 3.5. Feature Extraction 
**Feature Extraction은 이미지에서 중요한 정보를 벡터 형태로 변환하는 과정**으로, 객체 탐지나 이미지 매칭 같은 작업에서 핵심적으로 사용된다. 이 과정에는 **SIFT(Scale-Invariant Feature Transform)**와 **ORB(Oriented FAST and Rotated BRIEF)**와 같은 대표적인 기법들이 사용된다.

**SIFT는 이미지의 스케일(크기), 회전, 조명 변화에 강건한 특징점을 추출하는 방법**으로, 먼저 강도가 급격히 변하는 영역(예: 에지나 코너)을 탐지한 후, 해당 특징점을 중심으로 지역 패치를 분석하여 고유한 디스크립터를 생성한다. 이후, 다른 이미지의 디스크립터와 비교하여 유사성을 계산하는데, 이는 두 디스크립터 간의 Euclidean Distance를 통해 측정된다. 이 과정에서 거리 계산은 선형대수의 Vector Norm 개념에 기반하며, 수학적으로는 $|\mathbf{f}_1 - \mathbf{f}_2|_2$로 표현된다.

반면, **ORB는 SIFT와 유사한 작업을 수행하지만, 계산이 훨씬 간단**하며 실시간 애플리케이션에서도 효과적으로 사용된다. ORB는 빠르게 특징점을 탐지하고 회전 및 크기 변화에 강건한 디스크립터를 생성하며, 두 디스크립터 간의 유사성은 Hamming Distance를 통해 계산된다.

**특징 추출 과정에서 선형대수는 데이터의 표현과 비교를 위한 수학적 기초를 제공**한다. 이미지의 지역 패치를 행렬로 표현하고 이를 필터나 변환 행렬과 결합하여 특정 패턴을 추출하는 과정은 행렬 연산으로 이해할 수 있다. 또한, 추출된 특징점을 고차원 벡터로 변환하여 데이터의 구조를 보존하며, 벡터 간의 내적(Dot Product)이나 거리 계산을 통해 이미지 간의 관계를 분석하는 데 중요한 역할을 한다.


## 3.6. 3D Reconstruction 
**3D Reconstruction은 2D 이미지에서 3D 공간의 정보를 복원하는 과정**으로, 컴퓨터 비전의 핵심 응용 중 하나이다. 자율주행 차량, 증강현실(AR), 로봇 공학 등 다양한 분야에서 중요한 역할을 한다. 이 과정은 선형대수학을 기반으로 카메라 모델링, 투영 관계 분석, 그리고 3D 점 추정을 수행한다. 특히, **Triangulation**, **Fundamental Matrix**, **Essential Matrix**는 3D Reconstruction에서 중요한 구성 요소이다.


### 3.6.1. Triangulation

$$
\mathbf{A} \cdot \mathbf{X} = \mathbf{0}
$$
 
- $\mathbf{A}$: 두 카메라의 투영 행렬과 대응점 정보를 포함한 행렬.  
- $\mathbf{X}$: 추정하려는 3D 점의 호모지니어스 좌표.  

**Triangulation은 두 시점에서 촬영된 이미지를 기반으로 3D 공간상의 점을 계산하는 방법**이다. 이 방법은 두 카메라에서 관찰된 동일한 2D 점, 즉 대응점(Correspondence)을 활용하며, 각 카메라로부터 나오는 광선(ray)이 교차하는 지점을 계산해 3D 위치를 추정한다. 이 과정에서 두 카메라의 투영 행렬과 대응점 정보를 포함한 행렬을 사용하여 선형 방정식 시스템을 설정하고, 이를 통해 3D 점의 호모지니어스 좌표를 계산한다. 이러한 계산은 보통 **SVD와 같은 선형대수적 기법을 통해 수행**된다.


### 3.6.2. Epipolar Geometry

![](https://velog.velcdn.com/images/ldj923/post/a87323d8-348a-4eb8-ac95-f2789510206a/image.png)


**Epipolar Geometry(에피폴라 기하학)는 두 카메라 간의 기하학적 관계를 설명하는 이론**으로, 3D Reconstruction에서 매우 중요한 역할을 한다. 두 이미지에서 대응점(Correspondence)을 찾는 과정과 이를 기반으로 한 3D 구조 복원에 필요한 기초를 제공한다.  

1. **Epipole**  
**Epipole은 한 카메라의 중심이 다른 카메라의 이미지 평면에 투영된 점**으로, 두 카메라의 광축(optical axis)이 교차하지 않을 경우, 에피폴은 이미지 평면 내에 존재한다. 또한 에피폴은 모든 에피폴라 선이 지나가는 공통점이다.  

2. **Epipolar Line**  
**Epipolar Line은 한 이미지에서 특정 점이 주어졌을 때, 해당 점이 다른 이미지에서 대응점이 존재할 수 있는 선**이다. 이는 두 카메라와 3D 점을 연결하는 평면(에피폴라 평면)에 의해 정의된다. 대응점 탐색 공간을 2D에서 1D로 축소하여 계산 효율성을 크게 향상시키는 역할을 한다.

3. **Epipolar Constraint**  
$$
\mathbf{x}'^T \cdot \mathbf{F} \cdot \mathbf{x} = 0
$$  
Epipolar Constraint은 두 이미지 간의 기하학적 관계를 수학적으로 나타낸 것으로, 한 이미지에서의 점과 대응되는 점이 다른 이미지의 에피폴라 선 위에 있어야 한다는 제약 조건을 의미한다. 대응점 $\mathbf{x}$와 $\mathbf{x}'$는 위 수식을 만족해야 한다. ($\mathbf{F}$: Fundamental Matrix)


Epipolar Geometry는 Fundamental Matrix와 Essential Matrix의 기초가 되는 개념으로, 3D Reconstruction 과정에서 중요한 역할을 한다. **Fundamental Matrix는 두 이미지 간의 에피폴라 기하학을 수학적으로 정의**하며, **대응점들이 에피폴라 제약을 만족하도록** 제약을 제공한다. **Essential Matrix는 여기에 카메라의 내부 파라미터를 추가하여, 에피폴라 기하학을 실제 3D 공간으로 확장**하는 역할을 한다.

또한, Epipolar Geometry는 두 이미지 간의 대응점을 효과적으로 찾을 수 있는 기반을 마련한다. 에피폴라 제약을 통해 검색 공간을 줄이고, Triangulation과 같은 3D Reconstruction 작업을 수행하는 데 필요한 기하학적 관계를 명확히 정의한다. 이를 통해 두 카메라 간의 관계를 이해하고, 3D 공간에서 구조를 복원할 수 있도록 돕는다.



### 3.6.3. Fundamental Matrix와 Essential Matrix

### Fundamental Matrix ($\mathbf{F}$)   
$$
\mathbf{x}'^T \cdot \mathbf{F} \cdot \mathbf{x} = 0
$$  

  $$ 
  \mathbf{F} = \begin{bmatrix} 
  f_{11} & f_{12} & f_{13} \\ 
  f_{21} & f_{22} & f_{23} \\ 
  f_{31} & f_{32} & f_{33} 
  \end{bmatrix}, \mathbf{x} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \mathbf{x}' = \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} 
	$$
    
- **$\mathbf{x}'$**: 두 번째 이미지에서의 점의 호모지니어스 좌표  
- **$\mathbf{x}$**: 첫 번째 이미지에서의 점의 호모지니어스 좌표  
- **$\mathbf{F}$**: Fundamental Matrix  

**Fundamental Matrix는 두 카메라 간의 기하학적 관계를 나타내는 $3 \times 3$ 행렬**로, 두 이미지에서의 픽셀 좌표 간 관계를 정의한다. 구체적으로, 두 이미지의 픽셀 좌표 $\mathbf{x}$와 $\mathbf{x}'$ 사이의 관계를 수학적으로 표현하며, 이는 에피폴라 기하학(Epipolar Geometry)을 기반으로 한다. Fundamental Matrix는 두 이미지 간의 대응점들이 만족해야 할 제약 조건을 제공하며, 이 제약 조건을 통해 두 이미지 사이의 픽셀 좌표 간의 관계를 명확히 정의한다.

**Fundamental Matrix는 카메라의 내부 파라미터를 알지 못하는 경우에도 추정할 수 있으며, 이를 계산하기 위해 8-point 알고리즘과 같은 기법이 자주 사용**된다. 이러한 계산은 두 카메라의 위치와 방향에 기반하여, 두 이미지의 대응점 간의 구조적 관계를 파악할 수 있게 한다.


### Essential Matrix ($\mathbf{E}$)  
$$
\mathbf{E} = \mathbf{K}'^T \cdot \mathbf{F} \cdot \mathbf{K}
$$  

  $$ 
  \mathbf{E} = \begin{bmatrix} 
  e_{11} & e_{12} & e_{13} \\ 
  e_{21} & e_{22} & e_{23} \\ 
  e_{31} & e_{32} & e_{33} 
  \end{bmatrix} , \mathbf{K} = \begin{bmatrix} 
  f_x & 0 & c_x \\ 
  0 & f_y & c_y \\ 
  0 & 0 & 1 
  \end{bmatrix} 
  $$  

$f_x, f_y$는 각각 x, y 방향의 초점 거리이고, $c_x, c_y$는 이미지 센터의 좌표이다.
  
- **$\mathbf{K}$**: 첫 번째 카메라의 내부 파라미터 행렬  
- **$\mathbf{K}'$**: 두 번째 카메라의 내부 파라미터 행렬 (구조는 $\mathbf{K}$와 동일).  
- **$\mathbf{F}$**: Fundamental Matrix (위에서 정의한 $3 \times 3$ 행렬).  
- **$\mathbf{E}$**: Essential Matrix  
 
**Essential Matrix는 Fundamental Matrix와 유사하지만, 카메라의 내부 파라미터(Intrinsic Parameters)를 알고 있는 경우에 사용**된다. 카메라 내부 파라미터에는 초점 거리나 이미지 센서의 중심점 위치와 같은 정보가 포함되며, 이러한 정보를 통해 Essential Matrix는 두 카메라 간의 픽셀 좌표 관계를 더 정밀하게 나타낸다.

**Essential Matrix의 주요 역할은 두 카메라 간의 상대적인 위치와 회전을 포함하는 것으로, 이를 통해 카메라 간의 회전 행렬($\mathbf{R}$)과 이동 벡터($\mathbf{t}$)를 추정**할 수 있다. 이러한 계산은 Essential Matrix를 SVD로 분해하여 수행되며, 이를 기반으로 두 카메라 간의 3D 공간적 관계를 추정할 수 있다.


### 3.6.4. 3D Reconstruction의 수학적 흐름
**3D Reconstruction은 2D 이미지로부터 3D 공간의 구조를 복원하는 과정**으로, 선형대수학에 기반한 여러 단계의 수학적 흐름을 따른다. 먼저, **카메라 보정(Camera Calibration)을 통해 각 카메라의 내부 및 외부 파라미터를 추정**한다. 내부 파라미터는 초점 거리와 주점 위치처럼 카메라 자체의 광학적 특성을 나타내며, 외부 파라미터는 카메라의 3D 공간에서의 위치와 방향(회전 및 이동)을 나타낸다. 이 단계는 카메라와 실제 세계 간의 투영 관계를 정확히 모델링하기 위해 필수적이다.

다음으로, **Fundamental Matrix를 계산하여 두 이미지 간의 기하학적 관계를 정의**한다. Fundamental Matrix는 대응점(Correspondence)들이 에피폴라 기하학(Epipolar Geometry)을 만족하도록 제약을 제공하며, 두 이미지의 픽셀 좌표 간의 관계를 설명한다. 그러나 Fundamental Matrix는 카메라의 내부 파라미터를 고려하지 않으므로, 이를 보완하기 위해 **Essential Matrix를 계산**한다. Essential Matrix는 Fundamental Matrix를 카메라 내부 파라미터로 조정하여, 픽셀 좌표를 정규화된 카메라 좌표로 변환한다. 이를 통해 두 카메라 간의 회전($\mathbf{R}$)과 이동($\mathbf{t}$) 정보를 포함하며, 3D 공간에서의 기하학적 관계를 명확히 표현할 수 있다.

**Essential Matrix를 구한 후에는 이를 SVD를 통해 분해하여 두 카메라의 상대적 위치와 방향을 추정**한다. 이렇게 **얻어진 회전과 이동 정보는 마지막 단계인 Triangulation에 사용**된다. Triangulation은 두 카메라의 투영 행렬과 이미지 상의 대응점을 기반으로 **3D 점의 위치를 계산하는 과정**이다. 두 카메라에서 관찰된 동일한 2D 점에서 출발하는 광선(ray)이 교차하는 지점을 계산하여, 3D 공간에서 해당 점의 위치를 추정한다.

이러한 과정을 통해 2D 이미지의 픽셀 좌표를 3D 공간의 점으로 변환할 수 있으며, 이는 자율주행, 증강현실, 로봇 공학 등 다양한 응용 분야에서 활용된다. 3D Reconstruction은 선형대수학의 강력한 계산 능력을 활용하여 이미지 데이터를 공간적 정보로 변환하는 핵심 기술이다.

---

이제 "선형대수"에 대한 정리가 마무리되었다. 선형대수는 정말 중요하지만, 늘 공부했다가 까먹고, 필요할 때 다시 찾아보는 과정을 반복하는 학문인 것 같다.

그래도 오늘처럼 이렇게 정리해 두면, 머릿속에 좀 더 오래 남아 있지 않을까 하는 마음으로 글을 마무리해본다.