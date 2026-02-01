---
title: "텍스처와 표면 반사율을 활용한 재질 인식 연구"
date: 2024-11-24
categories:
  - AI Projects
tags:
  - Machine Learning
  - Computer Vision
---

> 이번 포스팅에서는 대학원 석사 과정에서 진행한 **Material Type Recognition** 연구를 소개합니다.

- _MaterialNet: Multi-scale Texture Hierarchy and Multi-view Surface Reflectance for Material Type Recognition(BMVC. 2022)_

---

# 재질 인식(Material Recognition)이란?

![](https://velog.velcdn.com/images/ldj923/post/f8ec3208-6737-4cfb-af48-db408fffffbe/image.png)

**재질 인식**은 **물체의 표면 재질(예: 나무, 금속, 유리 등)을 식별하는 기술**로, 일반적인 객체 분류와는 달리 물체의 유형이 아니라 그 구성 재질에 초점을 맞춥니다. 예를 들어, 위의 이미지에서 세 개의 물체는 모두 '의자'라는 공통점을 가지지만, 각각의 재질은 '나무(Wood)', '패브릭(Fabric)', '금속(Metal)'로 다릅니다.

재질 인식은 물체 분류와 의미적 분할(semantic segmentation)의 정밀도를 높이고, 로봇이 다양한 물체를 더 정확하고 섬세하게 다룰 수 있도록 돕습니다. 또한, 컴퓨터 그래픽스에서는 재질 특성(예: 표면 반사율)을 활용해 더 사실적이고 생동감 있는 3D 렌더링을 구현하는 데 중요한 역할을 합니다.


## 재질 인식의 주요 도전 과제
![](https://velog.velcdn.com/images/ldj923/post/7861a237-7e39-4c57-bac2-44e023198909/image.png)
- **복잡한 시각적 특성**: 재질은 텍스처, 반사율, 강도, 마찰 등의 시각적 및 물리적 특성으로 정의되는데, 그중 많은 특성이 단일 RGB 이미지로 추정하기 어렵습니다.
- **환경 변화**: 조명이나 시점 변화는 재질 특성의 시각적 표현에 큰 영향을 미쳐 인식 정확도를 떨어뜨립니다.
- **다양한 스케일**: 동일한 재질이라도 서로 다른 스케일에서의 텍스처(작은 표면, 중간 크기 패턴, 큰 컨텍스트)가 다양하게 나타납니다.


## 주요 접근법
기존 연구는 주로 두 가지 접근법에 기반했습니다.

- **텍스처 기반**: 합성곱 신경망(CNN)을 활용해 이미지의 텍스처 특징을 학습하는 방식이 주를 이루었습니다.
	(e.g. Fisher Vector CNN, Texture Encoding Network (DeepTEN))
    
- **반사율 기반**: 다중 시점 이미지에서 물체의 표면 반사율을 추정하여 재질을 분류했습니다.
	(e.g. Reflectance Hashing, Differential Angular Imaging (DAIN))

이들 방법은 각각 특정 재질 속성에만 의존했거나, 제한적인 실험 환경에서 성능을 보이는 한계가 있었습니다.

---

# Proposed Methods: MaterialNet
본 연구에서는 재질 인식을 위한 새로운 네트워크인 **MaterialNet**을 제안했습니다. MaterialNet은 재질을 보다 정밀하게 인식하기 위해 **Multi-Scale Texture Hierarchy**와 **Multi-View Surface Reflectance**를 효과적으로 추출하여 활용합니다.

## Multi-Scale Texture Hierarchy
<p style="margin: 0; padding: 0;">
    <img src="https://velog.velcdn.com/images/ldj923/post/8eefea2a-d4ca-41b3-84f6-8d692e7b1d18/image.png" alt="Image 2">
</p>

**Multi-Scale Texture Hierarchy**는 **다양한 스케일에서 텍스처(표면 질감) 특징을 계층적으로 분석**하는 것을 의미합니다.

- 작은 스케일: 재질의 미세한 표면 패턴(예: 나무의 결, 천의 섬유 조직)을 포착합니다.
- 중간 스케일: 개별 물체 또는 재질의 부분적 모양을 파악합니다.
- 큰 스케일: 재질이 배치된 맥락(예: 나무 판자 더미, 천이 덮인 소파)을 이해합니다.

이러한 텍스처에 대한 계층적 접근은 다음과 같은 이유로 재질 인식에서 필수적이라고 생각했습니다.

- **재질의 복잡한 특성 포착**: 대부분의 재질은 단일 스케일로는 충분히 설명되지 않습니다. 예를 들어, 패브릭 재질은 미세한 섬유의 질감(작은 스케일)뿐만 아니라 주름진 모양(중간 스케일)과 배치된 환경(큰 스케일)을 종합적으로 이해해야만 올바르게 분류할 수 있습니다.
- **환경 변화에 강인**: 텍스처는 조명이나 시점 변화에도 비교적 안정적이며, 다양한 스케일의 텍스처를 결합하면 더 견고한 재질 표현이 가능합니다.

<br>

### Multi-Scale Texture Hierarchy Network (MSTH-Net)
<p style="margin: 0; padding: 0;">
    <img src="https://velog.velcdn.com/images/ldj923/post/9ad69d77-0173-4a9c-9f93-2e2818ccbda4/image.png" alt="Image 1">
</p>

**MSTH-Net은 Multi-Scale Texture Hierarchy를 효과적으로 추출하기 위해 다양한 스케일에서 텍스처 특징을 계층적으로 분석하고, LSTM을 활용해 텍스처 간의 관계를 학습**합니다.

- 다양한 스케일에서 텍스처를 추출해 텍스처 계층 구조를 형성합니다.
- 환경 변화(조명, 시점)에 강인한 특징을 학습합니다.
- CNN 백본(ResNet 등)에서 추출한 다중 스케일 특징을 LSTM으로 결합하여 일관된 텍스처 계층을 구축합니다.

<br>

## Multi-View Surface Reflectance
<p style="margin: 0; padding: 0;">
    <img src="https://velog.velcdn.com/images/ldj923/post/53a9a0f9-b306-49f6-b814-3a45729f6741/image.png" alt="Image 2">
</p>

**Multi-View Surface Reflectance**는 물체의 **표면 반사 특성을 여러 시점에서 분석하여 재질의 고유한 반응을 파악**하는 접근법입니다.

- **표면 반사율이란**: 조명 조건이 변화할 때 표면이 빛을 반사하는 방식으로, 재질의 광학적 특성을 나타냅니다. 예를 들어, 금속은 강한 반사를, 천은 약한 반사를 보입니다.
- **다중 시점의 중요성**: 단일 시점에서는 조명 조건이 고정되어 반사 특성을 충분히 포착할 수 없습니다. 반면, 여러 시점에서 관찰하면 표면 반응의 변화를 통해 재질의 고유한 특성을 더 정밀하게 분석할 수 있습니다.

<br>

### Multi-View Surface Reflectance Network (MVSR-Net)
<p style="margin: 0; padding: 0;">
    <img src="https://velog.velcdn.com/images/ldj923/post/8cc4a1f6-7456-4c67-9c6c-37d008e2b243/image.png" alt="Image 1">
</p>

**MVSR-Net은 Multi-View Surface Reflectance를 추출하기 위해 다중 시점 이미지에서 관찰된 반사율 변화를 LSTM으로 학습**하여, 시점 간의 조명 변화에 따른 표면 특성을 효과적으로 파악합니다.

- 다중 시점 이미지를 활용해 표면 반사율 정보를 추출합니다.
- 조명 변화에 따른 반사율 차이를 학습하여 재질을 더욱 구체적으로 식별합니다.

### MaterialNet

**MaterialNet은 이 두 가지 모듈을 결합하여 텍스처와 반사율 정보를 모두 활용**하는 네트워크로 재질을 구체적이고 종합적으로 인식할 수 있습니다.

- Describable Texture Dataset (DTD), MINC-2500 등 6개의 벤치마크에서 기존 방법보다 우수한 성능 달성.
- GTOS 및 MIW 데이터셋에서 multi-view 이미지를 활용해 기존 모델보다 높은 재질 인식 정확도를 달성.

---

# 맺음말
재질 인식은 제가 대학원 석사 과정에서 연구했던 분야입니다. 운 좋게도 연구 결과를 BMVC 학회에 논문으로 게재할 수 있었던 건 정말 뿌듯한 기억으로 남아 있어요. 당시에는 나름 열심히 했다고 생각했지만, 지금 돌이켜보면 많이 부족하고 어설펐던 부분이 많았던 것 같습니다.

비록 지금 일하는 곳에서 재질 인식 연구를 직접적으로 활용하고 있지는 않지만, 그 과정에서 논문을 읽고 문제점을 파악하며 실험을 설계하고 성능을 개선해 나갔던 경험들은 지금 ML & Vision 엔지니어로 일하는 데 큰 밑거름이 되고 있습니다.