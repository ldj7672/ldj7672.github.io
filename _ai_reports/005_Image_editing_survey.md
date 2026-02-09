---
title: "이미지 편집 기술 최신 연구 동향 (2022-2025)"
date: 2025-02-02
categories:
  - AI Reports
tags:
  - Image Editing
  - Diffusion Models
  - Computer Vision
category: image-video-generation
---

최근 diffusion 모델의 발전과 함께 이미지 편집 기술은 급속도로 진화하고 있다. 단순한 필터 적용을 넘어 텍스트 기반 편집, 객체 삽입/제거, 구조 보존 편집 등 다양한 편집 시나리오를 지원하는 기술들이 등장하고 있다. 최근에는 나노바나나, GPT Image, z image, FLUX.2 등 스케일 아웃된 최신 이미지 편집 모델들이 많이 공개되었지만, 본 리포트에서는 2022년부터 2025년까지의 주요 이미지 편집 기법들이 어떻게 발전해왔는지 연도별로 추적하며, 각 기법의 핵심 아이디어와 한계를 분석한다.


## **1. 인트로**

본 리포트는 최근까지의 이미지 편집 연구가 어떻게 발전해왔는지를 추적하는 서베이를 목적으로 한다. 이미지 편집 기술은 크게 세 가지 접근 방식으로 발전해왔다. 첫째, **개체 개념 주입(Concept Injection)** 방식으로, DreamBooth와 Textual Inversion이 대표적이다. 소수의 이미지로 특정 객체를 학습시켜 새로운 장면에서도 동일한 객체를 생성할 수 있게 한다. 둘째, **조건 기반 편집(Conditional Editing)** 방식으로, ControlNet과 GLIGEN이 대표적이다. 구조 정보(edge, depth, pose 등)나 바운딩 박스를 활용해 객체의 위치와 형태를 정밀하게 제어한다. 셋째, **Inversion-free 편집** 방식으로, PixelMan과 Edicho가 대표적이다. 기존 이미지를 latent space로 되돌리는 inversion 과정 없이도 정확한 편집을 수행한다.

본 리포트에서 다루는 시점까지의 연구 동향은 **원하는 것만 비용 효율적으로 정밀하게 편집**하는 방향으로 발전해왔다. Inversion-free, prompt-less 편집, 구조 일관성 유지, 멀티 이미지 정합성, 객체 단위 정밀 편집 등이 핵심 키워드였다. 본 리포트에서는 이러한 발전 과정을 연도별로 추적하며, 각 기법의 기술적 특징과 한계를 분석한다.

## **2. Concept Injection (2022)**

### **2.1 DreamBooth (2022)**

**"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"**

![](/figures/image_editing/dreambooth.png)

DreamBooth는 소수의 이미지(3~5장)를 활용하여 기존 diffusion 모델에 새로운 개체 개념을 주입하는 기법이다. 핵심 아이디어는 특정 객체를 나타내는 **unique identifier**를 학습시키는 것이다.

**작동 원리:**
- 입력 이미지에 특정 토큰(예: "a [V] dog")을 할당하고, 해당 토큰의 임베딩을 학습한다.
- 학습된 토큰을 프롬프트에 사용하면(예: "a [V] dog in the living room"), 다양한 배경에서도 동일한 객체가 재현된다.
- **Class-specific prior preservation loss**를 도입하여 객체의 고유 특성은 유지하면서도 다른 클래스의 표현은 훼손하지 않도록 한다.

**장점:**
- 고정된 특정 상품이나 인물을 정확히 표현할 때 적합하다.
- 소수의 이미지로도 효과적인 학습이 가능하다.
- Subject recontextualization, text-guided view synthesis, appearance modification, artistic rendering 등 다양한 응용이 가능하다.

**한계:**
- 객체마다 개별 학습이 필요하여 확장성이 제한적이다.
- 학습 시간이 상대적으로 길다.

### **2.2 Textual Inversion (2022)**

**"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion"**

![](/figures/image_editing/textual_inversion.png)

Textual Inversion은 새로운 개념을 **단 하나의 임베딩 벡터**로 표현하는 기법이다. DreamBooth와 달리 모델의 전체 가중치를 업데이트하지 않고, 텍스트 임베딩 공간에서 새로운 토큰만 학습한다.

**작동 원리:**
- 특정 객체를 나타내는 새로운 텍스트 토큰(예: "@sofaA")의 임베딩 벡터를 학습한다.
- 기존 diffusion 모델의 가중치는 freeze하고, 텍스트 임베딩 레이어만 업데이트한다.
- 학습된 토큰을 프롬프트에 사용하면 해당 객체를 생성할 수 있다.

**장점:**
- 모델을 재학습하지 않고도 특정 객체를 생성할 수 있다.
- 학습 파라미터가 매우 적어 효율적이다.
- 여러 개념을 동시에 학습하여 조합 가능하다.

**한계:**
- 복잡한 객체나 세부 디테일 표현에 한계가 있다.
- 학습된 임베딩이 특정 모델에 종속적이다.

## **3. Conditional Editing (2023)**

### **3.1 ControlNet (2023)**

**"Adding Conditional Control to Text-to-Image Diffusion Models"**

![](/figures/image_editing/controlnet.png)

ControlNet은 Stable Diffusion의 UNet 중간 feature map에 **Condition Hint**를 제공하는 기법이다. 픽셀 기반 구조 정보를 활용해 객체의 형태, 위치, 윤곽선 등을 유지한 채 이미지를 생성한다.

**작동 원리:**
- Edge map, Depth map, Pose map, Segmentation map 등 구조 정보를 입력으로 받는다.
- 기존 UNet의 가중치를 freeze하고, ControlNet 모듈만 학습한다.
- Zero convolution을 통해 학습 초기에 기존 성능을 유지하면서 점진적으로 조건 정보를 반영한다.

**장점:**
- 기존 모델의 성능을 유지하면서 구조 보존이 가능하다.
- 다양한 조건 타입(edge, depth, pose 등)을 지원한다.
- 학습이 안정적이고 빠르다.

**한계:**
- 조건 맵 생성이 필요하여 추가 전처리 단계가 필요하다.
- 복잡한 편집 시나리오에서는 제어가 어렵다.

### **3.2 GLIGEN (2023)**

**"GLIGEN: Open-Set Grounded Text-to-Image Generation"**

![](/figures/image_editing/GLIGEN.png)

GLIGEN은 기존 diffusion pipeline에 **Object Injection Module**을 추가하여, 텍스트 + 바운딩 박스 + 이미지 참조를 조합해 객체 배치를 가능하게 하는 기법이다.

**작동 원리:**
- 텍스트 프롬프트, 바운딩 박스 좌표, 참조 이미지를 동시에 입력으로 받는다.
- Object Injection Module이 참조 이미지의 특징을 추출하여 지정된 위치에 주입한다.
- Cross-attention 메커니즘을 통해 텍스트와 이미지 정보를 융합한다.

**장점:**
- 객체 배치의 자유도가 높다.
- 참조 이미지를 스타일뿐만 아니라 구체적 이미지로 활용할 수 있다.
- 복잡한 장면 구성이 가능하다.

**한계:**
- 바운딩 박스 지정이 필요하여 사용자 편의성이 떨어진다.
- 객체 간 상호작용 표현에 한계가 있다.

### **3.3 Prompt-to-Prompt (2023)**

**"Prompt-to-Prompt Image Editing with Cross-Attention Control"**

![](/figures/image_editing/prompt-to-prompt.png)

Prompt-to-Prompt는 기존 이미지에서 **교차 어텐션 맵(cross-attention map)을 고정**한 채 텍스트만 일부 수정하는 기법이다. 이미지 전체 구조(레이아웃)는 유지하면서 부분적 객체 수정/삽입이 가능하다.

**작동 원리:**
- 원본 이미지의 cross-attention map을 저장한다.
- 편집 프롬프트에서 변경할 부분만 수정하고, 나머지는 원본과 동일하게 유지한다.
- 저장된 attention map을 활용하여 레이아웃을 보존하면서 선택적 편집을 수행한다.

**장점:**
- 이미지 구조를 크게 변경하지 않고 부분 편집이 가능하다.
- Inversion 과정이 필요 없어 효율적이다.

**한계:**
- 복잡한 편집 시나리오에서는 attention map만으로는 부족하다.
- 객체 제거나 대규모 변경에는 한계가 있다.

### **3.4 Imagic (2023)**

**"Imagic: Text-Based Real Image Editing with Diffusion Models"**

![](/figures/image_editing/imagic.png)

Imagic은 실제 사진을 기반으로 **잠재 공간을 최적화**해 텍스트 편집을 가능하게 하는 기법이다. 결과 이미지는 기존 사진과 매우 유사하면서도 텍스트 지시 사항을 반영한다.

**작동 원리:**
- 원본 이미지를 latent space로 인코딩한다.
- 텍스트 임베딩과 latent를 함께 최적화하여 편집 방향을 찾는다.
- 최적화된 latent를 디코딩하여 최종 이미지를 생성한다.

**장점:**
- 고해상도 실사 이미지 편집에 적합하다.
- 사용자 맞춤형 편집에 강력하다.
- 원본 이미지의 품질을 크게 손상시키지 않는다.

**한계:**
- 최적화 과정이 시간이 오래 걸린다.
- 복잡한 편집에는 여러 번의 최적화가 필요하다.

## **4. Inversion-free Editing (2024)**

### **4.1 PixelMan (2024)**

**"PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation"**

![](/figures/image_editing/pixelman.png)

PixelMan은 **inversion-free이고 training-free인** diffusion 기반 이미지 편집 기법으로, 텍스트 프롬프트 없이도 이미지 내 객체를 픽셀 수준으로 복제·이동·삽입할 수 있다.

**작동 원리:**
- 픽셀 공간에서 객체의 복사본을 생성하고, 효율적인 샘플링 방식을 사용하여 조작된 객체를 목표 위치에 자연스럽게 통합한다.
- Diffusion step 중간에 **pixel-wise manipulation**을 직접 수행한다.
- 객체의 구조와 질감을 그대로 유지한 상태로 새로운 위치에 자연스럽게 배치한다.
- Inversion 없이도 정확도 높은 복제 편집을 지원하며, 단 16 inference step만으로도 우수한 결과를 달성한다.

**장점:**
- Inversion 과정이 필요 없고 추가 학습도 필요 없어 빠르고 효율적이다.
- 텍스트 프롬프트 없이도 직관적인 편집이 가능하다.
- 객체 구조와 질감을 정확히 보존하며 이미지 일관성을 유지한다.
- 기존 방법들이 50 step을 필요로 하는 반면, 16 step만으로도 우수한 성능을 보인다.

**한계:**
- 복잡한 장면에서는 객체 간 상호작용 표현이 어렵다.
- 배경 일관성 유지에 한계가 있다.

### **4.2 Edicho (2024)**

**"Edicho: Consistent Image Editing in the Wild"**

![](/figures/image_editing/edicho.png)

Edicho는 **training-free diffusion 기반 방법**으로, 여러 in-the-wild 이미지에서 일관된 이미지 편집을 수행한다. 핵심 설계 원칙은 암묵적 attention feature에 의존하기보다는 **명시적 이미지 대응 관계(explicit image correspondence)**를 사용하여 편집을 지시하는 것이다.

**작동 원리:**
- **Attention manipulation module**과 **refined classifier-free guidance (CFG) denoising strategy**를 사용한다.
- 두 구성 요소 모두 사전에 추정된 이미지 간 대응 관계를 활용한다.
- 이미지 간의 **cross-image correspondence**를 활용해 객체 삽입, 제거, 위치 이동 등을 자연스럽게 처리한다.
- 다양한 객체 포즈, 조명 조건, 촬영 환경의 변화에도 robust하게 작동한다.

**장점:**
- **Plug-and-play 호환성**: ControlNet, BrushNet 등 대부분의 diffusion 기반 편집 방법과 호환된다.
- **Zero-shot capability**: 다양한 설정에서 zero-shot으로 작동한다.
- 복수 이미지 편집을 수행해야 하는 상황(예: before-after 비교, 동일 객체 위치 변경 등)에 뛰어난 결과를 보인다.
- 사용자 개입 없이도 객체 배치가 논리적으로 정합된다.
- 동일한 장면의 서로 다른 시점에서 일관된 편집 결과를 생성할 수 있다.

**흥미로운 확장 가능성:**
- 이미지 간 key feature matching을 통해 3D reconstruction(카메라 파라미터 없이)까지 가능하다는 연구 결과가 있다.
- 아직까지는 체리피킹이지만, 이 기술이 잘 발전하면 3D reconstruction을 위한 학습 데이터셋 생성에도 활용할 수 있을 것으로 기대된다.

**한계:**
- 두 이미지 간의 대응 관계가 명확하지 않으면 성능이 저하된다.
- 복잡한 장면에서는 계산 비용이 증가한다.

## **5. Intrinsic-based Editing (2025)**

### **5.1 IntrinsicEdit (2025)**

**"IntrinsicEdit: Precise generative image manipulation in intrinsic space"**

![](/figures/image_editing/intrinsicedit.png)

IntrinsicEdit은 intrinsic-image latent space에서 작동하는 생성적 워크플로우를 도입한다. 이미지를 shape, albedo(reflectance), lighting 구성 요소로 분해하여 정밀한 픽셀 수준 조작을 가능하게 한다. **RGB-X diffusion framework**를 기반으로 하며, identity preservation과 channel entanglement 문제를 해결한다.

**작동 원리:**
- 이미지를 **intrinsic 채널**(shape, albedo, lighting)로 분해한다.
- **Exact diffusion inversion**과 **disentangled channel manipulation**을 통해 필요한 채널만 선택적으로 편집한다.
- 편집된 채널들을 다시 조합하여 최종 이미지를 생성한다(re-rendering).
- 추가 데이터 수집이나 모델 파인튜닝 없이도 자동으로 전역 조명 효과를 해결한다.

**장점:**
- 전체 이미지가 아닌 **조명·재질·부분 구조** 등 세부적 컨트롤이 가능하다.
- 색상 및 텍스처 조정, 객체 삽입 및 제거, 전역 재조명 등 다양한 편집 작업을 정밀하게 수행할 수 있다.
- 기존 이미지를 크게 손상시키지 않고도 매우 정교한 스타일 변화나 객체 속성 조절이 가능하다.
- 고품질 이미지 보존에 강점을 가진다.

**한계:**
- Intrinsic 분해 과정이 복잡하고 계산 비용이 높다.
- 모든 이미지에 대해 정확한 intrinsic 분해가 가능한 것은 아니다.

## **6. Research Directions**

본 리포트에서 다루는 시점(2022-2025년 초)까지의 이미지 편집 연구는 다음과 같은 방향으로 발전해왔다.

### **6.1 Inversion-free, Prompt-less 편집**

기존 이미지를 latent space로 되돌리는 inversion 과정 없이도 정확한 편집을 수행하는 방향으로 발전했다. PixelMan과 Edicho가 대표적이며, 편집 속도와 효율성을 크게 향상시켰다. 또한 텍스트 프롬프트 없이도 직관적인 편집이 가능해 사용자 편의성이 향상되었다.

### **6.2 구조 일관성 유지 + 멀티 이미지 정합성**

편집 과정에서 이미지의 구조적 일관성을 유지하면서, 여러 이미지 간의 정합성도 보장하는 방향으로 발전했다. Edicho가 대표적이며, 복수 이미지 편집 시나리오에서 일관된 결과를 생성할 수 있다.

### **6.3 객체 단위 정밀 편집**

전체 이미지가 아닌 특정 객체만 선택적으로 편집하는 방향으로 발전했다. IntrinsicEdit이 대표적이며, 조명, 재질, 구조 등 세부적 속성을 정밀하게 제어할 수 있다.

### **6.4 비용 효율적 편집**

**원하는 것만 비용 효율적으로 정밀하게 편집**하는 것이 이 시점까지의 연구 목표였다. 이를 위해 다음과 같은 기술들이 연구되었다:
- 부분적 모델 업데이트 (LoRA, ControlNet 등)
- 효율적인 최적화 알고리즘
- 사전 학습된 모델의 재활용

---

이미지 편집 기술은 2022년의 개체 개념 주입 기법에서 시작하여, 2023년의 조건 기반 편집, 2024년의 Inversion-free 편집을 거쳐, 2025년의 Intrinsic 기반 편집까지 빠르게 발전해왔다. 각 기법은 고유한 장점과 한계를 가지고 있으며, 연구자들은 이러한 한계를 극복하고 더욱 효율적이고 정밀한 편집을 목표로 지속적으로 개선해왔다.

본 리포트에서 다루는 시점까지의 연구 동향은 **Inversion-free, prompt-less 편집**, **구조 일관성 유지**, **멀티 이미지 정합성**, **객체 단위 정밀 편집** 등으로 수렴해왔으며, 이는 실용적인 이미지 편집 도구 개발에 중요한 방향성을 제시했다.

다만, 본 리포트가 다루는 시점 이후로는 이미 이러한 consistency 유지를 넘어서 **multi-reference 편집**, **더욱 정밀한 객체 단위 제어**, 그리고 **대규모 스케일 아웃 모델**(나노바나나, GPT Image, z image, FLUX.2 등)들이 등장하여 이미지 편집 기술은 새로운 단계로 진입하고 있다. 본 리포트는 이러한 최신 트렌드로 이어지는 연구 발전 과정을 추적하는 서베이로서, 각 기법들이 어떻게 진화해왔는지를 정리한 것이다.
