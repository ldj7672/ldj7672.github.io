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

최근 diffusion 모델의 발전과 함께 이미지 편집 기술은 급속도로 진화하고 있다. 단순한 필터 적용을 넘어, 텍스트 기반 편집, 객체 삽입/제거, 구조 보존 편집 등 다양한 편집 시나리오를 지원하는 방법들이 등장했다. 특히 최근에는 나노바나나, GPT Image, z image, FLUX.2 등 스케일 아웃된 최신 이미지 편집 모델들이 공개되며 범용 에디팅 모델에 대한 기대도 커지고 있다. 다만 본 리포트는 이러한 최신 모델 자체를 비교하기보다는, **2022년부터 2025년 초까지** 공개된 대표 연구들을 통해 이미지 편집 기술이 어떤 문제의식과 설계 선택을 따라 발전해왔는지를 정리한다. 핵심은 편집에서 반복적으로 등장하는 질문, 즉 **무엇을 보존하고(정체성/구조/배경), 무엇을 바꿀지(속성/객체/레이아웃)**를 어떻게 분리해 다루는가이다.

## **1. 인트로**

본 리포트는 2022년부터 2025년 초까지 diffusion 기반 이미지 편집 연구의 발전 과정을 정리하는 서베이를 목적으로 한다. 

- **Concept Injection**: DreamBooth, Textual Inversion처럼 소수 이미지로 특정 개체/스타일을 모델에 주입해 재현성을 확보하는 방식  
- **Conditional / Grounded Control**: ControlNet, GLIGEN처럼 edge/depth/pose/box 등 구조·위치 조건을 사용해 편집의 공간적 정밀도를 높이는 방식  
- **Training-free Editing(Attention/Feature Control)**: Prompt-to-Prompt, PnP, MasaCtrl처럼 추가 학습 없이 attention/feature를 제어해 레이아웃과 정합성을 유지하는 방식  
- **Real Image Editing via Inversion/Optimization**: Null-text inversion, Imagic처럼 고정밀 재구성과 최적화를 통해 실사 이미지 편집 안정성을 확보하는 방식  
- **Instruction-following Editing**: InstructPix2Pix처럼 자연어 지시를 따르는 편집 인터페이스를 학습으로 구현하는 방식  
- **Consistency / Intrinsic Space**: PixelMan/Edicho의 일관성 강화, IntrinsicEdit의 intrinsic 분해처럼 보존-편집 분리를 더 명시적으로 다루는 방식

본 리포트에서 다루는 시점까지의 연구 동향은 궁극적으로 **원하는 것만 비용 효율적으로 정밀하게 편집**하는 방향으로 수렴해왔다. Inversion 부담을 낮추는 training-free 접근, 프롬프트 의존도를 줄이는 instruction 기반 편집, 구조·정체성의 일관성을 보장하는 정합성 제어, 그리고 조명·재질 등 속성 단위의 정밀 조작이 주요 키워드였다. 본 리포트에서는 이러한 흐름을 연도별 대표 기법과 함께 추적하며, 각 기법의 기술적 특징과 한계를 분석한다.

## **2. Concept Injection**

### **2.1 DreamBooth (2022)**

**"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation"**

![](/figures/image_editing/dreambooth.png)

DreamBooth는 소수의 이미지(3~5장)를 활용하여 기존 diffusion 모델에 새로운 개체 개념을 주입하는 기법이다. 핵심 아이디어는 특정 객체를 나타내는 **고유 식별자(unique identifier)**를 학습시키는 것이다.

**핵심**
- 입력 이미지에 특정 토큰(예: "a [V] dog")을 할당하고, 해당 토큰의 임베딩을 학습한다.
- 학습된 토큰을 프롬프트에 사용하면(예: "a [V] dog in the living room"), 다양한 배경에서도 동일한 객체가 재현된다.
- **Class-specific prior preservation loss**를 도입하여 객체의 고유 특성은 유지하면서도 다른 클래스의 표현은 훼손하지 않도록 한다.

**장점**
- 고정된 특정 상품이나 인물을 정확히 표현할 때 적합하다.
- 소수의 이미지로도 효과적인 학습이 가능하다.
- Subject recontextualization, text-guided view synthesis, appearance modification, artistic rendering 등 다양한 응용이 가능하다.

**한계**
- 객체마다 개별 학습이 필요하여 확장성이 제한적이다.
- 학습 시간이 상대적으로 길다.

### **2.2 Textual Inversion (2022)**

**"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion"**

![](/figures/image_editing/textual_inversion.png)

Textual Inversion은 새로운 개념을 **단 하나의 임베딩 벡터**로 표현하는 기법이다. DreamBooth와 달리 모델의 전체 가중치를 업데이트하지 않고, 텍스트 임베딩 공간에서 새로운 토큰만 학습한다.

**핵심**
- 특정 객체를 나타내는 새로운 텍스트 토큰(예: "@sofaA")의 임베딩 벡터를 학습한다.
- 기존 diffusion 모델의 가중치는 freeze하고, 텍스트 임베딩 레이어만 업데이트한다.
- 학습된 토큰을 프롬프트에 사용하면 해당 객체를 생성할 수 있다.

**장점**
- 모델을 재학습하지 않고도 특정 객체를 생성할 수 있다.
- 학습 파라미터가 매우 적어 효율적이다.
- 여러 개념을 동시에 학습하여 조합 가능하다.

**한계**
- 복잡한 객체나 세부 디테일 표현에 한계가 있다.
- 학습된 임베딩이 특정 모델에 종속적이다.

## **3. Conditional / Grounded Control**

### **3.1 ControlNet (2023)**

**"Adding Conditional Control to Text-to-Image Diffusion Models"**

![](/figures/image_editing/controlnet.png)

ControlNet은 Stable Diffusion의 UNet 중간 feature map에 **Condition Hint**를 제공하는 기법이다. 픽셀 기반 구조 정보를 활용해 객체의 형태, 위치, 윤곽선 등을 유지한 채 이미지를 생성한다.

**핵심**
- Edge map, Depth map, Pose map, Segmentation map 등 구조 정보를 입력으로 받는다.
- 기존 UNet의 가중치를 freeze하고, ControlNet 모듈만 학습한다.
- Zero convolution을 통해 학습 초기에 기존 성능을 유지하면서 점진적으로 조건 정보를 반영한다.

**장점**
- 기존 모델의 성능을 유지하면서 구조 보존이 가능하다.
- 다양한 조건 타입(edge, depth, pose 등)을 지원한다.
- 학습이 안정적이고 빠르다.

**한계**
- 조건 맵 생성이 필요하여 추가 전처리 단계가 필요하다.
- 복잡한 편집 시나리오에서는 제어가 어렵다.

### **3.2 GLIGEN (2023)**

**"GLIGEN: Open-Set Grounded Text-to-Image Generation"**

![](/figures/image_editing/GLIGEN.png)

GLIGEN은 기존 diffusion pipeline에 **Object Injection Module**을 추가하여, 텍스트 + 바운딩 박스 + 이미지 참조를 조합해 객체 배치를 가능하게 하는 기법이다.

**핵심**
- 텍스트 프롬프트, 바운딩 박스 좌표, 참조 이미지를 동시에 입력으로 받는다.
- Object Injection Module이 참조 이미지의 특징을 추출하여 지정된 위치에 주입한다.
- Cross-attention 메커니즘을 통해 텍스트와 이미지 정보를 융합한다.

**장점**
- 객체 배치의 자유도가 높다.
- 참조 이미지를 스타일뿐만 아니라 구체적 이미지로 활용할 수 있다.
- 복잡한 장면 구성이 가능하다.

**한계**
- 바운딩 박스 지정이 필요하여 사용자 편의성이 떨어진다.
- 객체 간 상호작용 표현에 한계가 있다.

## **4. Training-free Editing via Attention/Feature Control**

### **4.1 Prompt-to-Prompt (2022)**

**"Prompt-to-Prompt Image Editing with Cross-Attention Control"**

![](/figures/image_editing/prompt-to-prompt.png)

Prompt-to-Prompt는 텍스트 프롬프트의 일부를 바꾸되, diffusion 과정에서 **cross-attention map을 주입/고정**해 이미지의 **레이아웃을 보존하는 training-free** 편집 기법이다. 텍스트만으로 국소/전역 편집을 하면서도 원본의 구조를 유지한다는 문제의식을 정면으로 다룬 대표 작업이다.

**핵심**
- 원본 이미지의 cross-attention map을 저장한다.
- 편집 프롬프트에서 변경할 부분만 수정하고, 나머지는 원본과 동일하게 유지한다.
- 저장된 attention map을 활용하여 레이아웃을 보존하면서 선택적 편집을 수행한다.

**장점**
- 이미지 구조를 크게 변경하지 않고 부분 편집이 가능하다.
- Inversion 과정이 필요 없어 효율적이다.

**한계**
- 복잡한 편집 시나리오에서는 attention map만으로는 부족하다.
- 객체 제거나 대규모 변경에는 한계가 있다.

### **4.2 Plug-and-Play Diffusion Features (PnP) (2023)**

**"Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation"**

![](/figures/image_editing/pnp.png)

PnP는 모델 가중치를 바꾸지 않고, denoising 과정에서 **source image의 중간 feature를 주입(feature injection)** 하여 **구조/레이아웃을 강하게 고정**한 상태로 텍스트로 의미·스타일을 바꾸는 대표적인 **training-free I2I** 프레임워크다. "편집은 하고 싶은데 원본 구도를 무너뜨리고 싶지 않다"는 요구에 매우 직관적으로 대응한다.

**핵심**
- source image를 inversion해 denoising trajectory(또는 latent)를 확보한다.
- target prompt로 생성할 때, 특정 레이어/스텝에서 **source feature를 주입**해 구조를 고정한다.
- 주입 강도/레이어 선택으로 "얼마나 원본을 고정할지"를 조절한다.

**장점**
- 추가 학습 없이도 **구조 보존이 매우 강함**.
- "스타일 변경 / 재질 변화 / 분위기 변경" 같은 편집에서 특히 강력하다.
- inversion 계열과 결합할수록 실사 편집 성능이 안정적으로 올라간다.

**한계**
- feature 주입이 강할수록 **편집 자유도**가 줄어들 수 있다(보존↔변형 trade-off).
- 객체 단위 정밀 편집(특정 객체만 교체/제거)에는 마스크/세그멘트 등의 보조 신호가 필요해지는 경우가 많다.

### **4.3 MasaCtrl (2023)**

**"MasaCtrl: Mutual Self-Attention Control for Consistent Image Synthesis and Editing"**

![](/figures/image_editing/masactrl.png)

MasaCtrl은 diffusion 과정에서 **self-attention을 상호 제어(mutual self-attention control)** 하여, 특히 **non-rigid 변형(포즈 변화, 동물/사람 자세 변화 등)** 상황에서도 편집 일관성을 강화한 기법이다. Prompt-to-Prompt가 cross-attention 중심이라면, MasaCtrl은 **self-attention 제어**를 통해 "구조적 일관성/세부 정합성"을 더 직접적으로 다룬다.

**핵심**
- 원본 생성/편집 과정에서 self-attention map을 추적한다.
- 편집 생성 시 self-attention을 상호 참조/고정/혼합하여, 특정 영역/객체의 정합성을 유지한다.
- 결과적으로 텍스트 변경에 따른 의미 변화는 반영하되, 원본의 공간적/형태적 일관성을 보존한다.

**장점**
- 포즈 변화나 비강체(non-rigid) 편집에서 **일관성 유지**가 강하다.
- 추가 학습 없이 적용 가능해 plug-and-play 성격이 강하다.
- 기존 attention 제어 계열(P2P 등)과 비교해 "형태/구조" 보존 측면에서 장점이 있다.

**한계**
- attention map 품질/안정성에 민감하며, 복잡 장면에서는 제어가 불안정할 수 있다.
- 객체 제거/정교한 인페인팅 같은 "내용 삭제/복원" 계열에는 마스크 기반 방법이 더 적합한 경우가 많다.

## **5. Real Image Editing via Inversion/Optimization**

### **5.1 Null-text Inversion (2022)**

**"Null-text Inversion for Editing Real Images using Guided Diffusion Models"**

![](/figures/image_editing/null_text_inversion.png)

Null-text Inversion은 diffusion 기반 실사 편집에서 가장 큰 병목인 **"원본 이미지를 얼마나 정확히 재구성(inversion)하느냐"**를 크게 개선한 기법이다. 기존 inversion이 프롬프트/CFG 설정에 민감하고 재구성 품질이 낮으면, 편집 시 원본 구조가 쉽게 붕괴되는 문제가 있었다. Null-text Inversion은 **CFG의 unconditional branch(=null-text embedding)** 를 이미지별로 최적화해, **reconstruction fidelity**를 끌어올리면서도 편집 여지를 유지한다.

**핵심**
- 원본 이미지를 DDIM inversion 등으로 latent trajectory로 되돌린다.
- 각 denoising step에서 **unconditional embedding(null-text)** 을 이미지별로 최적화해 재구성 오차를 최소화한다.
- 편집 시에는 conditional prompt를 바꾸되, 최적화된 unconditional을 활용해 **구조/정체성을 유지**하면서 변화만 반영한다.

**장점**
- 실사 이미지에서 재구성이 강해져 **편집 안정성(구조 보존)** 이 크게 향상된다.
- 추가 학습 없이도 이미지별 최적화만으로 성능을 얻는 **training-free** 접근이다.
- Prompt-to-Prompt 같은 attention 제어 기법과 결합했을 때도 효과가 좋다.

**한계**
- 이미지별 최적화가 필요해 **추론 시간이 증가**한다(online optimization).
- 매우 큰 편집(대규모 레이아웃 변경, 객체 완전 교체)에서는 여전히 원본 보존/변형 트레이드오프가 존재한다.

### **5.2 Imagic (2023)**

**"Imagic: Text-Based Real Image Editing with Diffusion Models"**

![](/figures/image_editing/imagic.png)

Imagic은 실제 사진을 기반으로 **잠재 공간을 최적화**해 텍스트 편집을 가능하게 하는 기법이다. 결과 이미지는 기존 사진과 매우 유사하면서도 텍스트 지시 사항을 반영한다.

**핵심**
- 원본 이미지를 latent space로 인코딩한다.
- 텍스트 임베딩과 latent를 함께 최적화하여 편집 방향을 찾는다.
- 최적화된 latent를 디코딩하여 최종 이미지를 생성한다.

**장점**
- 고해상도 실사 이미지 편집에 적합하다.
- 사용자 맞춤형 편집에 강력하다.
- 원본 이미지의 품질을 크게 손상시키지 않는다.

**한계**
- 최적화 과정이 시간이 오래 걸린다.
- 복잡한 편집에는 여러 번의 최적화가 필요하다.

## **6. Instruction-following Editing**

### **6.1 InstructPix2Pix (2023)**

**"InstructPix2Pix: Learning to Follow Image Editing Instructions"**

![](/figures/image_editing/instructpix2pix.png)

InstructPix2Pix는 "prompt engineering"을 넘어 **자연어 지시(instruction)** 로 이미지를 편집하는 패러다임을 대중화했다. 특히 핵심은 **대규모 편집 데이터셋을 사람이 라벨링하지 않고 합성으로 만든 뒤**, 그 데이터로 모델을 학습해 **one-shot forward pass 편집**을 가능하게 만든 점이다. 즉, 온라인 최적화나 inversion 품질에 덜 의존하는 방향으로 편집을 확장했다.

**핵심**
- (데이터 생성) 입력 이미지에 대해 LLM이 편집 지시문을 만들고, T2I/이미지 편집 모델로 "편집 결과"를 합성해 (input, instruction, output) 트리플을 구성한다.
- (학습) 해당 트리플로 conditional diffusion 모델을 학습해, 추론 시에는 **입력 이미지 + instruction** 만으로 편집을 수행한다.

**장점**
- 사용자는 "프롬프트"가 아니라 **명령형 문장**으로 직관적인 편집이 가능하다.
- online optimization 없이 빠르게 동작하고, 다양한 편집 타입을 하나의 모델로 커버하기 쉽다.
- 이후의 instruction-following 편집/멀티모달 에이전트 편집으로 연결되는 기반을 만든다.

**한계**
- 합성 데이터 기반이라, **편집 충실도(Instruction fidelity)** 와 **원본 보존** 사이에서 편향이 생길 수 있다.
- 특정 객체 정체성 보존(subject preservation)이나 복잡한 관계 편집에서는 여전히 품질 편차가 크다.
- "어떤 부분을 바꿀지"가 명확하지 않으면 불필요한 영역까지 변형될 수 있다(implicit mask 문제).

## **7. Consistency & Inversion-free Editing**

### **7.1 PixelMan (2024)**

**"PixelMan: Consistent Object Editing with Diffusion Models via Pixel Manipulation and Generation"**

![](/figures/image_editing/pixelman.png)

PixelMan은 **inversion-free이고 training-free인** diffusion 기반 이미지 편집 기법으로, 텍스트 프롬프트 없이도 이미지 내 객체를 픽셀 수준으로 복제·이동·삽입할 수 있다.

**핵심**
- 픽셀 공간에서 객체의 복사본을 생성하고, 효율적인 샘플링 방식을 사용하여 조작된 객체를 목표 위치에 자연스럽게 통합한다.
- Diffusion step 중간에 **pixel-wise manipulation**을 직접 수행한다.
- 객체의 구조와 질감을 그대로 유지한 상태로 새로운 위치에 자연스럽게 배치한다.
- Inversion 없이도 정확도 높은 복제 편집을 지원하며, 단 16 inference step만으로도 우수한 결과를 달성한다.

**장점**
- Inversion 과정이 필요 없고 추가 학습도 필요 없어 빠르고 효율적이다.
- 텍스트 프롬프트 없이도 직관적인 편집이 가능하다.
- 객체 구조와 질감을 정확히 보존하며 이미지 일관성을 유지한다.
- 기존 방법들이 50 step을 필요로 하는 반면, 16 step만으로도 우수한 성능을 보인다.

**한계**
- 복잡한 장면에서는 객체 간 상호작용 표현이 어렵다.
- 배경 일관성 유지에 한계가 있다.

### **7.2 Edicho (2024)**

**"Edicho: Consistent Image Editing in the Wild"**

![](/figures/image_editing/edicho.png)

Edicho는 **training-free diffusion 기반 방법**으로, 여러 in-the-wild 이미지에서 일관된 이미지 편집을 수행한다. 핵심 설계 원칙은 암묵적 attention feature에 의존하기보다는 **명시적 이미지 대응 관계(explicit image correspondence)**를 사용하여 편집을 지시하는 것이다.

**핵심**
- **Attention manipulation module**과 **refined classifier-free guidance (CFG) denoising strategy**를 사용한다.
- 두 구성 요소 모두 사전에 추정된 이미지 간 대응 관계를 활용한다.
- 이미지 간의 **cross-image correspondence**를 활용해 객체 삽입, 제거, 위치 이동 등을 자연스럽게 처리한다.
- 다양한 객체 포즈, 조명 조건, 촬영 환경의 변화에도 robust하게 작동한다.

**장점**
- **Plug-and-play 호환성**: ControlNet, BrushNet 등 대부분의 diffusion 기반 편집 방법과 호환된다.
- **Zero-shot capability**: 다양한 설정에서 zero-shot으로 작동한다.
- 복수 이미지 편집을 수행해야 하는 상황(예: before-after 비교, 동일 객체 위치 변경 등)에 뛰어난 결과를 보인다.
- 사용자 개입 없이도 객체 배치가 논리적으로 정합된다.
- 동일한 장면의 서로 다른 시점에서 일관된 편집 결과를 생성할 수 있다.

**흥미로운 확장 가능성**
- 이미지 간 key feature matching을 통해 3D reconstruction(카메라 파라미터 없이)까지 가능하다는 연구 결과가 있다.
- 아직까지는 체리피킹이지만, 이 기술이 잘 발전하면 3D reconstruction을 위한 학습 데이터셋 생성에도 활용할 수 있을 것으로 기대된다.

**한계**
- 두 이미지 간의 대응 관계가 명확하지 않으면 성능이 저하된다.
- 복잡한 장면에서는 계산 비용이 증가한다.

## **8. Intrinsic-based Editing**

### **8.1 IntrinsicEdit (2025)**

**"IntrinsicEdit: Precise generative image manipulation in intrinsic space"**

![](/figures/image_editing/intrinsicedit.png)

IntrinsicEdit은 intrinsic-image latent space에서 작동하는 생성적 워크플로우를 도입한다. 이미지를 shape, albedo(reflectance), lighting 구성 요소로 분해하여 정밀한 픽셀 수준 조작을 가능하게 한다. **RGB-X diffusion framework**를 기반으로 하며, identity preservation과 channel entanglement 문제를 해결한다.

**핵심**
- 이미지를 **intrinsic 채널**(shape, albedo, lighting)로 분해한다.
- **Exact diffusion inversion**과 **disentangled channel manipulation**을 통해 필요한 채널만 선택적으로 편집한다.
- 편집된 채널들을 다시 조합하여 최종 이미지를 생성한다(re-rendering).
- 추가 데이터 수집이나 모델 파인튜닝 없이도 자동으로 전역 조명 효과를 해결한다.

**장점**
- 전체 이미지가 아닌 **조명·재질·부분 구조** 등 세부적 컨트롤이 가능하다.
- 색상 및 텍스처 조정, 객체 삽입 및 제거, 전역 재조명 등 다양한 편집 작업을 정밀하게 수행할 수 있다.
- 기존 이미지를 크게 손상시키지 않고도 매우 정교한 스타일 변화나 객체 속성 조절이 가능하다.
- 고품질 이미지 보존에 강점을 가진다.

**한계**
- Intrinsic 분해 과정이 복잡하고 계산 비용이 높다.
- 모든 이미지에 대해 정확한 intrinsic 분해가 가능한 것은 아니다.

## **9. Research Directions**

본 리포트에서 다루는 시점(2022–2025년 초)까지의 이미지 편집 연구는 다음과 같은 방향으로 발전해왔다.

### **9.1 Inversion-free & Low-prompt 편집**

기존 이미지를 latent space로 되돌리는 inversion 과정의 부담을 줄이거나, inversion 없이도 편집을 수행하는 방향이 강화되었다. PixelMan은 inversion-free로 객체 조작을 수행하며 편집 속도와 효율을 크게 향상시켰다. Edicho 역시 이미지 간 명시적 대응 관계(correspondence)를 활용하는 방식으로, 프롬프트/attention에 대한 암묵적 의존을 낮추고 다양한 환경(in-the-wild)에서의 일관성을 강화하려는 흐름을 보여준다.

### **9.2 구조 일관성 유지 + 멀티 이미지 정합성**

편집 과정에서 단일 이미지 품질뿐 아니라, 여러 이미지(또는 여러 뷰/상황) 간의 정합성을 함께 보장하는 방향으로 발전했다. Edicho는 cross-image correspondence를 활용해 복수 이미지 편집 시나리오에서 일관된 결과를 생성할 수 있음을 보여준다.

### **9.3 속성(재질·조명) 단위 정밀 편집**

편집 대상이 “객체 자체”뿐 아니라, **조명·재질·반사·색감** 같은 속성으로 확장되며 더 정교한 제어가 가능해졌다. IntrinsicEdit은 intrinsic space(예: shape/albedo/lighting)로 분해해 필요한 채널만 조작함으로써, 픽셀 수준 품질을 유지하면서도 전역 재조명이나 재질 변화 같은 편집을 정밀하게 수행하려는 시도를 보여준다.

### **9.4 Grounded/Spatial Control의 정밀화**

텍스트만으로는 제어하기 어려운 위치·크기·형태 정보를 외부 조건(edge/depth/pose/box 등)으로 주입해, “원하는 곳에 원하는 변화를” 만들려는 방향이 강화되었다. ControlNet과 GLIGEN은 구조·레이아웃 제어를 정교화하며, 편집을 보다 설계 가능한(problem-formulatable) 형태로 만드는 흐름을 대표한다.

### **9.5 Training-free 편집의 확장(Attention/Feature Control)**

추가 학습 없이도 편집을 수행하는 방향이 강해졌다. Prompt-to-Prompt, PnP, MasaCtrl처럼 attention/feature를 제어해 **레이아웃·형태 정합성**을 유지하면서 편집하는 흐름이 등장했고, 이는 “가볍게 적용 가능한 편집”의 실용성을 크게 높였다. 다만 보존 강도를 높일수록 편집 자유도가 줄어드는 trade-off는 여전히 남아 있다.

### **9.6 프롬프트에서 인스트럭션으로(UI의 변화)**

InstructPix2Pix 이후 편집 인터페이스는 “프롬프트 엔지니어링” 중심에서 **자연어 지시를 따르는 인스트럭션 기반**으로 이동했다. 이는 사용성 측면에서 큰 진전이지만, 합성 데이터 기반 학습이 만드는 편향(불필요한 영역 변형, 충실도/보존의 불안정성)을 어떻게 제어할지는 계속 중요한 연구 과제로 남아 있다.

---

이미지 편집 기술은 2022년 개인화(Concept Injection)와 함께 실사 편집을 위한 inversion/최적화 기반이 다져졌고, 2023년에는 조건 기반 제어(grounded control)와 training-free 편집, 인스트럭션 기반 편집이 본격적으로 확산되며 편집 인터페이스와 적용 범위가 크게 확장되었다. 이후 2024–2025로 오면서는 멀티 이미지 정합성과 in-the-wild 일관성(Consistency), 그리고 조명·재질 등 속성 단위의 정밀 제어(Intrinsic space)로 관심이 이동하며 “보존-편집 분리”를 더 명시적으로 다루는 방향으로 발전해왔다.

다만, 본 리포트의 범위를 넘어서는 최근 흐름에서는 **multi-reference 편집**, **더 정밀한 객체 단위 제어**, 그리고 **대규모 스케일 아웃 모델**(나노바나나, GPT Image, z image, FLUX.2 등)이 등장하며 이미지 편집은 새로운 단계로 진입하고 있다.