---
title: "이미지 편집 기술 최신 연구 동향 (2022-2025)"
date: 2025-12-01
categories:
  - AI Reports
tags:
  - Image Editing
  - Diffusion Models
  - Computer Vision
category: image-video-generation
---

최근 diffusion 모델의 발전과 함께 이미지 편집 기술은 급속도로 발전하고 있다. 단순한 필터 적용을 넘어, 텍스트 기반 편집, 객체 삽입/제거, 구조 보존 편집 등 다양한 편집 시나리오를 지원하는 방법들이 등장했다. 특히 최근에는 나노바나나, GPT Image, z image, FLUX.2 등 최신 이미지 편집 모델들이 공개되며 범용 에디팅 모델에 대한 기대도 커지고 있다. 다만 본 리포트는 이러한 최신 모델 자체를 비교하기보다는, **2022년부터 2025년 초까지** 공개된 대표 연구들을 통해 이미지 편집 기술이 어떤 문제의식과 설계 선택을 따라 발전해왔는지를 정리한다. 핵심은 이미지 편집에서 **무엇을 보존하고(정체성/구조/배경), 무엇을 바꿀지(속성/객체/레이아웃)**를 어떻게 분리해 다루는가이다.

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

Textual Inversion은 특정 개체/스타일 같은 “새로운 개념”을 **단 하나(또는 소수)의 텍스트 임베딩 벡터**로 표현하도록 학습하는 personalization 기법이다. DreamBooth가 diffusion 모델의 가중치까지 업데이트해 개념을 더 강하게 주입하는 반면, Textual Inversion은 **모델 가중치는 freeze** 한 채 **텍스트 인코더의 특정 토큰 임베딩만 최적화**한다. 즉, “@my_dog” 같은 새 토큰을 프롬프트에 넣었을 때 모델이 그 토큰을 특정 시각적 개념(대상/스타일)로 해석하도록 텍스트 공간을 조정하는 방식이다.

이 방식은 editing에서도 유효한데, 편집을 할 때 “원본 이미지의 정체성(특정 인물/상품)을 유지한 채 배경·속성만 바꾸기” 같은 요구가 자주 등장하기 때문이다. Textual Inversion으로 학습한 토큰은 프롬프트 기반 편집(예: 기존 이미지를 기반으로 한 img2img/inpainting)에서 **대상의 아이덴티티를 프롬프트로 안정적으로 고정**하는 역할을 한다. 다만 임베딩만으로 표현하는 만큼, 세부 디테일이나 복잡한 개체를 완벽히 재현하는 능력은 DreamBooth 대비 제한적일 수 있다.

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

ControlNet은 기존 Stable Diffusion의 UNet을 크게 바꾸지 않으면서, edge/depth/pose 같은 **픽셀-정렬 조건(condition hint)**을 denoising 과정에 주입해 생성 결과의 구조를 강하게 제어하는 방법이다. 핵심은 기존 T2I UNet은 그대로 두고(보통 freeze), condition을 처리하는 branch 네트워크를 추가해 UNet의 중간 feature에 condition 신호를 더해준다는 점이다.

**핵심**
- 입력 조건(예: canny edge, depth, pose)은 별도 인코더/컨볼루션 블록을 통해 처리되어, UNet의 여러 해상도 단계(resolution stage)에 대응하는 **control feature**로 변환된다.
- ControlNet branch는 구조적으로 UNet과 유사한 블록을 갖고, 각 stage마다 생성된 control feature를 원본 UNet의 대응 stage에 **residual 형태로 주입**한다.
- 이 주입은 개념적으로는 `h_l <- h_l + α * c_l` 같은 **element-wise add**로 이해해도 무방하다. (실제 구현은 stage/블록마다 주입 위치와 스케일링이 조금씩 다를 수 있지만, 핵심은 “원본 feature에 control residual을 더한다”는 구조)

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

GLIGEN은 기존 diffusion pipeline에 **grounded generation**을 위한 **Object Injection Module**을 추가해, 텍스트로 무엇을 만들지뿐 아니라 어디에(바운딩 박스) 무엇을(객체/개념) 넣을지를 동시에 제어하는 기법이다. 핵심은 단순히 박스 위치를 조건으로 주는 수준을 넘어, **박스별(region-wise)로 텍스트/참조 정보를 UNet의 토큰/feature에 주입**해 해당 영역에서의 생성 과정을 직접 가이드한다는 점이다. 결과적으로 open-set 객체 배치와 장면 구성에 강하다.

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

Prompt-to-Prompt는 텍스트 프롬프트의 일부를 바꾸되, diffusion 과정에서 **cross-attention map을 주입/고정**해 이미지의 **레이아웃을 보존하는 training-free** 편집 기법이다. Training-free임에도 편집 성능이 강해지는 이유는, diffusion 모델에서 **cross-attention이 사실상 텍스트 토큰이 이미지 공간에 배치되는 ‘레이아웃 신호’**로 작동하기 때문이다. 일반적으로 프롬프트를 바꾸면 attention 패턴이 함께 흔들리면서 구도까지 변형되기 쉬운데, P2P는 원본 생성 과정에서 얻은 **token별 cross-attention map을 저장**한 뒤, 편집 시 “바꾸지 말아야 할 토큰”의 attention을 **고정/재주입**한다. 결과적으로 모델을 재학습하지 않고도, 동일한 UNet으로 **레이아웃(구조)은 유지한 채 의미 변화만 국소적으로 반영**되도록 추론 과정을 제어한다.

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

Null-text Inversion은 diffusion 기반 실사 편집에서 가장 큰 병목인 **원본 이미지를 얼마나 정확히 재구성(inversion)하느냐**를 크게 개선한 기법이다. 실사 편집은 보통 (1) 원본 이미지를 diffusion의 latent trajectory로 되돌린 뒤(inversion), (2) 프롬프트를 바꿔 denoising을 다시 수행하는 방식으로 진행된다. 그런데 이 과정에서 inversion 단계에서 원본이 조금만 틀어져도(재구성 오차), 편집 단계에서 그 오차가 누적되어 **질감이 무너지거나, 배경/구조가 흔들리거나, 원치 않는 영역이 변형**되는 문제가 쉽게 발생한다. 특히 CFG(Classifier-Free Guidance)를 사용하는 경우, conditional/unconditional 두 경로의 조합이 민감하게 작동해 원본 재구성과 편집 자유도를 동시에 만족시키기 어렵다.

Null-text Inversion의 핵심은 inversion을 할 때 **latent를 과도하게 최적화하기보다**, CFG에서 사용되는 **unconditional branch의 텍스트 임베딩(=null-text embedding)** 을 이미지별·스텝별로 최적화한다는 점이다. 직관적으로는 “편집에 사용하는 프롬프트(conditional)는 유지하되, 원본을 정확히 재현할 수 있도록 **unconditional 쪽을 이미지에 맞게 보정**해 주는 것”에 가깝다. 이렇게 얻은 step-wise null-text는 재구성 fidelity를 크게 끌어올리면서도, 이후 편집 단계에서는 conditional prompt만 바꿔 **원본 구조는 붙잡고 변화만 유도**할 수 있게 해준다.

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

Imagic은 실제 사진을 기반으로 **latent와 텍스트 임베딩을 함께 최적화**해 텍스트 기반 편집을 수행하는 기법이다. 단순히 프롬프트를 바꾸는 것만으로는 실사 편집에서 원본 구조(배경·구도·정체성)가 쉽게 흔들리는데, Imagic은 먼저 프롬프트/임베딩 조합이 원본 이미지를 가장 잘 재현하도록 최적화된 표현을 찾은 뒤, 그 지점에서 편집 방향으로 이동하는 방식으로 **원본 보존과 편집 반영을 동시에** 노린다.

구체적으로는 (1) 주어진 원본 이미지에 대해 **텍스트 임베딩을 이미지에 맞게 보정**하여 재구성 fidelity를 높이고, (2) 편집 프롬프트를 적용할 때는 원본에서 크게 벗어나지 않도록 **보정된 임베딩과 latent를 기준점(anchor)으로 유지**하며 denoising을 진행한다. 결과적으로 원본에 최대한 붙어 있으면서도, 지시된 의미 변화만 반영되는 편집을 만들기 쉬워 실사 사진 편집에서 안정적인 결과를 낸다.

다만 최적화 기반 접근이라 이미지마다 반복 최적화가 필요해 추론 시간이 늘고, 변화 폭이 큰 편집(대규모 객체 교체/레이아웃 변경)에서는 보존-변형 trade-off가 여전히 존재한다.

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

InstructPix2Pix는 prompt engineering을 넘어 **자연어 지시(instruction)** 로 이미지를 편집하는 패러다임을 대중화했다. 이 작업의 핵심은 편집을 추론 시점의 최적화(inversion/online optimization) 문제로 두지 않고, **[입력 이미지, 지시문, 편집 결과]** 형태의 대규모 학습 데이터로 **instruction-following 편집 모델을 직접 학습**했다는 점이다. 특히 사람이 픽셀 단위로 편집 정답을 라벨링하기 어렵다는 현실적 제약을, **기성 T2I 모델과 자동 생성된 지시문(LLM 기반)을 이용한 합성 데이터 생성**으로 우회했다.

모델은 학습을 통해 “원본을 가능한 유지하면서 지시된 변화만 적용”하는 편집 규칙을 내재화하고, 추론 시에는 **단 한 번의 forward pass**로 편집 결과를 생성한다. 이로써 결과 품질이 inversion 품질이나 최적화 안정성에 덜 좌우되며, 다양한 편집 타입을 하나의 자연어 지시로 커버할 수 있게 되었다. 즉 InstructPix2Pix는 편집의 병목을 inversion을 얼마나 잘하느냐에서 instruction-following 데이터/학습을 얼마나 잘하느냐로 초점을 바꾼 대표 사례다.

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

PixelMan은 **inversion-free + training-free** 설정에서, 텍스트 프롬프트 없이도 이미지 내 객체를 **복제/이동/삽입**할 수 있도록 설계된 diffusion 기반 편집 기법이다. 핵심은 객체 편집의 본질은 새로 그리는 게 아니라, **이미지 안에 이미 존재하는 픽셀/패턴을 원하는 위치에 옮기고 자연스럽게 blending하는 것**이라는 점이다. PixelMan은 이 과정을 diffusion denoising 중간 단계에서 직접 수행해, 별도의 inversion이나 추가 학습 없이도 높은 구조 일관성을 확보한다.


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

Edicho는 **training-free diffusion 기반**으로, in-the-wild 환경(포즈/조명/배경/카메라 시점이 제각각인 이미지들)에서도 **일관된 편집 결과**를 얻는 것을 목표로 한다. 핵심 설계 원칙은 모델 내부 attention이 알아서 정합성을 맞춰주길 기대하는 것 대신, **이미지 간의 명시적 대응 관계(explicit correspondence)** 를 먼저 추정하고 그 정보를 denoising 과정에 직접 주입해 **무엇을 유지하고 어디를 바꿔야 하는지**를 안정적으로 고정한다는 점이다. 즉, Edicho는 편집을 단순한 single-image I2I 문제가 아니라, **cross-image consistency** 제약이 있는 문제로 명시화한다.

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

본 리포트에서 다루는 시점(2022–2025년 초)까지의 이미지 편집 연구는, “**원본 보존**(fidelity/identity/structure)과 **편집 반영**(editability/instruction fidelity)을 어떻게 분리·제어할 것인가”라는 축을 중심으로 빠르게 전개되었다. 2022년에는 **개인화(Concept Injection)** 와 **실사 편집을 위한 inversion/최적화**가 편집의 출발점이 되었고, 2023년에는 **공간적 제어(grounded/spatial control)** 와 **training-free 제어(attention/feature control)** 로 “원하는 변화만” 설계 가능하게 만드는 방향이 강화되었다. 2024–2025로 오면서는 단일 이미지 편집을 넘어서 **멀티 이미지 정합성(consistency)**, **in-the-wild 강건성**, 그리고 **조명·재질 같은 속성 단위의 분해/조작(intrinsic)** 으로 초점이 이동하며, 편집을 “픽셀 단위 결과”가 아니라 “제약을 만족하는 생성 과정”으로 다루는 경향이 뚜렷해졌다.

아래는 이러한 흐름을 대표하는 연구 방향들이다.

### **9.1 Minimal Inversion & Lightweight Editing**

실사 편집에서 inversion은 고품질 보존을 위한 강력한 도구였지만, 이미지별 최적화 비용과 민감도(CFG/step/프롬프트)에 의해 **확장성의 병목**이 되었다. 이에 따라 연구는 inversion을 “필요하면 쓰되 최소화”하거나, 아예 inversion 없이도 편집을 가능하게 하는 방향으로 발전했다.  
- **PixelMan**은 inversion-free로 객체 복제/이동/삽입을 수행하며, diffusion을 “새로 그리기”보다 “자연스러운 융합(harmonization)”에 활용해 효율을 끌어올렸다.  
- **Edicho**는 프롬프트/attention에 대한 암묵적 의존을 줄이고, 이미지 간 **명시적 correspondence**를 제어 신호로 사용해 in-the-wild에서의 안정성을 강화한다.  
결과적으로 “편집을 하려면 inversion이 필수”라는 전제를 약화시키고, **경량·고속·대규모 적용**을 가능하게 하는 흐름이 형성되었다.

### **9.2 Cross-Image Consistency**

초기 연구가 “단일 이미지에서 구조를 유지한 채 편집”에 집중했다면, 후반부로 갈수록 **여러 이미지(여러 컷/여러 뷰/여러 조건)** 에서 동일한 객체·장면의 정체성을 유지하는 문제가 핵심으로 떠올랐다. 이는 실제 제품(앨범 편집, 쇼핑/인테리어, 크리에이티브 워크플로우)에서 매우 빈번한 요구이기도 하다.  
- **Edicho**는 cross-image correspondence로 “무엇을 동일하게 유지해야 하는지”를 명시화하고, 이를 diffusion 과정의 제약으로 반영해 일관성을 확보하려는 대표 사례다.  
이 방향은 이후 **multi-reference 편집**, 시퀀스/비디오 정합성, 편집 전후의 “동일성 보장”으로 자연스럽게 확장된다.

### **9.3 Attribute-Level Editing (Lighting/Material)**

편집 대상이 객체의 존재/형태를 넘어 **조명·재질·반사·색감** 같은 속성으로 확장되며 “무엇을 바꾸고 무엇을 보존할지”를 더 세밀하게 나누려는 시도가 등장했다. 이는 단순한 마스크 기반 국소 편집을 넘어, **장면의 물리적/시각적 일관성**까지 다루려는 흐름이다.  
- **IntrinsicEdit**은 intrinsic space(예: shape/albedo/lighting)로 분해해 필요한 채널만 조작하고 재합성함으로써, 픽셀 수준 품질을 유지한 채 전역 재조명·재질 변화 같은 편집을 정밀하게 수행하려는 방향을 보여준다.  
이 방향은 “편집=픽셀 수정”에서 “편집=원인(속성/채널) 조작”으로 관점을 확장한다.

### **9.4 Grounded Spatial Control**

텍스트만으로는 위치·크기·형태 같은 공간 제약을 안정적으로 통제하기 어렵다. 따라서 edge/depth/pose/box 등 **외부 조건을 명시적으로 주입**해 “원하는 곳에 원하는 변화”를 재현하려는 방향이 강화되었다.  
- **ControlNet**은 픽셀 정렬 조건(edge/depth/pose 등)을 UNet에 잔차 형태로 주입해 구조 보존을 강화했고,  
- **GLIGEN**은 box 단위로 객체/개념을 주입하는 grounded generation을 통해 “공간적으로 설계 가능한 생성”을 가능하게 했다.  
이 흐름은 이후 객체 단위 제어, 레이아웃 계획, 관계 제약(가림/접촉) 등 **장면 구성 문제**로 확장된다.

### **9.5 Training-Free Control (Attention/Features)**

모델을 다시 학습시키지 않고도(inference-time), 내부 신호(attention/feature)를 조작해 편집 안정성을 높이려는 흐름이 커졌다. 이는 개발/운영 관점에서 비용이 낮고 적용이 빠르며, 기존 모델 생태계(Stable Diffusion 계열)에 쉽게 붙일 수 있다는 장점이 있다.  
- **Prompt-to-Prompt**는 cross-attention map의 고정/재주입으로 레이아웃 붕괴를 줄였고,  
- **PnP**는 source feature injection으로 구조를 강하게 고정했으며,  
- **MasaCtrl**은 self-attention 제어로 non-rigid 변형에서도 정합성을 강화했다.  
다만 이 계열은 공통적으로 **보존을 강하게 걸수록 편집 자유도가 줄어드는 trade-off**가 남아 있으며, “정교한 객체 단위 편집”에는 마스크/grounding과의 결합이 필요해지는 경우가 많다.

### **9.6 Instruction-Following Editing**

InstructPix2Pix 이후 편집 인터페이스는 프롬프트 엔지니어링 중심에서 **자연어 지시(instruction) 기반**으로 이동했다. 이는 사용성을 크게 개선했을 뿐 아니라, 편집을 inversion/최적화 중심의 파이프라인에서 **학습 기반 one-shot 편집 문제**로 전환시켰다는 점에서 중요하다.  
- 합성 데이터로 대규모 (input, instruction, output) 트리플을 구성해 학습함으로써, 추론 시 “한 번의 forward pass”로 다양한 편집을 수행할 수 있게 했다.  
- 반면 합성 데이터 편향으로 인한 **불필요 영역 변형(implicit mask 문제)**, **instruction fidelity vs 보존의 불안정성**, **정체성 보존의 한계**는 여전히 중요한 연구 과제로 남아 있다.  
이 방향은 이후 멀티모달 에이전트 편집, 편집 계획(planning), 다단계 제약 만족형 편집으로 자연스럽게 이어진다.있다.

---

이미지 편집 기술은 2022년 개인화(Concept Injection)와 함께 실사 편집을 위한 inversion/최적화 기반이 다져졌고, 2023년에는 조건 기반 제어(grounded control)와 training-free 편집, 인스트럭션 기반 편집이 본격적으로 확산되며 편집 인터페이스와 적용 범위가 크게 확장되었다. 이후 2024–2025로 오면서는 멀티 이미지 정합성과 in-the-wild 일관성(Consistency), 그리고 조명·재질 등 속성 단위의 정밀 제어(Intrinsic space)로 관심이 이동하며 “보존-편집 분리”를 더 명시적으로 다루는 방향으로 발전해왔다.

다만, 본 리포트의 범위를 넘어서는 최근 흐름에서는 **multi-reference 편집**, **더 정밀한 객체 단위 제어**, 그리고 **대규모 스케일 아웃 모델**(나노바나나, GPT Image, z image, FLUX.2 등)이 등장하며 이미지 편집은 새로운 단계로 진입하고 있다.