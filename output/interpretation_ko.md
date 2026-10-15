# DINOv2 ViT-S/14 × Fish4Knowledge (4종) 결과 해석

실험 조건: Fish4Knowledge 공식 GT/RECOG 클러스터 01–04, 종당 300장(seed=42), 총 1200장.
DINOv2 ViT-S/14는 freeze 상태로 학습 없이 CLS 토큰을 쓰고, 384차원 벡터를 L2 정규화해 분석함.

| 지표 | 값 |
|---|---|
| Silhouette (cosine, 384-d) | **0.127** (euclidean 0.073) |
| 최근접 중심 일치율 (leave-one-out) | D.ret 0.873 / P.dic 0.947 / C.chr 0.900 / A.cla 0.947 |
| 10-NN 다수결 일치율 (같은 궤적 이웃 제외) | **0.947** |
| PCA PC1+PC2 설명 분산 | 23.7% (13.7% + 10.0%) |
| 평균 벡터 cosine 최댓값 | P. dickii ↔ C. chrysura 0.901 |

## 1. 4개 어종이 각각 독립적인 cluster를 형성하는가?
부분적으로만 그렇다.
- A. clarkii는 t-SNE와 UMAP 모두에서 나머지 3종과 완전히 떨어진 영역에 있다.
- 나머지 3종(D. reticulatus, P. dickii, C. chrysura)은 각자 밀집 영역이 있지만, 경계가 맞닿아 하나의 연속된 띠(UMAP)를 이룬다.
- PCA에서는 A. clarkii만 PC1 축으로 분리되고, 나머지 3종은 크게 겹친다.

## 2. 가장 많이 겹치는 두 어종은?
**Plectroglyphidodon dickii ↔ Chromis chrysura**
- 평균 벡터 cosine 0.901로 가장 높다.
- 10-NN에서 P. dickii 이웃의 11.0%가 C. chrysura다.
- D. reticulatus → C. chrysura도 10.2%로 비슷한 수준이다.

## 3. 가장 명확하게 분리되는 어종은?
**Amphiprion clarkii**
- 10-NN 이웃의 97.3%가 같은 종이다.
- 종별 silhouette이 0.247로 가장 높다.
- 다른 종과의 평균 벡터 cosine이 0.71–0.74로 가장 낮다.

## 4. 같은 어종 내부에서도 여러 cluster가 생기는가?
그렇다.
- A. clarkii는 t-SNE와 UMAP 모두에서 큰 덩어리 하나와 작은 덩어리 하나로 갈라진다.
- D. reticulatus는 t-SNE에서 위·아래 두 그룹으로 나뉜다.
- 이미지를 확인해 보니 종의 형태 차이라기보다 촬영 조건(배경, 밝기, 크기, 자세) 차이로 보인다. 다만 확정한 것은 아니다.

## 5. 분포가 지나치게 넓은 어종이 있는가?
**Dascyllus reticulatus**가 가장 넓다.
- 자기 중심과의 평균 cosine이 0.731로 가장 낮다(spread 0.269).
- 종별 silhouette이 0.006으로 사실상 0이고, 샘플의 40%가 음수 silhouette이다.
- 이 종의 이미지는 역광 실루엣 형태가 많다. 그래서 색과 무늬 정보가 부족해 다른 자리돔류와 섞이는 것으로 보인다.

## 6. Silhouette Score는 어느 정도인가?
- 0.127(cosine)로, "클래스가 다소 겹치는 경향" 구간이다.
- 하지만 k-NN 일치율 0.947, 최근접 중심 일치율 0.87–0.95와 함께 보면 의미가 달라진다.
- 종 내부 분산은 크지만(종 내부에 여러 모드), 국소적으로는 같은 종끼리 모여 있다.
- 즉 전역적으로 둥근 cluster는 아니지만, 국소적으로는 구분 가능한 구조다.

## 7. 종별 평균 cosine similarity에서 특히 유사한 종은?
- P. dickii ↔ C. chrysura: 0.901
- D. reticulatus ↔ C. chrysura: 0.896
- D. reticulatus ↔ P. dickii: 0.876

자리돔과(Pomacentridae)의 비슷한 체형 3종이 서로 높은 유사도를 보인다. 흰동가리(A. clarkii)는 0.71–0.74로 멀다.

## 8. DINOv2 ViT-S/14를 Backbone 후보로 계속 테스트할 가치가 있는가?
**있다.** 다만 주의할 점이 있다.

긍정적인 근거:
- 학습 없이도 같은 궤적을 제외한 10-NN 일치율이 0.947이다.
- 가장 어려운 D. reticulatus도 최근접 중심 일치율이 0.87이다.

주의할 점:
- 2D PCA/t-SNE/UMAP의 분리는 384차원을 2차원으로 줄이며 정보가 손실된 결과다. 따라서 이것만으로 실제 분류 정확도가 높다고 결론 내릴 수 없다.
- 이미지가 작다(중앙값 약 96×87px를 224px로 업스케일).
- Fish4Knowledge는 고정 카메라 영상이다. 종과 카메라 위치·배경이 상관되어 있을 수 있어, 배경 단서가 분리에 기여했을 가능성이 있다.
- 4종 중 1종(A. clarkii)이 매우 쉬운 종이라 평균 지표를 끌어올린다.

## 다음 단계 (필수)
```
DINOv2 Freeze → 384-d embedding → Linear Classifier → 4 species
→ Linear Probe Accuracy / Macro-F1 측정
```
- train/test 분할은 trackingID(궤적) 단위로 해야 한다. 같은 궤적의 연속 프레임이 양쪽에 들어가면 정확도가 부풀려진다.
- 종별 confusion matrix로 D. reticulatus / P. dickii / C. chrysura 사이의 혼동을 확인한다.
- 가능하면 mask 이미지로 배경을 제거한 버전과 비교해 배경 의존도를 점검한다.
- 이후 ViT-B/14와 비교한다.
