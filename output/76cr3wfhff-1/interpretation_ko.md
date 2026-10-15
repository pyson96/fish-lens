# DINOv2 ViT-S/14 × 76cr3wfhff-1 (8종) 결과 해석

## 실험 조건
- 데이터: `dataset/76cr3wfhff-1/*.rar` 8개(종별 1개), 520×390 TIFF 사진
- 사진 형태: 바닥(타일·콘크리트 등) 위에 놓인 물고기를 위에서 촬영
- 샘플: 가장 적은 종이 240장이라 8종 모두 240장으로 맞춤(seed=42), 총 1920장
- 모델: DINOv2 ViT-S/14를 freeze한 채 학습 없이 사용, CLS 토큰 384차원 → L2 정규화

## 주요 지표

| 지표 | 값 |
|---|---|
| Silhouette (cosine, 384-d) | **0.235** (euclidean 0.133) |
| 최근접 중심 일치율 (leave-one-out) | 0.954 ~ 1.000 (6종이 1.000) |
| 10-NN 일치율, 같은 개체(같은 종·60초 이내) 제외 | **0.993** |
| 10-NN 일치율, 같은 촬영일 사진 전부 제외 | **0.994** |
| 10-NN 일치율, 같은 종 안에서 cosine ≥ 0.90 이웃 전부 제외 (스트레스 테스트) | 0.987 (가장 낮은 종 C. chanos 0.938) |
| PCA PC1+PC2 설명 분산 | 21.8% |
| 평균 벡터 cosine 최댓값 | C. chanos ↔ J. trachycephalus 0.899 |

## 1. 각 어종이 독립적인 cluster를 형성하는가?
대체로 그렇다.
- t-SNE와 UMAP에서 8종 모두 자기 영역을 갖고, 다른 종이 섞여 들어오는 일은 드물다.
- PCA에서는 크게 겹친다. 다만 2개 축이 설명하는 분산이 21.8%뿐이라, PCA 그림만으로 판단하면 안 된다.

## 2. 가장 많이 겹치는 두 어종은?
**Oreochromis mossambicus ↔ Oreochromis niloticus**
- 같은 속(틸라피아)인 두 종이다.
- 10-NN에서 O. mossambicus 이웃의 4.8%가 O. niloticus다.
- UMAP과 t-SNE에서도 두 종이 붙어 있다.

평균 벡터 cosine 기준으로는 C. chanos ↔ J. trachycephalus ↔ N. albiflora ↔ E. tetradactylum(0.885~0.899)이 서로 가깝다. 은색의 길쭉한 체형이라는 공통점이 있다. 하지만 개별 샘플 수준에서는 거의 섞이지 않는다(이웃 섞임 2% 이하).

## 3. 가장 명확하게 분리되는 어종은?
- **Eleutheronema tetradactylum**(10-NN 0.997, silhouette 0.285)
- **Johnius trachycephalus**(0.995)
- **Upeneus moluccensis**(0.992, silhouette 0.285)

## 4. 같은 어종 내부에서도 여러 cluster가 생기는가?
그렇다. 거의 모든 종이 UMAP과 t-SNE에서 작고 촘촘한 섬 여러 개로 나뉜다. 특히 **Chanos chanos**는 UMAP에서 5~6개의 떨어진 섬으로 흩어져 있다.
- 이 섬들은 촬영일과 거의 일치하지 않는다(종별 KMeans와 촬영일의 ARI 0.00~0.14).
- 같은 표본을 같은 배경·같은 자세로 찍은 사진이 **서로 다른 날짜에 반복해서** 존재하는 것을 확인했다(예: 10/15와 10/18). 즉 섬은 "표본 + 배경" 단위로 생기는 것으로 보인다.

## 5. 분포가 지나치게 넓은 어종이 있는가?
**Chanos chanos**
- silhouette 0.085로 가장 낮고, 19%가 음수다.
- 자기 중심과의 평균 cosine이 0.848로 낮은 편이다.
- 섬 여러 개로 흩어져 있어서, 중심이 실제 데이터가 거의 없는 곳에 위치한다.

R. faughni도 spread가 0.154로 넓은 편이다.

## 6. Silhouette Score는 어느 정도인가?
- 0.235로 Fish4Knowledge(0.127)보다 높지만, 여전히 "완전히 분리된 둥근 cluster"는 아니다.
- 종 내부에 여러 섬이 있어 silhouette이 낮게 나오는 반면, 국소 구조(k-NN)는 거의 완벽하다.

## 7. 종별 평균 cosine similarity에서 특히 유사한 종은?
- C. chanos ↔ J. trachycephalus: 0.899
- C. chanos ↔ N. albiflora: 0.898
- C. chanos ↔ E. tetradactylum: 0.896

가장 덜 유사한 쌍은 J. trachycephalus ↔ O. niloticus(0.708)다.

## 8. Backbone 후보로 계속 테스트할 가치가 있는가?
**있다.** 이 데이터셋에서는 Fish4Knowledge보다 훨씬 긍정적이다.

긍정적인 근거:
- 8종인데도 학습 없이 10-NN 일치율 0.99 수준이다.
- 같은 날 사진을 모두 빼거나 매우 유사한 같은 종 이웃을 빼도 결과가 거의 유지된다.
- 같은 속인 틸라피아 두 종만 소폭 혼동한다.

주의할 점:
- **개체(표본) ID가 없다.** 같은 표본이 여러 날 반복 촬영된 정황이 있어서, 결과의 일부는 "종 인식"이 아니라 "같은 표본 재인식"일 수 있다. 스트레스 테스트로 영향이 작다는 것은 확인했지만, 완전히 배제하지는 못한다.
- 통제된 환경의 사진이다(물 밖, 위에서 촬영, 단순한 바닥 배경). 수중 영상인 Fish4Knowledge보다 쉬운 조건이다. 수중이나 낚시 현장 사진에 그대로 일반화된다고 볼 수 없다.
- 2D 시각화의 분리가 실제 분류 정확도를 보장하지는 않는다.

## Fish4Knowledge(4종)와 비교

| | Fish4Knowledge | 76cr3wfhff-1 |
|---|---|---|
| 촬영 조건 | 수중 고정 카메라, 저해상도(약 96px) | 물 밖, 520×390, 바닥 배경 |
| 종 수 / 이미지 | 4 / 1200 | 8 / 1920 |
| Silhouette (cosine) | 0.127 | 0.235 |
| 10-NN 일치율 (중복 제외) | 0.947 | 0.993 |
| 가장 겹치는 쌍 | P. dickii ↔ C. chrysura | O. mossambicus ↔ O. niloticus |

## 다음 단계 (필수)
```
DINOv2 Freeze → 384-d embedding → Linear Classifier → 8 species
→ Linear Probe Accuracy / Macro-F1 측정
```
- train/test는 촬영일 단위로 나누거나, 가능하면 개체 단위로 나눈다. 같은 표본이 여러 날 등장하므로 near-duplicate 그룹(예: cosine ≥ 0.95로 연결된 묶음)으로 나누는 것도 함께 검토한다.
- O. mossambicus와 O. niloticus 사이의 confusion을 따로 확인한다.
- 두 데이터셋을 섞어 학습하지 말고 데이터셋별로 평가한다. 촬영 도메인 차이가 크다.
