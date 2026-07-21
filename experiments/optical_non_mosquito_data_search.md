# 광학 비모기 곤충 데이터 — 검색 결과 정리

> 2026-05-25 조사. 우리 deployment validation에 가장 큰 갭인 "광학 modality로 측정된 비모기 곤충 (특히 깔따구) 데이터" 확보 경로 정리. 향후 hardware prototype 측정 또는 추가 데이터 확보 시 참고.

---

## TL;DR

* **공개 다운로드 가능한 광학 비모기 데이터는 사실상 UCR InsectWingbeat 하나** (초파리·집파리만 포함, 깔따구 없음).
* 미공개 데이터셋은 5개 정도 있고 모두 저자에게 직접 요청 필요 (1-4주 응답 대기).
* 가장 잠재력 큰 미공개 후보 → **eBoss (Thomas 2024, NJIT)**, 302K 곤충 관측, Diptera 클러스터 포함.
* 깔따구(Chironomidae) 광학 측정 공개 데이터는 어떤 검색으로도 찾을 수 없음.
* 진짜 deployment validation은 **hardware 팀의 prototype 측정**이 답.

---

## 1. 공개 다운로드 가능 (우리가 이미 가진 것)

### 1.1 Wingbeats — Potamitis et al. 2018
* **출처**: Kaggle `potamitis/wingbeats`
* **곤충**: 모기 6종 (Aedes aegypti, Ae. albopictus, Anopheles arabiensis, An. gambiae, Culex pipiens, Cu. quinquefasciatus)
* **수량**: 279,566 wav 파일
* **modality**: 광학 (IR LED + phototransistor)
* **비모기**: ❌ 없음
* **참고**: https://paperswithcode.com/dataset/wingbeats

### 1.2 UCR InsectWingbeat — Chen et al. 2014
* **출처**: https://www.timeseriesclassification.com/description.php?Dataset=InsectWingbeat
* **곤충**: 모기 4종 × 암수 + **초파리 (Drosophila simulans)** + **집파리 (Musca domestica)**
* **수량**: 50,000 인스턴스 (TRAIN 25K + TEST 25K, 클래스당 5K)
* **modality**: 광학 (laser + phototransistor array, "pseudo-acoustic optical")
* **포맷**: 이미 spectrogram 처리됨 (200 freq bands × ~20 time steps)
* **비모기**: ✅ 파리 2종만
* **깔따구**: ❌ 없음
* **참고**: Chen Y et al. "Flying Insect Classification with Inexpensive Sensors." J Insect Behavior 27(5):657-77, 2014. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4541473/)

---

## 2. 미공개 — 저자 요청 필요 (우선순위 순)

### 2.1 eBoss — Thomas et al. 2024 ⭐ 최우선 후보
* **수량**: **302,093 곤충 관측** — 거대
* **곤충 범주**: 5 클러스터 (Lepidoptera 나비/나방, Odonata 잠자리, **Diptera 파리·모기·깔따구**, Hymenoptera 벌·말벌, Coleoptera 딱정벌레)
* **modality**: NIR 980nm laser diode + Si photodetector, 30,517 Hz sampling
* **장점**: 우리 hardware modality와 매우 유사. 데이터 양 압도적.
* **단점**: 종별 정확 라벨 없음 (클러스터만). 깔따구 포함 가능성은 있으나 명시 안 됨.
* **접근**: 데이터 "available on request from corresponding author"
* **연락**: Benjamin P. Thomas, bthomas@njit.edu
* **논문**: Temperature Dependency of Insect's Wingbeat Frequencies (2024). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11121811/)

### 2.2 KInsecta — Wührl et al. 2024
* **수량**: 7 species, proof-of-concept 수준
* **곤충**: Apis mellifera (꿀벌), Bombus terrestris (호박벌), Vespa crabro (말벌), Polistes dominula (paper wasp), Panorpa communis (scorpionfly), Eristalis tenax (drone fly), Episyrphus balteatus (banded hoverfly)
* **modality**: 광학 wingbeat sensor + 카메라 + 환경센서 (multisensor)
* **장점**: 7종 비모기 (벌, 파리, 나방류)
* **단점**: 깔따구 없음. 모기 없음. 데이터 양 적음.
* **접근**: 코드는 GitLab https://gitlab.com/kinsecta/sensorik_dev/sensorcluster, 데이터는 명확하지 않음
* **연락**: GitLab issue 또는 저자 이메일
* **논문**: arXiv 2404.18504, [arxiv html](https://arxiv.org/html/2404.18504v1)

### 2.3 InceptionFly Drosophila — Kalfas et al. 2022
* **수량**: 22,744 wingbeat (21,572 D. suzukii + 1,172 D. melanogaster)
* **곤충**: Drosophila 2종 (모기 없음)
* **modality**: 광학 wingbeat sensor (Potamitis 스타일), 8 kHz sampling
* **장점**: Drosophila 데이터 풍부
* **단점**: UCR과 중복 (초파리만). 깔따구·기타 곤충 없음.
* **접근**: "raw data...available on request, without undue reservation"
* **논문**: Front Plant Sci 13:812506 (2022). [Frontiers](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.812506/full)

### 2.4 Burkett-Cadena 모기 — Florida 2021
* **수량**: 21,825 wingbeat, 29종
* **곤충**: 북미 모기만 (29 species). **비모기 없음**.
* **modality**: 광학 sensor
* **접근**: 비공개, 저자 요청 필요
* **연락**: Nathan D. Burkett-Cadena (University of Florida)
* **논문**: Sci Reports 2021. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8113239/)

### 2.5 Brydegaard 그룹 (Lund University) lidar 데이터
* **곤충**: 다년간 다양한 곤충 lidar 관측 — Diptera, Lepidoptera 등
* **modality**: kHz multispectral lidar (장거리 광학)
* **장점**: 종 다양성 높음
* **단점**: 단일 통합 공개 dataset URL 없음. 여러 논문에 분산.
* **접근**: 저자 이메일 또는 institutional repository 확인
* **연락**: Mikkel Brydegaard (Lund University)

---

## 3. 깔따구 (Chironomidae) 특별 조사

* 깔따구 wingbeat에 대한 비행음 (acoustic) 연구는 1979년부터 있음 (Sawedal & Hall)
* **광학 modality로 측정한 깔따구 공개 데이터셋은 어떤 검색으로도 찾을 수 없음**
* eBoss나 KInsecta 같은 다종 dataset에 우연히 포함될 가능성은 있지만 명시 안 됨
* 우리 deployment 환경 (홍콩 야외)에서 깔따구는 모기와 wingbeat 가장 유사 → **false positive 최대 위험 종**

→ 깔따구 광학 데이터 확보는 **hardware 팀 prototype 측정이 사실상 유일한 현실적 경로**.

---

## 4. 권장 다음 단계 (우선순위)

| 우선순위 | 행동 | 노력 | 시간 | 기대 가치 |
|---|---|---|---|---|
| 1 | **eBoss 데이터 요청** (Thomas, NJIT) | 이메일 1통 | 1-4주 | ⭐⭐⭐ 광학 비모기 가장 다양, Diptera 포함 |
| 2 | **Hardware 팀 prototype 측정** 요청 | 협업 | 1-3개월 | ⭐⭐⭐⭐ 진짜 deployment-relevant 데이터 |
| 3 | KInsecta 데이터 요청 | 이메일/GitLab | 1-4주 | ⭐⭐ 비모기 7종 (벌·파리) — 깔따구는 없음 |
| 4 | InceptionFly 저자 요청 | 이메일 1통 | 1-4주 | ⭐ Drosophila 추가 (UCR과 중복) |

### 이메일 요청 시 포함할 내용

```
1. 자기소개 + 소속 + 프로젝트 목표
2. 어떤 데이터가 필요한지 구체적으로
3. 학술 용도 / non-commercial 명시
4. 인용 약속
5. 데이터 형식 (raw signal vs spectrogram) 요청
6. 가능하면 종별 라벨 포함 요청
```

---

## 5. 발표/보고 시 정직한 framing

> "공개 광학 비모기 데이터는 UCR InsectWingbeat이 사실상 유일 (초파리·집파리만 포함). 깔따구 같은 핵심 곤충은 어떤 공개 데이터셋에도 없음. eBoss(NJIT)·KInsecta·InceptionFly 등 미공개 데이터셋은 저자 요청을 통해 1-4주 내 확보 가능성. 가장 신뢰성 있는 deployment validation 경로는 hardware 팀의 prototype 측정."

---

## 6. 참고 링크 / 인용

- Wingbeats 데이터셋: https://paperswithcode.com/dataset/wingbeats
- UCR InsectWingbeat: https://www.timeseriesclassification.com/description.php?Dataset=InsectWingbeat
- Chen Y et al. (2014) "Flying Insect Classification with Inexpensive Sensors": https://pmc.ncbi.nlm.nih.gov/articles/PMC4541473/
- Thomas B et al. (2024) "Temperature Dependency of Insect's Wingbeat Frequencies" (eBoss): https://pmc.ncbi.nlm.nih.gov/articles/PMC11121811/
- Wührl L et al. (2024) "Multisensor Data Fusion for Automatized Insect Monitoring (KInsecta)": https://arxiv.org/abs/2404.18504
- Kalfas et al. (2022) "Optical Identification of Fruitfly Species Based on Their Wingbeats" (InceptionFly): https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.812506/full
- Burkett-Cadena et al. (2021) "Infrared light sensors permit rapid recording of wingbeat frequency": https://pmc.ncbi.nlm.nih.gov/articles/PMC8113239/
- Sawedal L, Hall R (1979) "Flight tone as a taxonomic character in Chironomidae" (acoustic, not optical)
