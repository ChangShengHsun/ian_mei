# 「幾何當輸出表示」這一格，有沒有人做過？

2026-09-06。文獻查證，不吃 GPU。做法與 `metric_novelty_check.md` 相同：
在寫任何一行方法程式碼之前查，因為這個 repo 已經被「忘記查前人」燒過兩次
（ERL 本身是 Januszewski 2018；「約定翻轉排名」是 Berger et al. 2024）。

## 0. 一句話結論

| 主張 | 狀態 |
|---|---|
| 學習式 flux 場當輸出，是新的 | **不是**。DeepFlux（自然影像骨架）、VesselPose（3D 血管） |
| flux 用於管狀結構中線抽取，是新的 | **不是**。Bouix & Siddiqi, MedIA 2005（古典，作用在既有分割上） |
| 「以中線為輸出、遮罩由它算出」在 **2D 眼底** 上是空的 | **看起來是**，但有殘留風險（見 §3） |
| 骨架訊號在 2D 眼底上只以**損失**或**架構偏置**形式出現 | **是**，這是這個領域的固定模式 |

## 1. 別處已經有的

- **DeepFlux**，arXiv:1811.12608 / IJCV 2021。CNN 預測 2D 向量場，把每個場景點
  映到候選骨架像素，骨架從 flux 表示還原。**自然影像的骨架偵測**，六個基準。
  原文的賣點：「flux 是區域式的向量場，因此更能處理寬度大的物件部位」。
- **ProMask**，Neural Networks 2023。骨架偵測的另一種機率遮罩表示。
- **Bouix & Siddiqi et al.**, "Flux driven automatic centerline extraction",
  *Medical Image Analysis* 2005。用歐氏距離函數梯度場的平均外向通量做骨架化，
  管狀結構。**古典方法，作用在既有分割上，不是學習式輸出。**
- **VesselPose**，arXiv:2605.00538（2026）。從學習到的 **voxel-wise 方向向量**
  重建血管圖，**3D** 血管影像。

## 2. 2D 眼底這個領域實際在做什麼

近期（2025–2026）確實有大量「骨架感知」的工作，但**骨架一律以損失或架構偏置
進入，不是輸出**：

- **Skeleton Distance Loss (SDL)**，2025（78K 參數的高效模型）。**一個損失**
  ——與本 repo 的 `clw`（D-E）同一類，而 `clw` 已經是我們的基線且是前人的
  （Skeleton Recall Loss, ECCV 2024）。
- **MVCN**（Multi-branch Vessel-shaped Convolution Network），2025。血管形狀
  感知的**卷積**，內部學拓樸與形狀表示，**輸出仍是遮罩**。
- **Optimized U-Net**，2026。用骨架圖去取像素，輸出仍是遮罩。
- 2D 眼底的中線抽取傳統上用「正規化梯度場的散度 + 形態學」——**古典、後處理、
  作用在既有分割或影像上**。

**模式很清楚：這個領域讓骨架當老師（損失）或當建築師（卷積結構），
從來不讓它當產品（輸出）。**

## 3. 殘留風險（必須寫在論文裡）

**沒讀到 Bansal et al., "Retinal Vessel Segmentation: A Comprehensive Review
From Classical Methods to Deep Learning Advances (1982–2025)", *Advanced
Intelligent Systems*, 2026**——Wiley 回 **HTTP 403**。這是涵蓋四十年的全領域
綜述，是本次查證最強的單一證據來源，而它沒被讀到。

同樣地 `arxiv.org/pdf/2306.06116`（Overview of Deep Learning Methods for
Retinal Vessel Segmentation）抓下來是壞的 PDF，也沒讀到。

**待辦：用 NTU 機構權限取 Bansal 2026 全文，檢查它的方法分類裡有沒有
「輸出為向量場／中線表示」這一類。** 在那之前，§0 第三列只能寫成
「六次搜尋未見，綜述未讀」，不能寫成「沒有人做過」。

（這與 `metric_novelty_check.md` 對 Decroocq 全文的處置相同：公開程式碼看得到、
正文看不到，就照實寫。）

## 4. 所以我們的定位是什麼

**跟 ERL 那篇一模一樣的故事結構：把別的領域成熟的東西搬過來，並且做修正。**

- 搬的是 DeepFlux 的表示（自然影像骨架）與 Bouix 的 flux 中線觀念（古典 3D 管狀）
- 修正的是：2D 眼底的失敗模式是**整條細血管沒被看見**（HRF 上 41% 的
  p<0.01 中線在 ≥20px 的長段裡），而 mask 損失給一條 4px 寬血管的監督訊號
  每單位長度只有約 4 個二元像素

**不能宣稱發明 flux 表示。** 可以宣稱的是：在這個領域，幾何只當過損失與架構偏置，
沒當過輸出；而失敗模式恰好是輸出表示能對症的那一種。

## 來源

- Wang et al., DeepFlux for Skeletons in the Wild, arXiv:1811.12608; IJCV 2021
- Bouix, Siddiqi et al., Flux driven automatic centerline extraction, MedIA 2005
- VesselPose, arXiv:2605.00538 (2026)
- ProMask, Neural Networks, 2023
- Bansal et al., Advanced Intelligent Systems 2026（**未讀，403**）
