# E0 · 讀完那兩篇撞題的論文

2026-08-16 執行。不需運算，純閱讀查證。

stage_1 第 0.4 與 0.5 節指出，我們原本規劃的 E1 與 E3 各自撞到一條既有研究線。
E0 的任務是把「撞到」講清楚：到底是完全重複，還是只是相鄰。
兩個問題，各要一個明確的是或否。

---

## 問題一：拓樸不確定度那篇，有沒有用真實的標註者分歧驗證過？

**論文**：*Topology-Aware Uncertainty for Image Segmentation*，NeurIPS 2023，
arXiv:2306.05671。它用離散莫爾斯理論把不確定度估在「拓樸結構」上，
而不是估在每個像素上，目的是標出高風險的連接讓人類複核。

**答案：沒有。全部評估都對單一標準答案（ground truth）做。** `[verified]`

查到的細節：

| 項目 | 內容 |
|---|---|
| 資料集 | DRIVE、ROSE（OCTA 視網膜）、ROADS（航照道路）、PARSE（3D 肺動脈 CT） |
| 不確定度指標 | Expected Calibration Error 與 reliability diagram（Table 1、Fig 9） |
| 分割指標 | Dice、clDice、ARI、VOI、Betti number error、Betti matching error |
| 人機協作模擬 | 有，在 ROSE 上（Fig 11） |
| STARE | 沒有用 |

關鍵在人機協作那段的做法。原文：

> "The user is given each method's final segmentation map, and inspects
> structures in decreasing order of uncertainty (till 0.5). Each uncertain
> structure is subjected to a yes/no decision, which is denoted as one 'click'."

那個「使用者」是模擬的，而且它的是或否是拿標準答案回答的。
訓練那一側也一樣：`"M_ϕ is trained by comparing with the ground truth (GT)
annotation"`。

**所以整條鏈路都假設標準答案是對的。**
論文說「模型的不確定度落在需要人複核的地方」，證據是「落在跟標準答案不合的地方」。
這兩件事只有在標準答案本身沒有爭議時才等價，
而 stage 0 第 2.6 節量到的 STARE 兩位標註者 Dice 只有 0.740。

**結論：E1' 成立，而且缺口比原本想的更具體。**
我們要補的不是「做一次不確定度」，是「把不確定度對到人類真的會吵架的地方」。
他們用 DRIVE（單一標註者）而沒有用 STARE（兩位標註者），
所以這個驗證他們做不了，不是不想做。

---

## 問題二：多標註者那條線，有沒有報過任何拓樸指標？

**論文**：*Learning from multiple annotators for medical image segmentation*，
Pattern Recognition 2023, Vol. 138, 109400，Zhang、Tanno 等人。
以及它的前身 *Disentangling Human Error from the Ground Truth in Segmentation
of Medical Images*，NeurIPS 2020，arXiv:2007.15963。
方法是兩個耦合的 CNN，一個估共識標籤，一個估每位標註者的逐像素混淆矩陣。

正式版在 ScienceDirect 與 PMC 都擋住（403 與 reCAPTCHA），
所以以下細節取自可公開取得的前身版本。兩版方法與資料集相同，正式版是延伸。 `[verified: 前身版本]`

**答案：沒有。一個拓樸指標都沒有。** `[verified]`

| 項目 | 內容 |
|---|---|
| 資料集 | MNIST（合成標註）、MSLSC（ISBI 2015 多發性硬化病灶）、BraTS 2019（腦瘤）、LIDC-IDRI（肺結節） |
| 指標 | Dice、混淆矩陣的 Root-MSE、Generalized Energy Distance |
| 拓樸指標 | 無 |
| 目標形狀 | 全部是團塊狀：病灶、腫瘤、結節 |

第四列才是真正的答案。**這條線從頭到尾沒有處理過管狀或線狀結構。**
沒有血管、沒有氣管、沒有道路、沒有神經纖維。
團塊狀目標的標註者分歧發生在邊界，多一圈少一圈；
線狀結構的標註者分歧會直接改變連通性，一段接不上去就是斷成兩塊。
這是兩種完全不同的錯誤，而後者從來沒被量過。

**結論：E3' 成立，而且交集是真的空的。**
不是「別人做過但沒做我們這個角度」，是「多標註者那邊沒碰過線狀結構，
拓樸那邊沒碰過多標註者」。

---

## E0 對排程的影響

兩題都是綠燈，所以 E1' 與 E3' 都從「待確認」變回「該做」。

而且兩者可以合併成一次運算：都在同樣那 8 個 STARE 模型上跑，
都需要 sigmoid 機率圖，都需要 ah 與 vk 兩份標註。
分兩次跑等於把 8 個模型載入兩遍。**合併為 E3'+E1' 一次執行。**

修正後順序：E2（已完成）→ **E3'+E1' 合併** → E4 → E5。

---

## 這次查證再一次證實的方法論

stage_1 的教訓是「先查文獻再設計實驗」。E0 補上後半句：
**查到撞題之後不要直接放棄，要讀進去確認撞的是哪一塊。**

E1 初看是「別人已經做過拓樸不確定度」，讀進去才發現他們的驗證迴路是封閉的，
標準答案同時當訓練目標和評分依據，所以「不確定度是否對應人類分歧」這題他們根本沒問。
E3 初看是「多標註者已經做爛了」，讀進去才發現他們的目標全是團塊。

兩次都是：**摘要層級看起來重複，實驗設定層級看起來互補。**

---

## 來源

- [Topology-Aware Uncertainty for Image Segmentation (arXiv:2306.05671)](https://arxiv.org/abs/2306.05671)
- [Learning from multiple annotators for medical image segmentation (Pattern Recognition 2023)](https://www.sciencedirect.com/science/article/pii/S0031320323001012)
- [Disentangling Human Error from the Ground Truth in Segmentation of Medical Images (arXiv:2007.15963)](https://arxiv.org/abs/2007.15963)
