# 量測這條線上，哪些是我們的，哪些已經有人做過

2026-09-03。文獻查證，不吃 GPU。查證動機與 `novelty_check.md` 相同：
那一份查的是**損失函數**那一格，這一份查的是**指標與量測程序**那一格。

起因是一個內部誤認。2026-09-03 的對話裡，「我們有 ERL 這一把新尺」被當成
論文的故事講出來。查證後這句話是錯的，而且 repo 自己在三個地方已經寫過
它是錯的。同一天稍早，「約定會翻轉方法排名」也被講成當天最尖銳的新發現，
查證後發現 2024 年 12 月就有人發表了。

**兩個誤認都不是憑空產生的，都是「記得結論、忘記出處」。這份報告的目的是
讓下一個 session 不必再查一次。**

---

## 0. 一句話結論

| 主張 | 狀態 |
|---|---|
| ERL 是我們發明的尺 | **錯**。Januszewski et al. 2018，connectomics |
| ERL 在視網膜血管上沒人用 | **對**（六次搜尋 + 兩篇專門盤點指標的論文都沒列它） |
| 「約定會翻轉排名」是我們的發現 | **錯**。Berger et al. 2024 已在 Betti 上發表 |
| ERL 的三處欠定會翻轉排名 | **未見前人**。Berger 換的是像素鄰接，不是 ERL |
| 選擇洩漏 / 共用門檻 / seed 閘門 / 等成本比價 | **未見前人**（在連通性指標的脈絡下） |

---

## 1. ERL 不是我們的

`exp/erl.py` 第一行自己就寫了：

> "Expected Run Length: the connectomics metric **this series re-invented**."

出處 **Januszewski et al. 2018**（flood-filling networks）。領域參考實作是
Allen Institute 的 `segmentation-skeleton-metrics`。`stage-report/e15` 記錄了
經過：E10 造了 `severs`/`intact`/`absent` 的逐斷口分類並驗證兩次，文獻查核
發現 connectomics 從 2018 年就有同一個想法且更好（有長度權重、有單位），
於是換掉。

**我們造過一把尺，它被淘汰了。現在用的這把是借來的。**

---

## 2. 撞題：Berger et al. 2024

**"Pitfalls of topology-aware image segmentation"**，Berger, Lux, Weers,
Menten, Rueckert, Paetzold，arXiv:2412.14619，2024-12-19。

三條 pitfall，第一條原文：

> "connectivity choices distort the performance ranking between different methods"

**在 DRIVE 上做的**，Spearman ρ = −0.63（平均）。direct vs all connectivity
使連通塊數從 0.8% 變到 47.1%。

### 差異（讀過全文後確認）

| | Berger et al. | 我們 |
|---|---|---|
| 變的約定 | **像素鄰接**（4- vs 8-連通） | **ERL 的三處欠定**（見 §4） |
| 涵蓋指標 | Dice, Betti, Betti matching, VOI, ARE, ARI, clDice | ERL（他們沒碰） |
| 資料集 | DRIVE, CREMI, Roads, MSSEG2 | DRIVE, STARE, HRF, VessMAP |
| 測試集選擇洩漏 | 沒有 | 有，已量化 |
| 共用門檻 vs 各自門檻 | 沒有 | 有 |
| seed 數與統計閘門 | 沒有 | 有 |
| 後處理 vs 等成本降門檻 | 沒有 | 有 |
| 提到 ERL | **沒有** | — |

**後果：`composition_verdict` / `transfer_postproc_verdict` 裡「約定翻轉贏家」
這個現象，不能寫成我們的發現。** 正確寫法是：Berger 等人已在 Betti 類指標上
證明此事；我們證明它在 ERL 上以三個獨立的軸發生，並補上他們未涵蓋的四個軸。

---

## 3. 沒撞到：ERL 在這個領域是空的

六次不同角度的搜尋，沒有任何一篇把 ERL 用在視網膜／眼底血管分割上。
最強的佐證不是搜不到，而是**兩篇專門盤點拓樸指標的論文，清單裡都沒有它**：

- Berger et al. 2024：Dice, Betti, Betti matching, VOI, ARE, ARI, clDice。
- **Decroocq, Poon, Schlachter, Skibbe**, "Benchmarking Evaluation Metrics for
  Tubular Structure Segmentation in Biomedical Images", ShapeMI @ MICCAI 2025
  （口頭 + Best Paper）。程式碼 `github.com/megdec/BenchmarkTopoSegMetrics`，
  `metric_evaluation/quality_metrics.py` 定義的全部函式是：cbDice（sr / mb /
  srmb / srimb 及正規化變體）、Dice、Betti_matching、Betti_error、clDice、
  ccDice、ccDice_1。**沒有任何 run length 相關函式**，目錄裡也沒有相關檔名。

這個領域的連通性指標是 clDice、Betti、CAL、cbDice、ccDice、NSDice。
ERL 只出現在 connectomics，以及零星的氣道／道路分割。

**未查證的殘留風險**：Decroocq 那篇的正文（Springer 付費、ACM 403）沒讀到，
只讀了公開程式碼。若正文討論了 ERL 而未實作，論述要相應調整。
**待辦：用機構權限取全文。**

---

## 4. 三個約定 —— 我們手上真正沒被佔走的那一格

ERL = Σlᵢ² / L。三個獨立的欠定，每一個都改變數字，且都沒有標準答案：

| 欠定 | 兩種做法 | 實測差距 | 檔案 |
|---|---|---|---|
| **分母** | 只算被覆蓋的骨架（參考實作）／ 算全部真值骨架（我們） | `ours = reference × coverage`，浮點誤差內成立。頭條數字差 **6.8–7.1 個百分點**，參考實作給的是高的那個 | `erl_reference.md` |
| **片段長度** | 像素個數 ／ 8-鄰接邊權（對角 √2）／ 最長路徑 | pixels vs edges 只差 0.5–0.9%（分子分母同時修正）；**diameter 差 22–24 個百分點** | `erl_length.md` |
| **橋接算不算斷** | 任何未覆蓋的中線像素都斷 ／ 預測繞過去的缺口不算斷 | **+19.9 到 +27.4 個百分點** | `erl_convention.md` |

---

## 5. 這三張表全部是 pre-heldout 的 —— 必須重做

`erl_convention.txt`（8/27）、`erl_reference.txt`（8/28）、`erl_length.txt`
（8/29）全部早於 held-out 協定（9/01 才落地）。三支腳本共用同一段開頭：

```python
points = selection.selection_points(selection.load())   # pre-heldout sweep
items  = drive.load_split("val")                        # report half
data   = train.stack_split("train")                     # 20 張，不是 fit 的 15 張
weights = selection.SWEEP / run / f"epoch{epoch:03d}.pt"
```

CLAUDE.md 的規定是「pre-heldout 的數字與 heldout 的**不可比較**」。
所以**論文第一張表目前建立在洩漏協定的數字上**，而這篇論文的主張正是
洩漏會改變結論。這是 blocker，不是 nice-to-have。

**處置**：不修改這三支腳本（它們是 pre-heldout 量測的紀錄，照 repo 慣例
supersede 不覆寫），改寫一支 `exp/erl_spec.py`，在 held-out 協定、12 seeds
下同時量三個約定，並交叉它們。

---

## 6. 文體先例（好消息）

**Kovács & Fazekas**, "A new baseline for retinal vessel segmentation:
Numerical identification and correction of methodological inconsistencies
affecting 100+ papers", *Medical Image Analysis* 75:102300, 2022
（arXiv:2111.03853）。

純稽核論文、在 DRIVE 上、指出 100+ 篇論文因 **FoV 遮罩**處理不一致而報出
不可比的分數。上了這個領域的頂級期刊。軸與我們完全不同（FoV 影響
Acc/Se/Sp，不碰拓樸），所以不是撞題，**是這個文體在這個領域能發表的證明**。

---

## 7. 給下一個 session 的三條

1. **不要再說 ERL 是我們的。** 引 Januszewski et al. 2018。
2. **不要再說「約定翻轉排名」是我們的發現。** 引 Berger et al. 2024，
   我們的貢獻是 ERL 的三個軸，以及他們沒碰的四個軸。
3. **§4 的三張表在重跑成 held-out 之前不可引用。**

## 來源

- Berger et al., Pitfalls of topology-aware image segmentation, arXiv:2412.14619
- Decroocq et al., ShapeMI @ MICCAI 2025, LNCS 16171, doi 10.1007/978-3-032-06774-6_7
- Kovács & Fazekas, Medical Image Analysis 75:102300, 2022, arXiv:2111.03853
- Januszewski et al. 2018（ERL 出處）；Allen Institute `segmentation-skeleton-metrics`（參考實作）
