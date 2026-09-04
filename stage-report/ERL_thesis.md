# ERL for retinal vessels: a specification —— 論文結論總整理

2026-09-05。這份是**論文本身的骨架**，不是某一次實驗的報告。
每一條主張後面都標了證據檔案與量測條件；沒有標的就是還沒量。

> **引用規則**：本檔的每個數字都附「在哪個 split 上選的」與「用哪個 ERL 約定讀的」。
> 沒有這兩項的數字在這個 repo 裡不算數（`CLAUDE.md` Never 條款）。

---

## 0. 一句話

**ERL 是一把好尺，但它有三處沒有標準答案的欠定；欠定不是我們造成的，它在
領域參考實作裡就已經存在；而在這個領域，這把尺根本還沒有人用。**

---

## 1. 這篇不是什麼（先劃清界線，避免再犯）

| 別再宣稱 | 真正的出處 |
|---|---|
| ERL 是我們的尺 | **Januszewski et al. 2018**（flood-filling networks, connectomics）。`exp/erl.py` 第一行自己就寫著 "the connectomics metric **this series re-invented**" |
| 「約定會翻轉方法排名」是我們的發現 | **Berger et al., arXiv:2412.14619（2024-12）**，在 **DRIVE 上**、Spearman ρ = −0.63，換的是像素鄰接餵給 Dice/Betti/Betti-matching/VOI/ARE/ARI/clDice |

**我們真正持有的**：這件事發生在 **ERL** 上（Berger 沒碰 ERL），沿 **ERL 自己的
三個欠定軸**，外加他們沒涵蓋的四個軸（測試集選擇洩漏、共用 vs 各自門檻、
seed 數、後處理 vs 等成本降門檻）。

**文體先例**：Kovács & Fazekas, *Medical Image Analysis* 75:102300 (2022) ——
純稽核論文，指出 100+ 篇論文因 FoV 遮罩處理不一致而報出不可比的分數，
上了本領域頂刊。軸與我們完全不同，所以不是撞題，是**這個文體能發表的證明**。

**沒撞到的**：ERL 在視網膜血管分割上是空的。六次不同角度搜尋無一命中，
而且**兩篇專門盤點拓樸指標的論文清單裡都沒有它**（Berger 2024；
Decroocq et al., ShapeMI@MICCAI 2025 Best Paper，其程式碼實作 cbDice / Dice /
Betti_matching / Betti_error / clDice / ccDice，**無任何 run-length 函式**）。

詳見 `metric_novelty_check.md`。

---

## 2. 六條主張

### 主張 1 —— ERL 有三個獨立的欠定軸，交叉後光譜寬 40 個百分點

`erl_spec.md` / `exp/erl_spec.py`。DRIVE，held-out 協定，10 個臂 × 12 seeds，
27120 列。epoch 在 5 張 dev 上選；操作點是各 run 自己的 **dev Dice 峰值**
（用 Dice 不用 ERL——ERL 是受測量，拿它挑閾值是循環論證）。

| 軸 | 兩種做法 |
|---|---|
| **分母** | 只算已覆蓋骨架（參考實作）／ 算全部真值骨架 |
| **片段長度** | 像素數 ／ 8-鄰接邊權（對角 √2）／ 最長路徑（diameter） |
| **橋接算不算斷** | 任何未覆蓋中線像素都斷 ／ 預測繞過去的缺口不算斷 |

**(1) 每個臂的光譜寬 38.6–42.0 個百分點**，十個臂皆然。同一批預測可以誠實地
報成 35% 或 82%，而本領域最大的誠實效果是 **+1.4 點**。

**(2) 切分規則的正負號取決於長度約定**：`bridged > split` 在 pixels/edges 是
**10/10**，在 diameter 是 **0/10**。三張分開的表看不到，這是交叉的唯一理由。

**(3) 排序不能跨格**：從參考格 `split/edges/covered` 到 `bridged/diameter/covered`
的 Spearman ρ = **−0.103**，最大名次移動 **7 名**。

錨定（selftest，對真實視網膜影像）：`(split,*,full)` 必須等於
`erl_length.run_length`；`(bridged,pixels,full)` 必須等於
`erl_convention.bridged_run_length`。後者第一次執行就抓到「把 ERL 當比例、
多除一次骨架長度」的錯——**ERL 是長度不是比例**。

### 主張 2 —— 光譜隨 coverage 縮放，不是常數

`erl_spec_transfer.md` / `exp/erl_spec_transfer.py`。三個資料集 × **24 seeds** ×
4 個臂。量測程式碼從 `erl_spec` **import 不重寫**。

| 資料集 | 光譜（最小臂–最大臂） | coverage |
|---|---|---|
| STARE | 27.2 – 40.5 | 73–79% |
| HRF | 35.8 – 40.1 | 80–84% |
| **VessMAP** | **14.1 – 16.8** | **96–97%** |

預登的「每個資料集都 > 20 點」在 VessMAP **被推翻**，而且機制從表上直接讀得到：
`ours = reference × coverage`，coverage → 100% 時分母那個軸整個塌掉。

> **正確的說法**：三個約定的分歧，上界由**預測沒覆蓋到的骨架比例**決定。
> 論文要報這條規律加各資料集的 coverage，不要報一個「約 40 點」。
> 反向讀更尖銳：**在容易的資料集上報 ERL 的論文，藏起來的分歧比較少**——
> 那正好是報高分的那種論文。

主張 1(2) 的正負號反轉則**完美重現**：pixels/edges 各 4/4、4/4、4/4，
diameter 各 **0/4、0/4、0/4**，三個資料集十二個臂零例外。

### 主張 3 —— 參考實作本身就有兩個分母、一個沒寫下來的前提、和一個不良定義的長度

`erl_reference.md` / `exp/erl_reference_check.py`。**真的執行了**
`segmentation-skeleton-metrics 5.9.5`（不是讀原始碼轉述）。
做法是直接從骨架影像建出他們的 `LabeledGraph`（一像素一節點、8-鄰接、
`node_voxel=(0,row,col)` 使其 `physical_dist` 給正交 1／對角 √2），
再呼叫他們自己的 `ERLMetric.compute_graph_erl`——**受測的是我們的抄寫，
不是移植**。

**(a) 公式抄寫正確。** 合成直線斷裂上，他們的 `ERL` 與我們的
`(split, edges, covered)` **浮點等值**，六案例皆然。

**(b) 同一個套件吐出兩個分母不同的數字。** `skeleton_metrics.py:644` 的 `ERL`
以已覆蓋片段為分母；`:857` 的 `normalized ERL` 再除以完整骨架長度。
**引用「ERL」的論文不會說自己用的是哪一個。**

**(c) 有一個沒寫下來的連通性前提。**
```python
for label in graph.node_labels():
    run_length = graph.run_length_from(nodes[0])   # 只走 nodes[0] 那一塊
```
同標籤但在骨架上斷開的部分完全不計，而載入器（`graph_loading.py:357-406`）
沒有拆分步驟。3D connectomics 默默成立；視網膜上不成立，因為好的預測是
**一整塊**連通血管而它覆蓋的骨架處處有斷口。真實模型（`H_aug_s0` @0.5）：

| 影像 | 未被走到的已覆蓋骨架 | 他們的 ERL | 重標籤後 |
|---|---|---|---|
| 01 | **97.8%** | **21.81** | 6394.21 |
| 02 | 13.7% | 7874.06 | 6929.71 |
| 03 | **95.3%** | **33.47** | 1033.64 |

影像 01 差約 **300 倍**，且誤差隨資料變動（13.7%–97.8%），不是固定偏差。

**(d) 連片段長度都不是良定義的。** `run_length_from` 走 DFS 生成樹且保留每條
斜邊；骨架圖只要有環（L 形轉角、樓梯形，8-鄰接下遍地都是）生成樹就不唯一。
三像素 L 形轉角在六種節點排列下他們給 **2.0 或 2.414**（我們六種都是 2.0，
因為 `edge_list` 在兩個正交替代存在時丟掉斜邊）。在他們的管線裡那個順序
就是 **SWC 檔案裡點的順序**。這也解釋了真實影像上修好 (c) 之後仍有的
**+0.76%～+1.71%** 殘差。

> **所以規格論文要規定的不只是三個約定**，還要加兩條：
> **片段長度必須與輸入順序無關**、**標籤在真值骨架上必須連通，否則先拆開**。

### 主張 4 —— 協定洩漏可以製造出比任何誠實效果都大的結果

`exp/leak_ledger.py`，DRIVE 487 runs。基準線：本 repo 裡任何已發表拓樸損失
在各臂自己的 dev 最佳門檻上能產生的最大誠實效果是 **+1.4 點**。

| 洩漏層 | runs | 平均落差 | 最差的 run | vs +1.4 標竿 |
|---|---|---|---|---|
| **checkpoint**（用測試集挑 epoch） | 487 | **+2.3** | **+24.9** | 1.6× |
| threshold | 468 | −2.7 | +0.0 | — |
| geometry | 72 | −0.8 | +0.0 | — |
| best.pt（用 Dice 偷看、報 ERL） | 487 | −1.9 | +8.4 | — |

threshold 與 geometry 兩層沿著「traced ERL 單調」的軸選，**argmax 不帶任何
資訊，洩漏全部在 Dice 預算上**（`threshold_argmax_only()` 被斷言為恰好 0）。
468 個 run 裡有 433 個，dev 挑的門檻一放到 20 張測試圖上就超支 0.02 的預算，
平均超支 0.004 Dice。

### 主張 5 —— 那個統計閘門不是檢定，是一個保證會失效的計數

`seed_stability.md` + `exp/seed_survival.py`。

閘門（`calibration.decide`）要三件事同時成立：平均 > 0、t > 2、
**每一顆 seed 的差都為正**。前兩個隨 seed 數變**鬆**，第三個變**嚴**。

**實測**：STARE 從 12 顆補到 24 顆，兩格 `HOLDS → fails`，
而且**效果變大、t 也變大**：

| 格子（erl_bridged vs A_dice @0.5） | 12 seeds | 24 seeds |
|---|---|---|
| `H_aug_clw` | +10.3% t 5.51 **HOLDS** | +10.6% t 6.93 **fails** |
| `K_focal_aug` | +10.1% t 5.79 **HOLDS** | +10.3% t 7.51 **fails** |

三個資料集合計 HOLDS 從 **6 格掉到 2 格**，**0 格**往反方向走。

**機制**：第三個條件撐過 n 顆的機率是 `(1−p)ⁿ`，p 是新種子反對的比率。
60 格量下來中位 p = **0.21**，半衰期 **3.0 顆種子**。只要 p > 0，
`(1−p)ⁿ → 0`——**每一格最終都會死，「過閘」只是「跑得還不夠多」。**

**兩個附帶發現**：
- 重抽通過率那一欄是封閉形式 `C(n−d,k)/C(n,k)`，`d` = 反對票數。
  表裡的 75/67/58/50/33/17/0 就是 d=1，55/42/32/23/9/2/0 就是 d=2，逐格吻合。
  **所以它量到的不是穩定度，是反對票的數目。**
- **`d = 0` 不是 `p = 0`。** 三法則：n=12 零事件 → p < 0.25 → 撐過 24 顆的機率
  `0.75²⁴ = 0.001`。我們有一條預登預測就是這樣死的（見「方法教訓」）。

**但這一段沒有殺掉本論文自己的主張**：加上常態模型 `p = Φ(−mean/sd)` 之後，
`d=0` 分成兩群——損失比較的 2 個存活者 0/2 半衰期 > 1000 顆（約 50 顆就死）；
DRIVE composition 的 80 個存活者有 **55 個**半衰期 > 1000 顆（中位模型 p =
2.87e-06，約 24 萬顆）。**同一個 `HOLDS` 是兩種東西。**

### 主張 6 —— ERL 不是既有指標的同義詞（存在性檢查）

`metric_redundancy.md` / `exp/metric_redundancy.py`。這支腳本寫來是**有可能
殺掉論文**的，預登了擊殺條件：若 ρ(ERL,clDice) > 0.95 且不一致率 < 10%，
clDice 就是排名替代品，論文必須改立論。**未觸發。**

資料是既有的 97,400 列（dice / cldice / betti0_err / erl **同在一列**，
不用重算也不可能走鐘），epoch 用 `chosen_epochs()`（讀各 run 自己的 log.csv）。

| 指標 | ρ vs ERL（56 個未挑選的臂） | 排反的臂對 |
|---|---|---|
| Dice | −0.304 | 60.2% |
| **clDice** | **+0.307** | **36.6%** |
| Betti-0 誤差 | +0.177 | 42.5% |

**論文該印的那一句不是 ρ**（ρ 的正負號在不同臂集之間不穩定），**是這個**：

> **22 個 run 的 clDice 落在 0.8153 ± 0.002 內——讀者會說這些模型一樣好——
> 它們的可追蹤長度從 21.3% 到 73.0%，差 51.6 個百分點。** 對齊 Dice 是 51.3 點。

加上 **ERL 有單位**（最差的 run 1052 px、最好 6423 px）：
「在撞到錯誤之前可以追蹤 1052 像素」是一句話，Dice、clDice、Betti 數都講不出來。

**約定不翻轉這個結論**（`composition` 的 `raw` 列，同閾值同影像）：
`erl_split` ρ vs Dice = −0.867、`erl_bridged` = −0.842，兩約定彼此 ρ = **0.964**。
**約定大幅改變 ERL 的數值，但不改變它與 Dice 的排序關係。**

---

## 3. 誠實的界線（論文必須自己寫出來的）

1. 主張 6 的 10 個 ledger 臂上 ρ(ERL,Dice) = −0.830，**有一部分是建構出來的**
   （`clw` 權重 1–64 掃的就是 Dice 換 ERL 的取捨）。56 臂數字當發現，
   10 臂當機制。這句話已印在報告的表正上方，不在註腳。
2. 主張 3(c) 說的是「把預測連通塊當標籤餵給他們的函式會發生什麼」——那是
   他們載入器的行為（已讀原始碼確認無拆分步驟），但**不是在他們的 3D 資料
   上實測**。主張 3(d) 不依賴任何重建，可直接重現。
3. 主張 6 的主表只用 `split/pixels/full` 一格，且閾值是 `sweep_score.py` 的
   固定值。切分軸已測（不翻轉），**長度軸與分母軸未測**。
4. Decroocq et al. 正文（Springer 付費）未讀，只讀了公開程式碼。
   **待辦：用 NTU 機構權限取全文。**
5. 主張 1、4、6 目前是 **DRIVE 12 seeds**。補到 24 seeds 的排程執行中
   （`exp/run_deep.sh`），完成後 `run_survival.sh`、`run_gaps.sh` 會重算並
   對預登預測做機械式結算。

---

## 4. 方法教訓（這些本身就值得寫進論文的方法段）

- **預先登記的預測，引用的證據必須真的支持它。** 我預登「兩個在重抽曲線讀
  100% 的格子會活到 24 顆」，但 n=12 時 k=12 只有一個子集（就是全集），
  那一欄不是重抽、資訊量為零。兩格都死了。預登的價值就在於它讓這個錯誤
  事後掩飾不掉。
- **抄寫別人的定義不等於驗證。** 主張 3 的三件事全部只有把程式跑起來才拿得到。
- **同一個工具的兩個輸出可以有不同分母**，而使用者不會察覺。
- **修正的方向不一定不利**：查證後「我們偏離參考實作」變成「參考實作自己就
  欠定」，論述更強。

---

## 5. 投稿判斷

| 場地 | 截稿 | 適配 |
|---|---|---|
| **ML4H 2026 Findings**（4 頁，非存檔） | **2026/9/10** | Area 2 明列 "benchmarks, evaluations and best practices"；非存檔所以不燒材料 |
| **ISBI 2027**（4 頁，存檔） | **2026/10/26** | 落在 "Uncertainty quantification and trustworthy AI in imaging"；**但沒有評估方法學軌道，審稿人來自方法開發者**，「沒有新方法」是典型死法 |
| ISBI 2027 workshop | 2027/1/26 | 需先有對題的 workshop 提案（9/21 截），不可控 |
| IPMI 2027 | 2026/12/7 | 全口頭、極度挑選，高風險 |
| MICCAI 2027 workshop | 約 2027/7 | 還要十個月，太久 |
| **Medical Image Analysis**（期刊） | 無截稿 | **文體先例所在**（Kovács & Fazekas）；週期中位約 262 天 |

**核心矛盾：材料是六張表的體量，而 4 頁只裝得下一張半。**

建議順序：**ML4H Findings（9/10，免費試水）→ ISBI 2027（10/26，主題砍成
「參考實作 + 規格」）→ 全文投 MedIA。** ISBI 那版建議以**主張 3** 為主軸——
它最新、沒人有、具體刺眼，而且只需要兩張表。主張 4、5、6 留給期刊版。

**這是取捨不是對錯，需要教授拍板。**

---

## 6. 證據檔案索引

| 主張 | 腳本 | 結果 | 報告 |
|---|---|---|---|
| 1 | `exp/erl_spec.py` | `results/erl_spec.txt` | `erl_spec.md` |
| 2 | `exp/erl_spec_transfer.py` | `results/erl_spec_transfer.txt` | `erl_spec_transfer.md` |
| 3 | `exp/erl_reference_check.py` | selftest 輸出 | `erl_reference.md` |
| 4 | `exp/leak_ledger.py` | `results/leak_ledger.txt` | — |
| 5 | `exp/seed_stability.py`, `exp/seed_survival.py` | `results/seed_stability.txt`, `results/seed_survival.txt` | `seed_stability.md` |
| 6 | `exp/metric_redundancy.py` | `results/metric_redundancy.txt` | `metric_redundancy.md` |
| 前人 | — | — | `metric_novelty_check.md` |

**注意**：`exp/results/` 目前是指向 `/tmp2/ivanchang/ian_mei_results` 的符號連結
（2026-09-03 為釋放磁碟所做），因此結果檔**不在版控內**。
腳本與報告在版控內；結果需從機器上取得或重跑。
