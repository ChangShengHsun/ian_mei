# 三個約定，換三個資料集 —— 光譜是 ERL 的性質還是 DRIVE 的性質？

2026-09-04。`exp/erl_spec_transfer.py`，CPU，四個 shard 約一小時，已跑完。
論文（寫法 a）的 Table 2。

## 問題

`erl_spec.md`（Table 1）在 DRIVE 上量到：同一批預測，在三個約定軸交叉出的
十二格裡，可以誠實地報成 **相差 38.6 到 42.0 個百分點**的數字。

審稿人的第一個問題必然是：**這是 ERL 的性質，還是 DRIVE 的性質？**
DRIVE 是 20 張測試圖、中位血管寬 4.00 px。VessMAP 是 5.66 px，成像方式完全不同。

## 設定

- 三個資料集（STARE 10 張測試圖、HRF 15 張、VessMAP 20 張），各 **24 顆種子**、
  4 個臂（`calib.ARMS`）。11520 / 17280 / 23040 列。
- **量測程式碼一行都沒有重寫**：`measure`、`bridged_labels`、`skeleton_total`
  全部從 `erl_spec` import。同一段算式抄第二份，就是兩張表悄悄走鐘的方法。
- 操作點是各 run **自己在自己資料集上**的 dev Dice 峰值，讀自
  `calibration_curve*.csv`（`calib.peak_of`）。用 Dice 不用 ERL：ERL 是受測量。
- selftest 在**真實 VessMAP 影像**上重新錨定一次（DRIVE 4.00 px vs VessMAP
  5.66 px，在一個尺度成立、另一個不成立的約定就是真缺陷）。

## 結果

### 預測 2 成立，而且是完美重現

DRIVE 的發現：`bridged > split` 在 pixels/edges 是 10/10，在 diameter 是 **0/10**。
切分規則的正負號**取決於長度約定**。

| 資料集 | pixels | edges | diameter |
|---|---|---|---|
| STARE | 4/4 | 4/4 | **0/4** |
| HRF | 4/4 | 4/4 | **0/4** |
| VessMAP | 4/4 | 4/4 | **0/4** |

**三個資料集、十二個臂，一次例外都沒有。** 這是本系列少數乾淨重現的結果。

機制（`erl_spec.md` 已解釋）：橋接把片段併得更大也更分岔，而 diameter 只算
一條最長路徑，所以併起來反而掉。這在三張分開的表上看不到，這是交叉的唯一理由。

### 預測 1 被推翻，而推翻的方式是有機制的

預測是「每個資料集的每個臂光譜都 > 20 個百分點」。

| 資料集 | 最小臂 | 最大臂 | 判定 | coverage |
|---|---|---|---|---|
| STARE | 27.2 | 40.5 | 成立 | 73–79% |
| HRF | 35.8 | 40.1 | 成立 | 80–84% |
| **VessMAP** | **14.1** | **16.8** | **推翻** | **96–97%** |

不是雜訊，是機制，而且從表上直接讀得出來：

- **coverage 越高，分母那個軸就越沒有空間**。`ours = reference × coverage`，
  coverage → 100% 時 `full` 與 `covered` 兩欄合而為一。
- VessMAP 的 coverage 是 **96–97%**（STARE 只有 73–79%）。
- 而且 VessMAP 的基準 ERL 已經在 76–92%，天花板本身壓縮了可移動範圍。

**所以光譜不是 ERL 的常數，它隨「預測漏掉多少骨架」放大。**

這比原本的說法**更好**，因為它是一個可預測的規律而不是一個數字：

> 三個約定的分歧，上界由預測沒覆蓋到的骨架比例決定。在容易的資料集
> （VessMAP，覆蓋 97%）三個約定同意到 15 個百分點以內；在困難的資料集
> （STARE，覆蓋 73%）它們差 40 個百分點。

論文要據此改寫：**不要報一個「約 40 點」的常數，要報「光譜隨 coverage 縮放」
這條規律，並附各資料集的 coverage。** 反過來說，一篇在容易資料集上報 ERL 的
論文，它藏起來的分歧比在困難資料集上少——這正好是最需要被檢查的那種論文
（報高分的那種）。

## 預先登記

三條預測寫在 `exp/erl_spec_transfer.py` 標頭，在這支腳本計分任何一個 run
之前。第 3 條（`full == covered × coverage`）是恆等式不是發現，放在 selftest
裡斷言，這樣違反會被當成 bug 抓到而不是被讀成結果。

## 檔案

- `exp/erl_spec_transfer.py`（selftest：分片是精確分割、三資料集 fit/dev/test
  依檔名互斥、VessMAP 上重錨兩條、恆等式六格、dev 峰值不看 test 列）
- `exp/results/erl_spec_transfer.txt`
- `exp/results/heldout_transfer/{stare,hrf,vessmap}/erl_spec.shard*.csv`
