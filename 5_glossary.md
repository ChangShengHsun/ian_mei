# 5 · 術語表

> 白話定義為主，需要時附上「為什麼這個概念在這個領域重要」。按主題分組。

---

## 問題定義類

**curvilinear structure segmentation（曲線結構分割）**
把影像裡細長、彎曲、會分岔的結構（血管、神經纖維、氣道、神經元、道路、裂縫）從背景分出來。這是這個領域的正式名稱，搜文獻用這個關鍵字。同義或相近：tubular structure segmentation（管狀結構）、thin structure segmentation。

**topology（拓樸）**
只關心「連通關係」而不關心形狀大小的性質。一條血管被切成兩段，拓樸就變了；血管被畫粗一點，拓樸沒變。這個領域的核心價值主張是：**臨床上重要的是拓樸，但主流指標 Dice 量的是面積。**

**breakage / disconnection（斷裂）**
模型把一條連續結構預測成兩段。最主要的失敗模式。

**partial volume effect（部分容積效應）**
一個像素/體素同時包含血管與背景組織，所以它的灰階值是兩者的混合。當結構寬度接近成像解析度時（1–2 像素的微血管、神經纖維）特別嚴重，這是「真值標註本身就不可靠」的物理根源。

**class imbalance（類別不平衡）**
前景（血管）只佔 8–10% 像素，背景佔 90%+。導致 accuracy 這種指標毫無意義（全預測成背景就有 90% accuracy），也導致區域型 loss 數值不穩。

---

## 拓樸與幾何工具

**skeletonization / centerline extraction（骨架化 / 中心線萃取）**
把一個有寬度的區域壓成一像素寬的中心線。傳統做法是反覆的形態學侵蝕。clDice 的貢獻就是把這個操作寫成**可微分**的形式。

**differentiable（可微分）**
可以計算梯度、因此可以放進反向傳播訓練的操作。很多拓樸概念天生是離散的（連不連通是 0/1），要當 loss 用就必須先找到可微分的近似——這是這整個領域的技術主軸。

**soft erosion / dilation（軟性侵蝕／膨脹）**
形態學侵蝕（把區域縮小一圈）與膨脹（放大一圈）的連續可微版本，用 min/max pooling 的軟化版實作。clDice 的骨架化就是靠反覆交替這兩個操作。

**persistent homology（持續同調）**
代數拓樸的工具。白話：把機率圖當成地形，水位從高降到低，記錄每個「島」與「湖」在什麼水位誕生、什麼水位消失。得到的 **persistence diagram（持續圖）／barcode（條碼）** 完整編碼了所有閾值下的拓樸結構。優點是數學嚴謹，缺點是全域序列演算法、難平行化、很貴。

**Betti number（貝蒂數）**
拓樸特徵的計數：β₀ = 連通塊數量，β₁ = 洞的數量（2D），β₂ = 空腔數量（3D）。**Betti number error** 就是預測與真值的貝蒂數差。致命盲點：只數數量不管位置。

**induced matching（誘導配對）**
Betti Matching 的核心技術：在兩張 persistence diagram 的特徵之間建立**空間上有意義的一對一對應**，而不只是比較總數。這修掉了「數量對但位置全錯」的漏洞。

**homotopy equivalence（同倫等價）**
兩個形狀可以連續變形成彼此（不撕開、不黏合）。是「拓樸相同」的嚴格數學定義。Topograph 宣稱提供這個層級的保證。

**genus（虧格）**
一個曲面上「洞」的數量（甜甜圈的虧格是 1）。在血管上對應到迴路（例如 Willis 環）。

**signed distance map / SDF（符號距離圖／場）**
每個像素標上「到最近邊界的距離」，內部為負、外部為正。Boundary loss 用它把面積積分改寫成輪廓積分；SDF-TopoNet 用它當預訓練的回歸目標。**它天然帶有寬度資訊**——這是拓樸 loss 缺的東西。

**distance transform（距離轉換）**
計算距離圖的演算法本身。Hausdorff loss 與一票邊界 loss 都建立在它上面。

---

## 指標

**Dice coefficient**
2×交集 / (兩者面積和)。醫學分割的普世指標。**對細結構幾乎無感**——這是整個領域存在的理由。

**clDice（當指標用）**
拓樸精確度（預測骨架落在真值遮罩裡的比例）與拓樸敏感度（真值骨架落在預測遮罩裡的比例）的調和平均。事實上的第二標準。

**Hausdorff distance (HD) / 95HD**
兩個形狀之間「最壞情況」的邊界距離。95HD 取 95 百分位以抵抗離群點。盲點：距離小不代表拓樸對。

**APLS (Average Path Length Similarity)**
道路網路領域的拓樸指標：比較兩張圖之間所有節點對的最短路徑長度差異。醫學領域少用但概念可搬。

**DIADEM metric**
神經元重建專用，比對分岔點與末端的位置與拓樸關係。

**ARI / VOI**
連結體學（connectomics，神經元電子顯微鏡重建）的實例分割指標。用在二元血管遮罩上不對味。

---

## 模型與架構

**U-Net / encoder-decoder**
醫學分割的預設架構：編碼器逐步下採樣抽特徵，解碼器逐步上採樣還原解析度，中間用 skip connection 把細節接回來。**問題是下採樣本身會消滅 1 像素寬的結構**，skip connection 也救不回已經沒了的資訊。

**nnU-Net**
會自動根據資料集配置前處理、架構與訓練排程的 U-Net 框架。**醫學分割的預設強 baseline**，沒贏過它的方法沒有說服力。

**downsampling / aliasing（下採樣 / 混疊）**
把影像縮小時，高於取樣頻率一半的高頻訊號（＝細結構）會摺疊成假訊號或消失。這是訊號處理的基本定理（Nyquist）。**BlurPool** 的做法就是在下採樣前先低通濾波，是這個問題的教科書解法。

**ViT (Vision Transformer) / patchification**
把影像切成固定大小的方塊（通常 14×14 或 16×16 像素）當作 token 餵給 Transformer。**對細結構致命**：一條 2 像素寬的血管在一個 16×16 patch 裡只佔 1%，資訊在 tokenize 當下就被平均掉。而且這種失敗不會表現為「低信心」，模型是根本沒有表示那個資訊。

**Mamba / state space model (SSM)**
用線性複雜度處理長序列的模型家族（Transformer 是平方複雜度）。用在影像上需要把 2D 影像展平成 1D 序列，而**展平的順序**很重要——按行掃描會把斜向血管切碎，Serp-Mamba 因此改成沿血管路徑蜿蜒掃描。

**deformable convolution / dynamic snake convolution（可變形卷積／動態蛇形卷積）**
讓卷積核的取樣點位置可以學習偏移，而不是固定方形。DSCNet 的版本專門讓取樣點沿管狀結構的走向排列。

**foundation model（基礎模型）**
在超大規模資料上預訓練、可以泛化到多種下游任務的大模型（SAM、DINOv2、MedSAM）。**在細長結構上有已證實的架構性失敗**，見 [[2_methods#B5. 基礎模型：一個有證據的死路，以及兩個繞路]]。

**LoRA (Low-Rank Adaptation)**
凍結預訓練模型主體，只訓練少量低秩矩陣來適配新任務。VesselSAM 用它只訓 7% 參數就把 SAM 改造成血管可用。

**adapter**
插進凍結模型裡的小模組，同樣是為了低成本微調。

**domain randomization（域隨機化）**
訓練時大幅隨機化合成資料的外觀（對比、噪聲、形變），逼模型學結構而非外觀，用來提升對真實資料的泛化。vesselFM 的三個資料來源之一。

**flow matching / diffusion（流匹配／擴散）**
生成模型技術，近年被搬來做分割：把分割當成從噪聲逐步精修到答案的過程，而不是一次前向預測。在曲線結構上證據還很早。

---

## 損失函數

**loss function（損失函數）**
告訴模型「這次錯得多離譜」的計分規則，決定梯度往哪走。這個領域最主流的創新點，因為改 loss 不用動架構。

**region-based loss（區域型）**：Dice、交叉熵。對面積積分。
**boundary-based loss（邊界型）**：Boundary loss、Hausdorff loss。對輪廓的距離積分。
**topology-aware loss（拓樸型）**：clDice、TopoLoss、Betti Matching。針對連通關係。

**cross-entropy (CE，交叉熵)**
分類問題的標準 loss，逐像素獨立計算。

**perceptual loss（感知損失，VGG/LPIPS）**
用預訓練網路的特徵距離當 loss，讓輸出「看起來像自然影像」。**在醫學診斷影像上是紅線**——它的機制就是產生「看起來對」的紋理，也就是幻覺。

**perception-distortion trade-off（感知—失真取捨）**
被證明的數學結果：不可能同時最大化「看起來真實」與「忠於原始資訊」。這是所有生成式增強在診斷場景的根本風險。

---

## 資料與流程

**ground truth (GT，真值)**
人工標註的正確答案。這個領域的痛點是：**細結構的真值本身就不可靠**（見 partial volume effect）。

**train/test split（訓練測試切分）**
STARE 沒有官方切分，所以跨論文比較無效——見 [[3_datasets_metrics#4. 量測文化：這領域的數字可信嗎]]。

**ablation study（消融實驗）**
逐一拿掉自己方法的某個元件，看效能掉多少，用來證明「增益真的來自我說的那個東西」。

**tiling / patching（切塊推論）**
影像太大放不進顯存時，切成小塊分別推論再拼回去。**切塊邊界本身就是斷裂的來源**——SEMIR 之類的方法就是想避開它。

**self-supervised / weakly-supervised（自監督／弱監督）**
不用完整像素標註來訓練。YoloCurvSeg 是極端例子：只要一條有噪聲的骨架標註。

**FOV mask（視野遮罩）**
眼底照片是圓形視野，四角是黑的。要不要把黑角算進指標，會顯著改變數字——這是 DRIVE/STARE 論文不可比的原因之一。
