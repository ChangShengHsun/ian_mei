% 醫學影像細緻紋理與細長結構分析 — 文獻調研
% 整理：Ivan Chang（張勝勳）
% 2026-08-05


# 0 · 導讀 — 醫學影像中的細緻紋理／細長結構分析（2026-08-05 開檔）

> 這個資料夾是為了幫朋友（ian-mei）盤點一個題目：**在醫學影像上判讀神經、血管這類細微紋理的結構**，現在最前沿在做什麼、用什麼方法、還有什麼沒解決。
>
> 掃描範圍：CVPR / ICCV / ECCV / NeurIPS / ICLR（電腦視覺主會）＋ MICCAI / IPMI / MIDL / IEEE TMI / Medical Image Analysis（醫學影像領域）＋ grand-challenge.org 上的公開競賽。
> 產出方式：2026-08-05 由四個背景研究代理平行做真實文獻掃描（loss / 架構 / 資料集與指標 / 視覺領域前沿），原始 findings 存在 A-losses、B-architectures、C-data-metrics、D-vision-frontier，本資料夾各檔是統整版。

---

## 先講結論：這個領域現在長什麼樣

**一句話定位**：這不是一個「還沒有人做」的新領域，而是一個**已經高度成熟、但卡在一個明確瓶頸**的領域。瓶頸是：

> 神經／血管這類結構只有 1–3 個像素寬，**傳統的評分方式（Dice）根本感覺不到它斷掉**。整個 2019 年之後的研究史，就是在想辦法讓模型「不要把一條血管切成兩段」。

用具體數字說明為什麼 Dice 沒用：一張眼底照片裡，血管大概只佔 8–10% 的像素。如果模型漏掉了一段 20 像素長、2 像素寬的細血管，Dice 分數大概只掉 0.1%，但在臨床上這條血管「斷了」——拓樸（topology，指結構的連通關係，例如「這棵樹有幾個分支、有沒有斷開」）已經錯了。這就是為什麼 **clDice** 和 **boundary loss** 這兩個你提到的方法會存在。

### 五個關鍵判斷（我認為朋友最該先知道的）

1. **主戰場是「拓樸正確性」，不是「像素準確率」。**
   2019 年之後幾乎所有新方法（clDice、Betti matching、Skeleton Recall、GraphMorph）都在解同一件事：讓模型輸出的血管樹**連通關係是對的**。誰能用更低的計算成本拿到更強的拓樸保證，誰就贏。

2. **loss function 這條線已經進入「微幅改良期」。**
   2023–2026 的新論文，幾乎每一篇的自我定位都是「修正 clDice / Betti Matching 的弱點 W」，而不是提出全新的數學框架。這意味著：想靠「再發明一個 loss」進頂會，門檻已經很高。詳見 2_methods · A. Loss function 家族。

3. **基礎模型（foundation model）在這個題目上是一個有證據的死路。**
   SAM / MedSAM 這類「萬用切割模型」在細長分支結構上表現明確地差，而且 WACV 2026 有一篇診斷性論文證明**這是架構層次的問題，微調救不回來**（原因：ViT 的 patch 是 14–16 像素寬，比血管還粗，資訊在 tokenize 的當下就沒了）[verified: arXiv:2412.04243]。這是一個**可以拿來當論文起點的負面結果**。

4. **量測文化是壞的，而這本身是機會。**
   同一個資料集名稱（DRIVE、STARE）底下，不同論文用不同切分、不同指標實作，數字互相不可比。有一篇論文就是專門在盤點「100+ 篇 DRIVE/STARE 論文的方法學不一致」。除了官方競賽（TopCoW、ATM'22、ASOCA）以外，跨論文比較大多不成立。詳見 3_datasets_metrics · 4. 量測文化：這領域的數字可信嗎。

5. **視覺領域（CVPR 那邊）有一批便宜的工具，醫學這邊還沒人用。**
   例如 BlurPool（抗混疊下採樣）、PointRend / CascadePSP（邊界精修）、FeatUp / LoftUp（特徵上採樣）。這些都是「掛上去就能用」的模組，而且針對的正是「細結構在下採樣時被消滅」這個問題，但掃描沒有找到明確的醫學細長結構應用。詳見 2_methods · D. 從視覺領域搬過來的工具。

---

## 檔案結構

| 檔 | 內容 | 什麼時候看 |
|---|---|---|
| 0_overview | 本檔：領域全貌、五個關鍵判斷、怎麼用這份資料 | 先看這個 |
| 1_papers | 論文清單：12 篇必讀 + 完整分組清單（~70 篇，全部標註查證狀態） | 要決定讀什麼的時候 |
| 2_methods | 方法百科：每個方法家族的機制、解決什麼失敗、代價、已知弱點 | 要理解技術細節的時候 |
| 3_datasets_metrics | 資料集（21 個）、競賽、評估指標，含「明天就能開始做」的推薦 | 要動手前 |
| 4_open_problems | 還沒解決的問題 + 五個可以當論文的候選切入點 | 要選題的時候 |
| 5_glossary | 術語表（拓樸、persistent homology、骨架化…全部白話定義） | 隨時查 |
| `_raw/` | 四個研究代理的原始掃描，未經統整，資訊量最大 | 想追某條線的細節時 |

---

## 標註規則（沿用本 vault 慣例）

- `[verified: URL]` — 代理實際載入該頁面，確認過標題／出處／年份。
- `[verified via search]` — 只從搜尋結果片段確認，沒有直接載入頁面，可信度低一級。
- `[unverified]` — 憑記憶或二手資訊，**動手引用前必須自己再查一次**。

這個領域月更（每個月都有新的 arXiv 預印本），任何「X 沒有人做過」的主張，寫進論文前都要重新查證。特別注意本資料夾裡幾篇 2026 年的預印本（arXiv:2601.11409、2606.24935、2606.21608）都還沒有同儕審查。

---

## 給朋友的三句話總結

1. 你要做的事在學術上叫 **curvilinear structure segmentation（細長結構分割）** 或 **tubular structure segmentation**，關鍵字是 topology-preserving、connectivity、centerline，不是 "texture analysis"（那是另一個偏材質辨識的領域）。
2. 這個領域的入場券是 **clDice（CVPR 2021）+ boundary loss（MIDL 2019）+ nnU-Net baseline**，這三個是所有論文的比較基準，不熟這三個沒辦法跟人對話。
3. 最容易做出東西的起點：**DRIVE 資料集 + 拓樸指標**（不是 Dice）。理由是零門檻取得、有官方切分、而且 Dice/F1 天花板只在 0.82–0.84，還有真的進步空間——這跟大家以為的「眼底血管已經做爛了」不一樣。詳見 3_datasets_metrics · 5. 如果明天就要開始，選哪兩個資料集。


\newpage


# 1 · 論文清單

> 分兩層：先是**12 篇必讀**（不讀這些沒辦法跟這領域的人對話），再是**完整分組清單**。
> 每篇標註查證狀態，規則見 0_overview · 標註規則（沿用本 vault 慣例）。
> 方法本身的機制解說在 2_methods，這裡只講「這篇在整個故事裡的位置」。

---

## 第一層：12 篇必讀

按建議閱讀順序排列。前 5 篇是地基，後 7 篇是 2023 年後的前沿。

### 地基（一定要讀，全部是別人拿來當 baseline 的東西）

**1. clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation**
Shit et al., CVPR 2021
[verified: https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html]
→ 這個領域的中心點。把「骨架化」（skeletonization，把一條血管壓成一像素寬的中心線）寫成可微分的卷積操作，然後比較預測骨架與真值遮罩的重疊。**所有後續論文都在跟它比。** 先讀這篇。

**2. Boundary loss for highly unbalanced segmentation**
Kervadec et al., MIDL 2019 → Medical Image Analysis 2021
[verified: https://arxiv.org/abs/1812.07032]
→ 另一條線的起點。它不管拓樸，只解決「前景太小導致 Dice 數值不穩」的問題，做法是把 loss 從「面積積分」改寫成「沿著輪廓的距離積分」。你提到的兩個方法之一。

**3. Topology-Preserving Deep Image Segmentation**
Hu, Li, Samaras, Chen, NeurIPS 2019
[verified: https://proceedings.neurips.cc/paper_files/paper/2019/file/2d95666e2649fcfc6e3af75e09f5adb9-Paper.pdf]
→ 拓樸損失的數學正統起源，用 persistent homology（持續同調，代數拓樸的工具）算「預測圖與真值圖的拓樸特徵差距」。比 clDice 嚴謹但貴很多。理解這篇才知道 clDice 是在「近似」什麼。

**4. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation**
Isensee et al., Nature Methods 2021
[verified: https://github.com/mic-dkfz/nnunet]
→ 不是細長結構專用，但它是**整個醫學影像分割的預設強 baseline**。任何新方法沒有贏過 nnU-Net 就沒有說服力。2024 年後開始有方法（U-Mamba 等）宣稱贏過它，但不是無條件的。

**5. Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation (DSCNet)**
Qi et al., ICCV 2023
[verified: https://arxiv.org/abs/2307.08388]
→ 架構端最漂亮的一個想法：卷積核不再是方形，而是**沿著血管的走向蜿蜒變形**（像蛇一樣）。直觀好懂，而且同時報告了準確率與連通性的提升。

### 前沿（2023 年後，決定現在該做什麼）

**6. Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes (Betti Matching)**
Stucki et al., ICML 2023
[verified: https://arxiv.org/abs/2211.15272]
→ 修正第 3 篇的核心弱點：舊方法只確認「拓樸特徵的數量」對不對，不管**位置**對不對。這篇用「induced matching」把預測與真值的拓樸特徵一一空間配對。後續有 Efficient Betti Matching（2024）純加速版。

**7. Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures**
Kirchhoff et al. (nnU-Net 團隊), ECCV 2024
[verified: https://arxiv.org/abs/2404.03010]
→ clDice 的實用化：只對**真值**做骨架化（可以離線在 CPU 上先算好），不用每次迭代都在 GPU 上算，宣稱降低約 90% 的計算成本，而且是第一個宣稱支援多類別的拓樸損失。想做 3D 的話這篇最重要。

**8. Topograph: An efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation**
Lux et al., ICLR 2025
[verified: https://arxiv.org/abs/2411.03228]
→ 目前拓樸損失這條線最新的「正統」。把拓樸關係編碼成連通元件圖（component graph），宣稱有嚴格的同倫等價（homotopy equivalence）保證，而且比 persistent homology 系的方法快約 5 倍。

**9. Centerline Boundary Dice Loss for Vascular Segmentation (cbDice)**
Shi et al., MICCAI 2024
[verified: https://arxiv.org/abs/2407.01517]
→ 修正 clDice 的「管徑盲」問題：clDice 只看骨架有沒有被打到，所以粗血管跟細血管的誤差被同等對待，實際訓練起來會偏向大血管。cbDice 用局部半徑加權。**如果朋友要做細微神經／微血管，這個弱點正是他的痛點。**

**10. GraphMorph: Tubular Structure Extraction by Morphing Predicted Graphs**
Zhang et al., NeurIPS 2024
[verified: https://arxiv.org/abs/2502.11731]
→ 代表另一種思路：不要輸出像素遮罩，直接輸出**圖（graph）**——節點是分岔點、邊是血管段——然後再把圖「變形」成遮罩。從根本上避免了「像素級預測不保證全域連通」的問題。

**11. Quantifying the Limits of Segmentation Foundation Models: Modeling Challenges in Segmenting Tree-Like and Low-Contrast Objects**
WACV 2026
[verified: https://arxiv.org/abs/2412.04243]
→ **最重要的負面結果。** 系統性地證明 SAM 家族在「樹狀結構」與「低對比」物體上的失敗率與這兩個性質強相關，而且**針對性微調解決不了**。這篇讓「我為什麼不直接用 SAM」這個必問問題有標準答案。

**12. vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation**
Wittmann et al., CVPR 2025
[verified: https://arxiv.org/abs/2411.17386]
→ 對第 11 篇的回應：既然通用基礎模型不行，那就做血管專用的。用真實資料 + 域隨機化合成資料 + 生成資料三種來源訓練，做到零樣本／少樣本泛化。**2026 年做血管的人繞不開這個 baseline。**

---

## 第二層：完整分組清單

### A. Loss function — 拓樸保持（persistent homology 系）

數學正統路線，保證強但貴。演化線：TopoLoss → Clough → Betti Matching → 加速與空間修正。

| 論文 | 出處/年 | 一句話 | 查證 |
|---|---|---|---|
| Topology-Preserving Deep Image Segmentation | NeurIPS 2019 | persistence diagram 距離當 loss，起源 | [verified: proceedings.neurips.cc] |
| A Topological Loss Function... using Persistent Homology (Clough et al.) | TPAMI 2020/2022, arXiv:1910.01877 | 只給目標 Betti 數當先驗，不需要完整拓樸真值 | [verified: arXiv:1910.01877] |
| Betti Matching (Stucki et al.) | ICML 2023, arXiv:2211.15272 | 空間配對，修「數量對但位置錯」 | [verified] |
| Efficient Betti Matching | arXiv:2407.04683, 2024 | C++ 實作加速，讓 3D 可用 | [verified] |
| Spatial-Aware Persistent Feature Matching | arXiv:2412.02076, 2024 | 把影像空間資訊加進配對步驟 | [verified: 標題方法；成本宣稱未證實] |
| PI-Att | arXiv:2408.08038, 2024 | 用 persistence image（向量化表示）而非 barcode | [verified] |
| TopoSculpt | arXiv:2509.03938, 2025 | 全體積建模 + Betti 完整性約束 + 課程式精修 | [verified] |
| Topograph | ICLR 2025, arXiv:2411.03228 | 連通元件圖，嚴格保證且快 5 倍 | [verified: OpenReview] |

### B. Loss function — 骨架／中心線（clDice 系）

工程實用路線，便宜但保證弱。演化線就是一連串「修 clDice 的某個弱點」。

| 論文 | 出處/年 | 一句話 | 查證 |
|---|---|---|---|
| clDice | CVPR 2021 | 可微分軟骨架化 + 中心線 Dice | [verified] |
| Homotopy Warping | NeurIPS 2022, arXiv:2112.07812 | 用數位拓樸找「關鍵像素」，只在那裡算 loss | [verified] |
| Skeleton Recall Loss | ECCV 2024, arXiv:2404.03010 | 只骨架化真值、離線 CPU 計算，省 90% 成本 | [verified] |
| cbDice | MICCAI 2024, arXiv:2407.01517 | 用血管半徑加權，修管徑不平衡 | [verified: papers.miccai.org] |
| Centerline Cross-Entropy (clCE) | MICCAI 2024 | 底座從 Dice 換成 CE，兼顧穩健與拓樸 | [verified: Springer] |
| ContextLoss | ICIP 2025, arXiv:2506.11134 | 把關鍵像素擴張成鄰域再罰 | [verified] |
| Smooth clDice | LRDE 技術報告 2025 | 加「不確定區」以抵抗 1 像素標註噪聲 | [unverified 細節：PDF 無法解析] |
| Topology-Guaranteed Segmentation（連通+虧格+寬度） | arXiv:2601.11409, 2026-01 | 首次把「寬度」折進拓樸約束，用 PDE 平滑 | [verified，但未經審查] |
| TopoVST | arXiv 2026, 審查中 | 針對血管骨架追蹤階段的拓樸保真 | [unverified：僅索引頁] |

### C. Loss function — 邊界／距離

不碰拓樸，處理「前景太小」與「邊界不準」。

| 論文 | 出處/年 | 一句話 | 查證 |
|---|---|---|---|
| Boundary loss | MIDL 2019 / MedIA 2021 | 輪廓距離積分取代區域積分 | [verified] |
| Hausdorff Distance loss | IEEE TMI 2020 | 直接優化最壞情況邊界距離的三種可微近似 | [verified: PubMed 31329113] |
| Active Contour loss | CVPR 2019 | 經典 snake 的長度項 + 區域項寫成 loss | [verified via 索引] |
| How Distance Transform Maps Boost Segmentation CNNs | MIDL 2020 | 五種距離轉換 loss 的實證比較 | [verified: OpenReview] |
| Boundary Difference over Union (BoundaryDoU) | MICCAI 2023, arXiv:2308.00220 | 純區域公式但自動加權邊界帶 | [verified] |
| Region Mutual Information (RMI) loss | NeurIPS 2019 | 用 3×3 鄰域的互資訊取代逐像素獨立假設 | [verified] |

### D. 架構 — CNN 與注意力（仍是主要 baseline）

| 論文 | 出處/年 | 一句話 | 查證 |
|---|---|---|---|
| CS-Net / CS2-Net | MICCAI 2019 / MedIA 2020 | 通道+空間注意力，第一個明確針對 curvilinear 的架構 | [verified] |
| DeepVesselNet | Front. Neurosci. 2020, arXiv:1803.09340 | 十字型（cross-hair）濾波器近似 3D 卷積，省算力 | [verified] |
| FR-UNet | IEEE JBHI 2022 | 全解析度分支，不做下採樣瓶頸 | [verified: GitHub] |
| SGL (Study Group Learning) | MICCAI 2021, arXiv:2103.03451 | 針對「人工血管標註本身有噪聲」的集成訓練法 | [verified] |
| TransUNet | arXiv:2102.04306, 2021 | CNN 特徵上接 ViT，通用醫學分割 | [verified] |
| Swin-UNet | arXiv:2105.05537 | 純 Transformer U 型架構 | [verified] |
| DSCNet（動態蛇形卷積） | ICCV 2023 | 卷積核沿管狀走向變形 | [verified] |
| Serp-Mamba | IEEE TMI 2025, arXiv:2409.04356 | Mamba 狀態空間模型 + 蜿蜒掃描順序，處理超高解析眼底 | [verified] |
| U-Mamba | arXiv:2401.04722, 2024 | CNN+SSM 混合，宣稱贏 nnU-Net（非血管專用） | [verified] |

### E. 架構 — 圖表示與追蹤（我認為最有前途的一條）

| 論文 | 出處/年 | 一句話 | 查證 |
|---|---|---|---|
| Relationformer | ECCV 2022, arXiv:2203.10202 | 單階段 transformer 同時預測節點與邊，可直接做 3D | [verified] |
| VesselGraph | NeurIPS 2021 D&B, arXiv:2108.13233 | 全腦血管圖資料集與 GNN benchmark | [verified] |
| GraphMorph | NeurIPS 2024, arXiv:2502.11731 | 先預測分支級圖，再變形成遮罩 | [verified] |
| RoadTracer | CVPR 2018 | 非醫學，迭代式「下一步往哪走」的圖搜尋追蹤 | [verified] |
| Sat2Graph | ECCV 2020, arXiv:2007.09547 | 非醫學，一次前向就輸出整張圖 | [verified] |
| Deep RL for Vessel Centerline Tracing | MICCAI 2018 | DQN 代理逐步走過 3D 體積追中心線 | [unverified：Springer 頁未載入] |

### F. 基礎模型（foundation model）與其極限

| 論文 | 出處/年 | 一句話 | 查證 |
|---|---|---|---|
| Segment Anything (SAM) | ICCV 2023 | 可提示的通用分割模型，1.1B 遮罩訓練 | [verified] |
| SAM 2 | arXiv:2408.00714, 2024 | 加上影片與串流記憶，細結構弱點沿用 | [verified] |
| MedSAM | Nature Comms 2024 | 157 萬醫學影像對微調的 SAM | [verified: GitHub] |
| **Quantifying the Limits of Segmentation Foundation Models** | **WACV 2026, arXiv:2412.04243** | **證明樹狀/低對比失敗是架構問題，微調無效** | [verified] |
| vesselFM | CVPR 2025, arXiv:2411.17386 | 3D 血管專用基礎模型，零/少樣本 | [verified] |
| VesselSAM (AtrousLoRA) | arXiv:2502.18185, 2025 | 用 LoRA 只訓 7% 參數改造 SAM，主動脈 +12～13 分 | [verified] |
| UCS: Universal Model for Curvilinear Structure Segmentation | arXiv:2504.04034, 2025 | 稀疏 adapter + 傅立葉提示，橫跨醫學與非醫學曲線結構 | [verified] |
| DINOv2 / DINOv3 | arXiv:2304.07193 / 2508.10104 | 自監督密集特徵骨幹；patch 大小是細結構的隱憂 | [verified] |

### G. 標註稀缺與自監督

| 論文 | 出處/年 | 一句話 | 查證 |
|---|---|---|---|
| YoloCurvSeg | MedIA, arXiv:2212.05566 | 只標一條有噪聲的骨架，用背景修補+曲線生成合成訓練對；1.4% 標註量達 97% 全監督效能 | [verified] |
| C-DARL | MedIA 2023 | 對比式擴散對抗表示學習，無標籤血管分割 | [unverified：期刊頁 403] |
| CurvSegFlow | arXiv:2606.21608, 2026-06 | 時間條件流匹配（flow matching）做迭代精修，針對低訊噪比 | [verified，未經審查] |

### H. 視覺領域可搬用的工具（非醫學）

| 論文 | 出處/年 | 為什麼相關 | 查證 |
|---|---|---|---|
| BlurPool（Making ConvNets Shift-Invariant Again） | ICML 2019 | 下採樣前加低通濾波，避免細結構被混疊消滅。改動只有幾行 | [verified] |
| Alias-Free ViT | NeurIPS 2025, arXiv:2510.22673 | 證明 ViT 也有同樣的混疊問題並給修法 | [verified] |
| PointRend | CVPR 2020 | 只在不確定的邊界點上做全解析度精修 | [verified] |
| CascadePSP | CVPR 2020, arXiv:2005.02551 | 類別無關的邊界精修網路，可事後掛在任何分割器後面 | [verified] |
| FeatUp | ICLR 2024, arXiv:2403.10516 | 模型無關的特徵上採樣，救 ViT patch 造成的解析度損失 | [verified] |
| LoftUp | ICCV 2025 | FeatUp 的後繼，座標式上採樣 | [verified] |
| Scaling Laws in Patchification | arXiv:2502.03738, 2025 | 實證 patch 越小效能越好，即 patch 化本身是有損壓縮 | [verified] |
| Wavelet CNNs for Texture Classification | CVPR Workshop 2017 | 小波多解析度分解放進 CNN，細結構住在高頻子帶 | [verified] |
| Octave Convolution | ICCV 2019 | 顯式切分高頻/低頻運算預算 | [verified] |
| ImageNet-trained CNNs are biased towards texture | ICLR 2019 | 提供「模型到底看的是紋理還是結構」的診斷實驗設計 | [verified] |
| Deep Image Matting | CVPR 2017 | 把細結構當成連續 alpha 值而非二元遮罩的重構思路 | [verified] |
| DeepCrack | Neurocomputing 2019 | 裂縫分割，與醫學細結構問題形狀幾乎相同 | [verified] |
| SCSegamba | CVPR 2025, arXiv:2503.01113 | 輕量 Mamba + 結構感知掃描做裂縫 | [verified] |
| TopoMortar | BMVC 2025 (oral), arXiv:2503.03365 | 受控合成細結構 benchmark，用來隔離「拓樸真的變好」與「其他混淆因素」 | [verified] |
| SDF-TopoNet | arXiv:2503.14523, 2025 | 先用符號距離場預訓練，再拓樸精修 | [verified] |
| SEMIR | arXiv:2606.24935, 2026-06 | 把像素塌縮成邊界對齊的超節點圖，全解析度推論不切塊 | [verified，極新且未審查] |
| Hallucination Score | arXiv:2507.14367, 2025 | 量化生成式超解析「幻覺出來的細節」——診斷影像的安全紅線 | [verified] |

---

## 讀的時候要注意的陷阱

1. **「SOTA」宣稱幾乎都是跟前一篇比。** 掃描沒有找到任何 2026 年的獨立第三方 leaderboard 把這些方法放在同一個 protocol 下重排名。看到「we achieve SOTA」請當成「在作者選的設定下贏了作者選的 baseline」。
2. **小波／頻域這一串（WaveRNet、FreqUNet、WDM-UNet…）互相引用但沒有獨立複現。** 概念合理，證據還沒鞏固。
3. **擴散模型做曲線結構分割目前是敘事多於證據。** CurvSegFlow 是 2026-06 的預印本，C-DARL 沒有被獨立驗證。
4. **2026 年的幾篇（2601.11409、2606.24935、2606.21608）都還沒同儕審查**，引用時要標明。


\newpage


# 2 · 方法百科

> 這份的目的是**理解機制**，不是列清單（清單在 1_papers）。
> 每個方法回答四件事：解決什麼失敗 → 怎麼做 → 代價 → 已知弱點。
> 術語第一次出現時會白話定義，完整術語表在 5_glossary。

---

## 先講清楚：這個領域到底在對抗什麼

三個具體的失敗模式，所有方法都是在打其中一個：

| 失敗 | 長什麼樣 | 為什麼 Dice 抓不到 |
|---|---|---|
| **斷裂（breakage）** | 一條連續血管被切成兩段 | 缺的可能只有 5 個像素，Dice 幾乎不動 |
| **管徑錯誤（width error）** | 血管連通關係對，但粗細畫錯 | 細血管本來像素就少，權重天生被壓低 |
| **邊界模糊（boundary blur）** | 輪廓不準，糊成一團 | 細結構的「邊界像素」幾乎就是全部像素，但 Dice 對每個像素一視同仁 |

再加一個系統性成因：**下採樣（downsampling）**。U-Net 這類架構為了看得更廣，會把影像逐步縮小（例如 512→256→128）。一條 1 像素寬的血管在第一次縮小後就有一半機率消失，之後解碼器再怎麼放大也救不回來。**這是所有架構端方法的共同敵人。**

---

## A. Loss function 家族

Loss function（損失函數）＝告訴模型「你錯得多離譜」的計分規則。改 loss 是這個領域最主流的做法，因為不用改架構、成本低。

### A1. clDice 系（骨架路線）— 便宜、直觀、有明確弱點

**核心想法**：如果我把預測的血管和真值的血管都「壓成一像素寬的中心線」（這個操作叫 **skeletonization，骨架化**），那斷裂就無所遁形——中心線斷了就是斷了。

**clDice 的技術貢獻**是讓骨架化**可微分**（differentiable，指可以計算梯度、能反向傳播訓練）。傳統骨架化是離散的形態學操作，沒有梯度。clDice 用「反覆的軟性侵蝕與膨脹」（soft erosion / dilation，寫成卷積層）近似它，於是可以塞進訓練迴圈。

計分方式是雙向的：
- **拓樸精確度**＝預測的骨架有多少落在真值遮罩裡（罰假血管）
- **拓樸敏感度**＝真值的骨架有多少落在預測遮罩裡（罰漏血管）
- clDice ＝ 兩者的調和平均

**代價**：每次迭代都要在 GPU 上跑數次侵蝕/膨脹，3D 體積上會很吃記憶體。

**已知弱點（重要，這是後續所有論文的切入點）**：
1. **管徑盲**：骨架只有一像素，所以「打中一條粗血管的骨架」和「打中一條細血管的骨架」得分一樣。實際訓練起來，模型會偏向把粗血管做好，因為粗血管在 Dice 那一項貢獻大。→ cbDice 用局部半徑加權來修。
2. **假設「骨架像素＝拓樸關鍵像素」，這常常是錯的。** 真正決定連不連通的關鍵像素，可能在骨架旁邊。→ ContextLoss 把關鍵像素擴張成一個鄰域。
3. **對標註噪聲敏感**：真值本身就是人畫的，1 像素寬的結構在不同標註者之間根本不一致，軟骨架化在這種噪聲上會不穩定甚至訓練發散。→ Smooth clDice 加「不確定區」（但只是技術報告，未審查）。
4. **貴**：→ Skeleton Recall Loss 只對真值骨架化（可以離線在 CPU 上預先算好），省約 90% 計算，代價是變成單向的（只罰漏、不罰多），所以必須配一個精確度側的 loss 一起用。

### A2. Persistent homology 系（拓樸路線）— 數學嚴謹、貴

**要先懂的概念**：**persistent homology（持續同調）**是代數拓樸的工具。白話講：把模型輸出的機率圖想成一座地形，然後把水位從高到低慢慢降。過程中會有「島嶼冒出來」（一個連通元件誕生）、「兩座島連起來」（一個元件消滅）、「湖被圍出來」（一個洞誕生）。把每個特徵的「誕生水位」和「死亡水位」記下來，就得到一張 **persistence diagram（持續圖）**。它完整編碼了這張圖在所有閾值下的拓樸結構。

**Betti number（貝蒂數）**則是更粗的摘要：β₀＝有幾個連通塊、β₁＝有幾個洞。

**TopoLoss（NeurIPS 2019）** 的做法：算預測與真值的 persistence diagram 之間的距離當 loss，梯度會反向傳到「造成拓樸特徵誕生/消滅的那幾個關鍵像素」上。

**演化史就是一連串補洞：**

| 版本 | 修了什麼 |
|---|---|
| Clough et al. (TPAMI) | 不需要完整拓樸真值，只要給「應該有幾個洞」這種先驗即可訓練 |
| **Betti Matching (ICML 2023)** | 修「數量對但位置錯」——用 induced matching 把兩邊的拓樸特徵**空間上一一配對**再比 |
| Efficient Betti Matching (2024) | 純工程加速（自寫 C++），讓 3D 體積可用。注意：是「可用」不是「快」 |
| Spatial-Aware Matching (2024) | 把影像空間資訊加進配對步驟，避免 barcode 上接近但空間上遙遠的錯配 |
| Topograph (ICLR 2025) | 改用連通元件圖來找「拓樸關鍵區域」，宣稱嚴格保證且比 PH 系快 5 倍 |

**共同代價**：persistent homology 是全域、序列式的演算法，很難平行化與批次化。這是它輸給 clDice 的唯一原因，也是每一篇後續論文的主題。

**共同弱點**：只管拓樸，**完全不管寬度**。一條血管被畫得太細但連通關係正確，這類 loss 給滿分。對做微血管／神經纖維的人來說這是實質問題。2026-01 有一篇預印本首次嘗試把寬度折進拓樸約束（arXiv:2601.11409），但用了 PDE 求解器，複雜度高且未審查。

### A3. 邊界／距離系 — 不碰拓樸，解決類別不平衡

**問題**：一張影像裡血管只佔 8%，背景佔 92%。Dice 或交叉熵是對「區域」做積分，兩個類別的積分量級差好幾個數量級，數值上不穩定。

**Boundary loss（你提到的第二個方法）的核心手法**：把 loss 從「面積積分」**改寫成「沿著輪廓的距離積分」**。實作上是預先算好真值的 **signed distance map（符號距離圖：每個像素到最近邊界的距離，內部為負外部為正）**，訓練時把預測機率乘上這張圖再積分。因為輪廓是一維的、面積是二維的，兩個類別的量級就對齊了。

- **代價**：需要預先算距離圖（前處理）。
- **弱點**：單獨用會不穩定（訓練初期預測還很亂的時候，距離項的梯度會亂拉），實務上一定要跟區域 loss（Dice/CE）加權混用，而且權重要隨訓練排程調整。

**同家族的其他成員**：
- **Hausdorff distance loss（TMI 2020）**：直接優化「最壞情況的邊界距離」。給了三種可微分近似（距離轉換、形態學侵蝕、多半徑圓形卷積）。因為 HD 是取最大值的指標，可微近似必然是平滑過的，準確性與梯度友善之間有取捨。
- **Active Contour loss（CVPR 2019）**：把經典 snake 模型的「輪廓長度項 + 區域項」寫成 loss。**注意這個對細長結構可能有害**：長度項會懲罰長的邊界，但血管天生周長／面積比就很大，這會把模型往「更短更平滑（也就是更錯）」的方向推。
- **BoundaryDoU（MICCAI 2023）**：純區域公式，但用一個隨物體大小自適應的 α 去放大邊界帶的權重，不需要距離圖。
- **RMI loss（NeurIPS 2019）**：把每個像素連同 8 個鄰居當成 9 維點，最大化預測與真值分布的互資訊下界。是「不要假設像素獨立」這條線的代表，但 3×3 鄰域對「整條血管的連續性」來說可能太小。

### A4. 這三家的關係圖

```
      解決「前景太小」          解決「連通性斷裂」
            │                         │
    ┌───────┴────────┐      ┌─────────┴──────────┐
  Boundary loss   Active   clDice 系            Persistent homology 系
  (MIDL'19)       Contour  (幾何近似，便宜)      (代數拓樸，嚴謹)
      │           (CVPR'19)      │                      │
   HD loss                  Skeleton Recall        Betti Matching
   DT maps                  cbDice / clCE          Topograph
   BoundaryDoU              ContextLoss            TopoSculpt
                                 └──── Homotopy Warping ────┘
                                      （兩家的交會點）
```

**2023 年之後的現況**：幾乎每一篇新論文的自我定位都是「修正 clDice 或 Betti Matching 的弱點 W」。這個子領域已經進入**微幅改良期**，不是新數學框架湧現期。含義：想靠新 loss 進頂會，要嘛找到一個沒人處理過的弱點（寬度、標註噪聲、多類別），要嘛換戰場。

---

## B. 架構家族

### B1. 抗下採樣（保住細節）

- **FR-UNet**：整條網路保留一個全解析度分支，不讓細血管在瓶頸處消失；推論時再用雙閾值迭代把低信心的細分支撿回來。
- **DeepVesselNet**：3D 卷積太貴，改用三個正交平面的「十字型濾波器」近似 3D 感受野；另外針對「血管體素 <3%」設計了類別平衡的損失。
- **BlurPool（來自視覺領域，ICML 2019）**：在每個 stride-2 下採樣前加一個低通濾波。原理是訊號處理的**抗混疊（anti-aliasing）**——高頻訊號（＝細結構）在降採樣時會摺疊成假訊號。改動只有幾行，掃描沒找到明確的醫學細長結構應用。

### B2. 形狀感知的卷積

- **DSCNet（動態蛇形卷積，ICCV 2023）**：讓卷積核的取樣點**沿著血管走向蜿蜒偏移**，而不是固定的方形鄰域。直觀理由：一條 1 像素寬的彎曲血管，用 3×3 方形核去看，9 個取樣點裡有 6 個是背景，資訊被稀釋。加上多視角特徵融合與一個 persistent-homology 連續性損失。同時報告了準確率與連通性提升。

### B3. 圖表示（我認為最有前途的一條）

**核心轉念**：不要輸出「每個像素是不是血管」，直接輸出**圖**——節點是分岔點與端點，邊是血管段。這樣連通性是模型的**輸出格式本身保證的**，不是像素預測的副產品。

- **Relationformer（ECCV 2022）**：單階段 transformer 同時預測節點 token 與關係 token，不走「先偵測再連線」的兩階段流程；是第一個直接在 3D 做血管圖的方法。
- **GraphMorph（NeurIPS 2024）**：先用 Graph Decoder 預測分支級的圖，再用 Morph Module 把圖與中心線機率圖融合成拓樸一致的遮罩。等於是「用圖來監督像素」。
- **RoadTracer / Sat2Graph（非醫學）**：道路網路萃取，問題形狀與血管樹幾乎相同。RoadTracer 是迭代式「下一步往哪走」，Sat2Graph 是一次前向輸出整張圖。這兩種設計選擇的取捨可以直接搬過來。
- **VesselGraph（NeurIPS 2021 D&B）**：全腦血管圖資料集，如果要做 GNN 方向這是起點。

**代價**：圖表示需要圖形式的真值（節點/邊），大部分醫學資料集只有像素遮罩。要嘛自己從遮罩骨架化生成圖（會引入誤差），要嘛選有中心線標註的資料集（ROSE、TubeTK、ASOCA、BigNeuron）。

### B4. 長程依賴：Transformer 與 Mamba

- **TransUNet / Swin-UNet**：通用醫學分割，不是細長結構專用，在細血管 benchmark 上很少是 SOTA。列出來是因為所有論文都會拿來當 baseline。
- **Serp-Mamba（TMI 2025）**：Mamba 是**狀態空間模型（state space model, SSM）**，可以用線性複雜度處理長序列（Transformer 是平方複雜度）。Serp-Mamba 的貢獻是**掃描順序**：一般 SSM 按 raster order（一行一行）掃，這會把一條斜向血管切得支離破碎；它改成沿著血管路徑蜿蜒掃描。針對超廣角眼底影像這種超高解析度場景。
- **hype check**：Mamba 在醫學影像的整體證據還在早期。U-Mamba 宣稱贏 nnU-Net，但測的是腹部器官與內視鏡，不是細血管。

### B5. 基礎模型：一個有證據的死路，以及兩個繞路

**死路（證據強）**：SAM / SAM2 / MedSAM 在細長分支結構上明確地差。WACV 2026 那篇診斷論文的結論是：失敗率與物體的「樹狀程度」及「低對比程度」強相關，而且**針對性微調無法解決**——這是架構層次的限制。

**機制解釋**（1_papers 的 D 組提供了佐證）：ViT 把影像切成 14×14 或 16×16 像素的 patch 再 tokenize。一條 2 像素寬的血管在一個 patch 裡只佔 1%，資訊在 tokenize 的當下就被平均掉了。arXiv:2502.03738 實證了「patch 越小效能越好」，也就是 patch 化本身就是有損壓縮。**這種失敗特別危險，因為它不會表現為「低信心」，模型是根本沒有表示那個資訊。**

**繞路 1 — 領域專用基礎模型**：vesselFM（CVPR 2025）用真實 + 域隨機化合成 + 生成三種資料訓 3D 血管專用模型，零/少樣本泛化。目前是最強的通用起點，但評估範圍還窄。

**繞路 2 — adapter 微調**：VesselSAM 用 Atrous Attention + LoRA（low-rank adaptation，只訓練少量低秩矩陣而凍結主幹）只訓 ~7% 參數，在主動脈資料上比 SAM-ViT-b baseline 高 12–13 個 Dice 點。UCS（arXiv:2504.04034）則用稀疏 adapter + 傅立葉提示生成，橫跨醫學與非醫學曲線結構——**這篇是最直接的競爭者，要仔細讀。**

### B6. 標註稀缺的處理

醫學細結構的標註成本極高（一張眼底照片的血管遮罩要標數小時），所以這條線很重要。

- **YoloCurvSeg**：只要**一條有噪聲的骨架標註**，用背景修補 + 演算法生成曲線來合成訓練對，達到全監督 97% 的效能、只用 1.4% 的標註量。在 OCTA-500、CORN、DRIVE、CHASE_DB1 上驗證。
- **SGL**：不是減少標註量，而是承認**現有標註本身有噪聲**，用「學習小組」集成 + 平滑偽標籤來避免過擬合噪聲。
- **C-DARL / CurvSegFlow**：擴散與流匹配路線，證據都還早。

---

## C. 表示層的替代思路（比較少人走，可能是機會）

1. **Alpha matting 重構**：Deep Image Matting（CVPR 2017）把細髮絲當成連續的不透明度值而非二元遮罩。血管在成像解析度極限附近本來就是**部分容積效應（partial volume effect：一個像素同時包含血管與背景）**，用連續 alpha 表示在物理上比二元標籤更誠實。掃描沒找到醫學細長結構的 alpha matting 應用。障礙：需要 trimap（三分圖：確定前景/確定背景/待定區）這種先驗。
2. **符號距離場（SDF）當輔助目標**：SDF-TopoNet（2025）先用 SDF 回歸預訓練再做拓樸精修。SDF 天然帶有寬度資訊，這正是拓樸 loss 缺的東西。
3. **超節點圖塌縮**：SEMIR（arXiv:2606.24935，2026-06）把數百萬像素塌縮成邊界對齊的超節點小圖，讓 21 MP 影像可以不切塊做全解析度推論——切塊（tiling）本身就是斷裂的來源之一。極新、未審查。

---

## D. 從視覺領域搬過來的工具（低成本、可能還沒人做）

這一組的共同特徵是**幾乎不用改主架構就能掛上去**，而且針對的正好是細結構的痛點。

| 工具 | 解決什麼 | 搬過來的難度 | 醫學細結構應用是否存在 |
|---|---|---|---|
| **BlurPool** | 下採樣的混疊消滅細結構 | 極低（換掉 stride-2 層） | 沒找到明確論文（信心：中） |
| **PointRend** | 邊界被低解析度遮罩上採樣糊掉 | 低（掛一個精修頭） | 一般醫學分割有，細長結構未確認 |
| **CascadePSP** | 同上，且類別無關、可事後掛 | 低（不用重訓主模型） | 未確認 |
| **FeatUp / LoftUp** | ViT patch 造成的特徵解析度損失 | 低（模型無關，有開源） | 沒找到（信心：中高，純粹是太新） |
| **Octave Conv** | 高頻/低頻運算預算的顯式切分 | 中 | 沒找到（但這技術 2019 後沒人跟進，要先查為什麼） |
| **小波層** | 細結構住在高頻子帶 | 中 | **有一堆**（WaveRNet 等），但互相引用、未獨立複現 |
| **Geirhos 紋理/形狀偏誤診斷** | 檢查模型看的到底是紋理還是結構 | 低（是實驗設計不是模組） | 未在醫學細紋理上確認 |
| **TopoMortar 的實驗設計** | 用受控合成資料隔離「拓樸真的變好」與混淆因素 | 低（方法論可直接抄） | 該論文本身是非醫學 |

---

## E. 危險清單：在診斷場景會出事的做法

這一節很重要，因為朋友做的是醫學影像，這些在自然影像上是加分項，在診斷上是紅線。

1. **擴散／GAN 超解析或紋理增強，用在「人或模型下判斷之前」的診斷影像上。**
   感知—失真取捨（perception-distortion trade-off）是被證明的數學結果：**你不可能同時最大化「看起來真實」與「忠於原始資訊」**。看起來更銳利的神經纖維紋理，可能是模型合理編造出來的。[verified: arXiv:2507.14367]。這是最高風險項。
2. **用感知損失（VGG / LPIPS）訓練任何醫學影像的復原或增強網路。** 感知損失的定義就是「長得像我看過的紋理」，這正是產生「自信但錯誤」的細結構的機制。
3. **不檢查 patch 大小就信任 SAM / DINOv2 的凍結特徵。** 比 patch 還細的結構會被靜默丟棄，而且**不會有低信心訊號**。
4. **把風格化／紋理隨機化增強當成通用穩健性配方。** 在自然影像上它是把模型從紋理依賴推向形狀依賴；但在「病理本身就是紋理」的領域，這等於訓練模型忽略要研究的訊號。只能當受控 ablation，不能當預設增強。
5. **全域池化型的紋理描述子（bilinear pooling 等）用在需要精確定位的地方。** 它們的設計目的就是平移不變，而平移不變會摧毀「病灶在哪」的資訊。


\newpage


# 3 · 資料集、競賽與評估指標

> 動手前先讀這份。重點結論在最後兩節：3_datasets_metrics · 4. 量測文化：這領域的數字可信嗎 與 3_datasets_metrics · 5. 如果明天就要開始，選哪兩個資料集。
> 原始掃描：C-data-metrics。

---

## 1. 資料集總表

「還有空間嗎」這欄是最重要的——很多人以為眼底血管已經做爛了，實際上不是。

### 2D 眼底（fundus）— 入門首選

| 名稱 | 影像數 / 解析度 | 標註 | 取得 | 現況 | 還有空間嗎 |
|---|---|---|---|---|---|
| **DRIVE** | 40 張（20 訓練/20 測試），768×584 | 逐像素血管遮罩，測試集有 2 位獨立標註者 | Grand Challenge，免費註冊 [verified: https://drive.grand-challenge.org/] | Dice/F1 約 **0.82–0.84**；論文常報的 0.95+ 是 accuracy（被背景像素灌水），不是 Dice | **有**，而且有官方切分 |
| **STARE** | 20 張，700×605 | 逐像素，2 位標註者，含病變眼 | 官方頁 TLS 憑證錯誤，多處鏡像 [verified via search] | **沒有官方切分**，每篇論文自己切 | 有，但可比性從根本就壞掉 |
| **CHASE_DB1** | 28 張（14 位兒童雙眼），999×960 | 逐像素，2 位標註者 | Kingston 大學資料庫，CC-BY（本次連線逾時） | 無中央追蹤 | 有，但 N 太小 |
| **HRF** | 45 張（健康/DR/青光眼各 15），高解析 | 逐像素金標準 + FOV 遮罩 + 視神經盤 | 免費，CC-BY 4.0 [verified: https://www5.cs.fau.de/research/data/fundus-images/] | 無中央追蹤 | 有；四個經典集裡視野最大 |
| **IOSTAR** | 30 張，1024×1024，糖尿病視網膜病變 | 逐像素 + 動靜脈分類 + 視神經盤 | IDIAP 鏡像 [verified via search] | 無中央追蹤 | 有 |
| **FIVES** | **800 張**，2048×2048，DR/AMD/青光眼/正常各 200 | 3 位眼科醫師共識 + 24 位住院醫師，附品質分級 | Figshare 開放 [verified via search] | 最大的公開眼底血管集 | **有**，適合做跨疾病泛化 |

### OCT-A（光學同調斷層掃描血管造影）

| 名稱 | 規模 | 標註 | 取得 | 備註 |
|---|---|---|---|---|
| **ROSE-1/2** | 229 張，ROSE-1 有 117 張/39 位受試者 | **同時有中心線級與像素級標註**（很罕見） | Zenodo + iMED-Lab，學術用途 [verified via search] | 想做圖/中心線方法的話，雙標註很有價值 |
| **OCTA-500** | 500 位受試者，6 種投影，標大血管/微血管/動靜脈/FAZ/視網膜層 | 多任務 | IEEE DataPort，**需寄信索取密碼** | 標註最豐富但門檻最高 |

### X 光冠狀動脈造影

| 名稱 | 規模 | 取得 | 備註 |
|---|---|---|---|
| **XCAD** | 126 張有標註（84/42）+ 1621 對未配對影像 | 公開，但正規下載連結未確認 | 大量未標註資料 → 適合自監督 |
| **DCA1** | 130 張，300×300 | CIMAT 鏡像免費 [verified via search] | 近期輕量網路論文 F1 在 0.80s–0.90s |

### 3D — CT / MRA

| 名稱 | 規模 | 標註 | 取得 | 還有空間嗎 |
|---|---|---|---|---|
| **ImageCAS** | **~1000 個冠狀動脈 CTA 體積** | 體素級，2 位放射科醫師 + 第 3 位仲裁，左右冠脈分開 | Kaggle + 官方 GitHub，**有官方切分**（罕見） | 有，Dice 在 0.80–0.85；規模夠做有統計意義的 ablation |
| **ASOCA** | 40 訓練 + 隱藏測試 | 體素遮罩 **+ 中心線 + 局部半徑** | Grand Challenge + UK Data Service（兩層註冊） | 有，前段隊伍 Dice 在 0.80s 中段 |
| **TopCoW 2023/2024 → TopBrain 2025** | 2024: 125 對訓練 + 226 掃描多中心隱藏測試 | Willis 環多類別分割 + 偵測框 + **圖/拓樸標註** | Grand Challenge，開放 | **分割 Dice >90% 已接近飽和；但拓樸/變異分類只有 ~70% 平衡準確率——真正的空間在這裡** |
| **TubeTK (MRA)** | 100 對 T1-MRA，其中 42 位有中心線+半徑 | 中心線圖（.tre 格式） | KitwarePublic 免費 | 部分標註，不適合當主 benchmark |
| **SMILE-UHURA** | 12 訓練 + 2 驗證（7T ToF-MRA） | 體素級微血管 | ISBI 2023 挑戰 | 遠未飽和；N 小正是因為這種標註幾乎做不出來 |

### 氣道 CT（結構問題與血管同型）

| 名稱 | 規模 | 備註 |
|---|---|---|
| **ATM'22** | 500 例（300/50/150），含 COVID 噪聲 CT | MICCAI 2022 官方挑戰；用 tree-length-detected-rate 等拓樸指標決勝負，不是 Dice |
| **AIIB23** | 312 例，纖維化 + COVID 肺 | 刻意設計成不會飽和；效能在纖維化子集顯著掉落 |

### 神經元與角膜神經（跟「神經」最直接相關）

| 名稱 | 模態 | 規模 | 標註 | 備註 |
|---|---|---|---|---|
| **CORN-1/2/3** | 角膜共軛焦顯微鏡（CCM），活體 | CORN-1: **1698 張**，384×384，北京+帕多瓦兩地 | 逐像素神經纖維分割（CORN-2 增強對、CORN-3 迂曲度分級） | Zenodo，Scientific Data 發表。**這是「醫學影像判讀神經細微紋理」最對口的公開資料集**；標註不一致本身就是活躍研究題目 |
| **BigNeuron (Gold166)** | 光學顯微鏡影像堆疊 | 166 個神經元 | 完整 3D 樹狀重建（.swc 骨架圖，非像素遮罩） | 跨實驗室成像條件異質，很難 |
| **DIADEM** | 6 種物種/模態 | 2010 年競賽 | 手工 3D 重建 + 專用 DIADEM 指標 | 資料本身老舊，但 DIADEM 指標仍在用 |

### 非醫學代理任務

**Massachusetts Roads / DeepGlobe Road**：道路網路，拓樸結構與血管樹同型，常被拿來當第三個測試域（clDice 原論文就用了道路）。重點是這裡 **APLS**（路徑相似度）這種拓樸指標比像素 F-score 早成熟。

---

## 2. 競賽（2022–2026）

| 競賽 | 年 | 測什麼 | 備註 |
|---|---|---|---|
| ATM'22 | MICCAI 2022 | 肺氣道樹分割 | 前 10 名有 MedIA 2023 專文回顧；**拓樸指標決定排名** |
| TopCoW 2023 | MICCAI 2023 | Willis 環血管分割（CTA/MRA） | — |
| AIIB23 | MICCAI 2023 | 纖維化/COVID 肺的氣道分割 | 頭條結論是「在困難子集上效能崩落」，不是哪個架構贏 |
| SMILE-UHURA | ISBI 2023 | 7T MRA 微血管 | 16 個方法 + 2 個 baseline，含完全外部測試集 |
| TopCoW 2024 | MICCAI 2024 | 加入物件偵測 + **圖/拓樸分類**任務 | MRA 賽道有公開冠軍方案 repo |
| TopBrain 2025 | 2025 | 全腦血管解剖 | TopCoW 的後繼，尚無期刊 benchmark 論文 |
| SEG.A. 2023 | MICCAI 2023 | 主動脈血管樹分割/建模/網格化 | 掃描中順帶發現，未深入 |

**觀察**：grand-challenge.org 上**沒有一個統一的「拓樸賽道」**，競賽是按器官/模態散落的。想找 benchmark 要一個個器官找。

---

## 3. 評估指標

這一節是整個資料夾最實用的部分之一——**選錯指標會讓整篇論文白做**。

| 指標 | 量什麼 | 盲點 | 採用程度 |
|---|---|---|---|
| **Dice / IoU** | 像素重疊 | 對細結構幾乎無感：漏掉一整條細分支 Dice 掉不到 1%；報成 "accuracy" 更是被背景灌水 | 仍是普世預設，但**每一篇拓樸論文都明講它不夠用** |
| **clDice（當指標用）** | 預測骨架與真值遮罩的雙向重疊 | 仍偏袒粗血管：1–2 像素的中心線偏移對 >4 像素粗的血管無所謂，對 <4 像素的血管是災難 | **事實上的第二標準**，2021 後廣泛採用，同時當指標與 loss |
| **Betti number error** | \|β(預測) − β(真值)\|，只數拓樸特徵數量 | **完全不檢查位置**：兩張拓樸完全不同但特徵數相同的圖得同分 | 概念上已被 Betti matching 取代，實務上還在用 |
| **Betti matching error** | 用 induced matching 空間配對後再比 | 計算貴，3D 擴展不容易 | 2023 年後在拓樸專門論文中成長中，臨床論文尚未普及 |
| **Skeleton Recall** | 真值骨架被預測抓到的比例 | 只有召回，不管假陽性，必須配精確度側指標 | 2024 新，採用還早 |
| **Hausdorff / 95HD** | 最壞（或 95 百分位）邊界距離 | 對單一離群點極敏感；而且**只看距離不看拓樸**——兩張 95HD 很小的遮罩可以拓樸完全不同 | 醫學分割的普世第二指標 |
| **ccDice（連通元件 Dice）** | 把 Dice 提升到連通元件層級 | 很新（MICCAI 2024 workshop），在有很多真實小碎片（微血管床）的資料上行為未知 | 幾乎只有原論文在用 |
| **tree-length / branch-detected-rate 等連通性指標** | 直接數漏掉的分支、錯誤連接、斷點——最接近「臨床上這棵樹對不對」 | **每個競賽自己定義自己的公式**，跨論文完全不可比 | ATM'22 用它決勝負，但沒有標準化 |
| **ARI / VOI** | 實例分割的聚類一致性 | 是連結體學（connectomics）的原生指標，用在二元血管遮罩上不對味 | 神經元/EM 領域標準，血管領域幾乎不用 |
| **DIADEM metric** | 比對分岔點與末端的位置與拓樸 | 專為神經元重建設計 | 神經元追蹤領域仍在用 |
| **Metrics Reloaded**（metrics-reloaded.dkfz.de） | 不是指標，是 DKFZ 主導的「哪個任務該用哪個指標」的標準化倡議 | 只記錄問題，不解決可比性 | 正在成為引用參考點 |

**實務建議**：報 **Dice + clDice + Betti matching error（或至少 Betti number error）+ 95HD** 四個。只報 Dice 會被審稿人打；只報拓樸指標會被質疑犧牲了準確率。

---

## 4. 量測文化：這領域的數字可信嗎

直說：**預設不可信。**

- 有一篇專門的論文存在，就是因為 **100+ 篇 DRIVE/STARE/CHASE_DB1 的論文方法學不一致**——FOV（視野遮罩）處理方式不同、訓練/測試切分不同、指標實作不同。同一個資料集名稱下的數字經常不是在量同一件事。[verified via search]
- **STARE 根本沒有官方切分**，每篇自己做交叉驗證，所以 STARE 的跨論文比較基本上是虛構的。
- DRIVE 有官方切分，所以是唯一「跨論文 Dice 比較勉強有意義」的資料集——但仍然有論文混用 accuracy 與 Dice 來讓數字好看。
- 拓樸類指標（Betti 家族、ccDice、Skeleton Recall）**每一個都不到 3 歲**，沒有一個變成必填欄位。所以 2024 年用 Betti matching error 的論文，找不到更早的 baseline 可以比同一個數字。
- 競賽自訂的連通性指標（ATM'22 的 tree-length-detected-rate）是**per-challenge 定義**，不是可攜的具名指標。

**淨結論**：同一個官方競賽內部（同一批主辦、同一個隱藏測試集、同一份指標程式碼）的數字**是**可比的——TopCoW、ATM'22、ASOCA、ImageCAS（用官方切分）是可信的比較點。跨獨立論文、只共用資料集名稱的比較，可信度從「搖晃」到「虛構」都有。

**這對朋友的含義**：這件事本身就是機會。一篇「用統一 protocol 重跑 N 個方法、報完整拓樸指標」的論文在這個領域是有價值的貢獻（參考 TopoMortar 的做法，BMVC 2025 oral）。

---

## 5. 如果明天就要開始，選哪兩個資料集

**第一個：DRIVE。**
理由完全是務實的——零門檻取得、有官方切分、幾十年的先前工作可以拿來驗證 pipeline 有沒有裝錯，而且（跟大家的印象相反）Dice/F1 天花板在 0.82–0.84 而不是 0.98，**如果切入點是拓樸正確性而不是像素重疊，還有真實可量測的進步空間**。這是把整條 pipeline 跑通最省事的路。

**第二個：看要不要碰 3D。**
- **ImageCAS（冠狀動脈 CTA，3D）**：開放取得、有官方切分、~1000 個體積的規模足以支撐有統計意義的 ablation、Dice 還在 0.80–0.85。條件是要吃得下 3D CT 的算力。
- **TopCoW（腦血管 CTA/MRA）**：MICCAI 官方競賽基礎設施（隱藏測試集＝可信的外部驗證），但要注意純分割 Dice 已 >90% 接近飽和，真正的空間在**圖/拓樸分類任務**（平衡準確率約 70%）——這是更具體的賭注。

**如果題目重點是「神經」而不是血管**：直接上 **CORN-1**（1698 張角膜共軛焦顯微鏡神經纖維影像，Zenodo 開放）。這是掃描中找到跟「醫學影像判讀神經細微紋理」最對口的公開資料集，而且**標註不一致本身就是這個資料集上的活躍研究題目**——對做「標註噪聲下的拓樸學習」這種題目是加分。

**刻意避開的**：
- OCTA-500（要寄信要密碼，第一輪不值得這個摩擦）
- STARE（沒有官方切分，可比性從一開始就壞了）
- BigNeuron / DIADEM（有趣但社群小、沒有明確的官方比較點）


\newpage


# 4 · 未解問題與候選切入點

> 前半是**這個領域公認還沒解決的東西**（有文獻佐證）；後半是**我從這些缺口推出來的五個候選題目**，明確標為推論而非事實。
> 所有「還沒有人做」的主張，動手前務必自己重查一次——這領域月更。

---

## Part I：領域公認的未解問題

### P1. 沒有任何 loss 能同時做到「便宜 + 管徑正確 + 拓樸保證」

三個性質，目前每個方法都至少放棄一個：

| 方法 | 便宜 | 管徑感知 | 形式化拓樸保證 |
|---|---|---|---|
| clDice | ✅ | ❌（管徑盲） | ❌（幾何近似） |
| cbDice | ✅ | ⚠️（靠半徑估計，估計本身有誤差） | ❌ |
| Skeleton Recall | ✅✅ | ❌ | ❌ |
| Betti Matching 系 | ❌ | ❌（完全不管寬度） | ✅ |
| Topograph | ⚠️（比 PH 快 5 倍但仍需建圖） | ❌ | ✅ |
| arXiv:2601.11409（2026-01） | ❌（加了 PDE 求解器） | ✅ | ✅ |

最後一列是唯一一次把寬度折進拓樸約束的嘗試，但它是未審查的預印本而且引入了 PDE 求解複雜度。**「便宜 + 管徑感知 + 有保證」目前沒有被接受的解。**

**為什麼重要**：做微血管或神經纖維的人，管徑本身就是臨床指標（例如角膜神經纖維密度與長度是糖尿病神經病變的生物標記）。拓樸對但粗細錯，臨床上仍然是錯的。

### P2. 所有拓樸 loss 都假設真值的骨架是對的

1 像素寬的結構，真值遮罩在不同標註者之間本來就不一致（部分容積效應：一個像素同時含血管與背景）。目前的 loss 把真值骨架當成金標準去優化，**等於是在成像解析度極限附近對噪聲做優化**。

掃描中唯一的嘗試是 Smooth clDice 的「不確定區」，而那是一份技術報告，不是同儕審查的解法。要真正解決需要：機率式／不確定性感知的骨架定義，或是有多標註者共識的資料集——後者在細血管/神經領域基本上不存在（CORN 是例外，而它的標註不一致正是活躍題目）。

### P3. 沒有任何方法診斷「為什麼」斷掉，只罰「斷掉了」

從 TopoLoss 到 TopoSculpt，每一個 loss 都是對**症狀**（有個縫、少個環、Betti 數對不上）的懲罰。沒有一個去區分：

- 這個斷裂是因為該處**局部對比/訊噪比真的太低，資訊不存在**（模型不可能做對）
- 還是因為**模型的架構或訓練容量不足**（模型該做對但沒做到）

這兩種失敗的修法完全不同（前者要改成像或加先驗，後者要改模型），但現有方法一視同仁地加大懲罰。**這正是我自己 VLA 方向在做的那種「失敗歸因」框架，在這個領域是空的**（見 00-overview）。

### P4. 3D + 多類別 + 保證 + 速度，四者不可兼得

- Skeleton Recall Loss（2024）宣稱是第一個多類別可用的細結構 loss、省 90% 成本，但只有 1.5 歲，而且尚未被 nnU-Net 生態圈之外的獨立團隊交叉驗證。
- Betti Matching 家族（保證最強）在自己 2024 年的後續論文裡，被描述成「需要自寫 C++ 才能在 3D 尺度上**可用**」——是可用，不是快。

一個同時做到（a）全 3D 體積尺度、（b）多類別、（c）有拓樸保證、（d）訓練時間與純 Dice 相當的 loss，掃描中不存在。

### P5. 基礎模型在細長結構上的架構性失敗，還沒有被真正解決

WACV 2026 的診斷論文證明了 SAM 家族的失敗是架構性的、微調救不回來 [verified: arXiv:2412.04243]。目前兩條繞路（vesselFM 領域專用模型、VesselSAM/UCS 的 adapter 微調）都有效果，但：
- vesselFM 的評估範圍還窄（4 種模態）
- adapter 路線是在補救一個根本不適合的表示（patch 比血管粗），不是解決它

**「該用什麼表示來取代 patch tokenization 以保留次-patch 寬度的結構」是開放的。** FeatUp/LoftUp（特徵上採樣）與「patch 越小越好」的 scaling law（arXiv:2502.03738）指向同一個方向，但沒有人在醫學細長結構上把這條線做完。

### P6. 量測本身是壞的

見 3_datasets_metrics · 4. 量測文化：這領域的數字可信嗎。沒有統一 protocol、拓樸指標都不到 3 歲、競賽自訂指標不可攜。這既是障礙也是機會。

---

## Part II：五個候選切入點（我的推論，非文獻事實）

按「可實現性」排序，不是按「有趣程度」。每個都標明**為什麼可能是真空**與**最大的風險**。

### C1. 把視覺領域的細節保全工具，系統性搬到醫學細長結構上

**做什麼**：拿一個標準 baseline（nnU-Net 或 FR-UNet + clDice），逐一掛上 BlurPool（抗混疊下採樣）、PointRend/CascadePSP（邊界精修）、FeatUp/LoftUp（特徵上採樣），在 DRIVE + CORN + 一個 3D 集上量**拓樸指標**（不只 Dice），做完整 ablation。

**為什麼可能是真空**：掃描找不到明確把這些用在醫學細長結構的論文，信心「中～中高」（FeatUp/LoftUp 純粹是太新；BlurPool 可能已經被靜默包含在某些函式庫裡而沒人明講）。

**為什麼合理**：這些工具針對的正是 2_methods 裡講的核心成因——下採樣消滅細結構。而且改動極小、可完全消融。

**最大風險**：這是「工程整合 + 實證」型論文，不是新方法。在 CVPR 這種地方可能被嫌貢獻不足；在 MICCAI 或期刊比較有機會。要拉高貢獻，得加上一個**機制解釋**（例如量測「多少細血管在第 k 層下採樣後消失」，把混疊理論與拓樸誤差直接連起來）。

### C2. 標註噪聲下的拓樸學習（打 P2）

**做什麼**：正面處理「1 像素寬結構的真值本身就不可靠」。可能的形式：把骨架定義成機率分布而非二元、用多標註者不一致度當樣本權重、或設計一個對標註偏移不變的拓樸 loss。

**為什麼可能是真空**：目前唯一的嘗試（Smooth clDice）是未審查的技術報告。

**為什麼合理**：CORN 資料集的標註不一致本身就是公開的活躍議題，資料現成。而且這個問題在「越細的結構越嚴重」，跟朋友的題目完全對齊。

**最大風險**：需要多標註者資料才能證明。要先確認 CORN 或哪個資料集真的提供多標註者版本，否則得自己標或用模擬擾動（說服力較弱）。

### C3. 失敗歸因框架：區分「資訊不存在」與「模型做不到」（打 P3）

**做什麼**：不是提出新 loss，而是提出**診斷方法**。針對每一個斷裂點，量測局部證據強度（對比、訊噪比、與鄰近結構的可分性），把斷裂分成「資訊不足型」與「模型能力型」，然後證明兩類需要不同的修法（例如：對前者加拓樸 loss 是無效甚至有害的，對後者才有效）。

**為什麼可能是真空**：掃描明確指出沒有任何 loss 論文做因果層級的診斷，全部停在症狀層級。

**為什麼合理**：這在方法論上跟我在 VLA 那邊做的失敗歸因是同一個骨架（見 06-analysis-design），可以互相借鑑實驗設計。而且它產生的是**可以被引用的觀察**，而不是又一個 +0.3 Dice 的方法。

**最大風險**：「資訊不足」的操作型定義很難不循環論證（怎麼證明資訊真的不存在，而不是你的模型看不到？）。需要一個乾淨的做法——例如用受控合成資料（TopoMortar 的路子）刻意調控局部對比，或用更高解析度的同源影像當金標準。**這個風險是實質的，要先想清楚再投入。**

### C4. 用圖表示直接解決連通性（跟隨 P1 的最強一線）

**做什麼**：跟隨 GraphMorph / Relationformer 的路線，輸出圖而非像素遮罩，讓連通性由輸出格式保證。可以做的增量：把**管徑**當成邊的屬性一起預測（直接打 P1 的管徑盲）。

**為什麼合理**：B-architectures 的判斷是圖/拓樸感知方法在「真正重要的指標」上有最一致的增益，信心中高。而且圖表示天然可以帶半徑屬性——ASOCA 與 TubeTK 就有中心線+半徑的真值。

**最大風險**：競爭最激烈的一條線（GraphMorph 是 NeurIPS 2024）。而且需要圖形式的真值，資料集選擇受限。

### C5. 統一 protocol 的再現性研究（打 P6）

**做什麼**：用單一 protocol 重跑 N 個代表性方法，報完整指標矩陣（Dice + clDice + Betti matching + 95HD + 連通性），並用受控合成資料（TopoMortar 式）隔離「拓樸真的變好」與「靠其他混淆因素變好」。

**為什麼合理**：TopoMortar（BMVC 2025 oral）證明這類論文拿得到好 venue，而且它的發現很有價值——**它發現在受控條件下，資料增強與自蒸餾能讓標準 loss 追上 clDice**，也就是很多拓樸 loss 的增益可能被高估。

**最大風險**：工作量大且純實驗，需要穩定的算力與很嚴謹的執行；也容易得罪人。

---

## 我的建議（如果只能挑一個）

**C1 當第一個專案，C3 當長期押注。**

C1 有明確的可交付成果、風險低、能在幾個月內跑完，而且過程中會把整條 pipeline 與所有指標建起來——這些基礎設施 C3 也要用。C3 的智識價值高很多（提出診斷框架而非又一個 loss），但它的核心風險（怎麼定義「資訊不存在」）需要先用 C1 累積的實作經驗來想清楚。

C4 若有 3D 算力與圖真值資料就值得做，但要有心理準備跟 NeurIPS 級的團隊競爭。


\newpage


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
在超大規模資料上預訓練、可以泛化到多種下游任務的大模型（SAM、DINOv2、MedSAM）。**在細長結構上有已證實的架構性失敗**，見 2_methods · B5. 基礎模型：一個有證據的死路，以及兩個繞路。

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
STARE 沒有官方切分，所以跨論文比較無效——見 3_datasets_metrics · 4. 量測文化：這領域的數字可信嗎。

**ablation study（消融實驗）**
逐一拿掉自己方法的某個元件，看效能掉多少，用來證明「增益真的來自我說的那個東西」。

**tiling / patching（切塊推論）**
影像太大放不進顯存時，切成小塊分別推論再拼回去。**切塊邊界本身就是斷裂的來源**——SEMIR 之類的方法就是想避開它。

**self-supervised / weakly-supervised（自監督／弱監督）**
不用完整像素標註來訓練。YoloCurvSeg 是極端例子：只要一條有噪聲的骨架標註。

**FOV mask（視野遮罩）**
眼底照片是圓形視野，四角是黑的。要不要把黑角算進指標，會顯著改變數字——這是 DRIVE/STARE 論文不可比的原因之一。


\newpage

