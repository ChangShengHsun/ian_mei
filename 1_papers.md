# 1 · 論文清單

> 分兩層：先是**12 篇必讀**（不讀這些沒辦法跟這領域的人對話），再是**完整分組清單**。
> 每篇標註查證狀態，規則見 [[0_overview#標註規則（沿用本 vault 慣例）]]。
> 方法本身的機制解說在 [[2_methods]]，這裡只講「這篇在整個故事裡的位置」。

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
