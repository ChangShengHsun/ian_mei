# paper/ — ISBI 投稿工作區

建立於 2026-09-19。範本是**官方的**：從 ISBI 2026 的
[Initial Author Instructions](https://biomedicalimaging.org/2026/initial-author-instructions/)
頁面所連的 `ISBI_template-master.zip` 下載（該頁指向 2025 年的壓縮包，
ISBI 這組樣式檔多年沒變）。

## 這台機器**編不了** LaTeX

`pdflatex`、`xelatex`、`latexmk`、`tectonic`、`bibtex` **全部沒有**
（2026-09-19 確認）。所以：

- **建議用 Overleaf**：把整個 `paper/` 資料夾上傳（不含 `template/` 也可以），
  主檔選 `main.tex`，編譯器用 pdfLaTeX。
- 想在本機編譯的話需要 `apt install texlive-latex-recommended
  texlive-fonts-recommended texlive-latex-extra`——那是系統層級安裝，
  依 `~/.claude/CLAUDE.md` 要你先同意，我沒有裝。
- 在那之前，`python3 paper/check.py` 會擋掉編譯器會抓到的那類錯誤
  （`\cite` 沒有對應條目、用了沒定義的數字巨集、環境沒收尾、圖檔不存在）。
  **它不能證明編得過，只能證明沒有這幾類錯。**

## 檔案

| 檔案 | 是什麼 | 可以改嗎 |
|---|---|---|
| `main.tex` | 外殼：開關、數字巨集、標題作者，然後 `\input` 把各節接進來 | 就是要改 |
| `sections/*.tex` | **正文在這裡**，一節一個檔，檔名的數字就是出場順序 | 就是要改 |
| `refs.bib` | 參考文獻，已填入 13 筆 | 要改 |
| `numbers.tex` | **生成的**：論文用到的每個量測值，一個巨集一個數字 | **不要手改** |
| `figures/fig1_case_set.{pdf,png}` | 構造案例集 | **生成的** |
| `figures/fig2_conventions.{pdf,png}` | 十二個約定格，DRIVE | **生成的** |
| `figures/fig3_coverage.{pdf,png}` | 分歧隨 coverage 縮放 | **生成的** |
| `tables/tab{1,2,3}_*.tex` | 三張表，`sections/` 裡用 `\input` 引入 | **生成的** |
| `figures/blind_spot.png` | 舊版 Fig 1，已被 `fig1_case_set` 取代 | 可刪（我不刪你的檔） |
| `check.py` | 結構檢查，見上（會自動展開 `\input`，所以連 `sections/` 一起檢查） | 要改 |
| `.gitignore` | LaTeX 編譯產生的暫存檔不要進版控 | 很少改 |
| `spconf.sty` | ISBI 官方樣式 | **不要改** |
| `IEEEbib.bst` | ISBI 官方書目樣式 | **不要改** |
| `template/` | 官方壓縮包原封不動，含 `format.pdf`（官方格式說明）與兩份原始 `.tex` | 參考用，不要改 |

## 圖表全部由腳本生成

```
.venv/bin/python exp/paper_figures.py
```

重畫三張圖、三張表、以及 `numbers.tex` 裡的 33 個數字巨集。**論文裡沒有任何
一個手打的量測值**（唯一例外是參考實作那五個，`main.tex` 裡標了
TRANSCRIBED 並說明原因：那個套件沒裝在這台機器上）。

它的 selftest 會從 57600 列原始 CSV 重算 `erl_spec.txt` 印出來的整張表，
**對不上就拒絕寫出任何東西**——所以圖不可能跟它要說明的判決檔脫節。

配色用的是繪圖規範裡那組已驗證的預設色票，照固定槽位使用；每條序列同時帶
marker 與線型，所以**印成黑白也分得出來**（ISBI 審稿人是看紙本的）。
色票驗證器是 node 腳本而這台機器沒有 node，所以這次**沒有重跑驗證**，
是照原樣使用。

## `main.tex` 與 `sections/` 是怎麼安排的

1. **開關**：`\drafttrue` / `\draftfalse` 一行切換。draft 模式下
   `\TODO{...}` 會在 PDF 上印成紅字，投稿前改成 `\draftfalse` 就全部消失。
2. **數字區**：論文裡每一個量出來的數字都定義成一個 `\newcommand`，
   後面的註解寫明它是從哪個結果檔量到的。
   **理由**：同一個數字在論文裡出現兩次，遲早會自相矛盾；改一處、全篇跟著動。
   審稿人問「這個 17.3 哪來的」，一行註解就能答。
   `check.py` 會告訴你還有幾個數字只躺在註解裡、還沒寫進正文。
3. **章節**：一節一個檔，放在 `sections/`，由 `main.tex` 底部的 `\input` 依序接進來。
   每個檔開頭的註解寫了**這一節該講什麼、佔多少篇幅**，正文留 `\TODO{}`，
   不是留空白——空白會忘記，紅字不會。
   **要換敘事順序，改 `\input` 的排列就好**，不必搬動正文。

## ISBI 的規則（2026-09-19 從官網讀到）

- **四頁**，所有技術內容（含圖表）必須在前四頁內。
- 第五頁要加錢（US$200），而且**只能**放三種東西：
  Compliance with Ethical Standards、Acknowledgments、References。
  放了任何技術內容 → 直接退稿。
- **Compliance with Ethical Standards 是必填的**，不管你需不需要倫理審查。
  `main.tex` 已經放了一節。
- ISBI 2026 對全文採**單盲**審查，所以**作者名字要寫**。
  `main.tex` 的 `\name` 已經填了你的名字，共同作者待補。

## 時程：這份要投的其實是 ISBI 2027

ISBI **2026** 的投稿截止是 **2025-11-14，已經過了**（會議在倫敦）。
所以目標是 ISBI 2027，而它的 CFP 還沒出來——
`stage-report/ERL_thesis.md` §5 寫的 2026-10-26 是**推估**，不是官方日期。

投稿前務必重新確認三件事：**(1)** 範本有沒有更新、
**(2)** 還是不是單盲（改雙盲的話要把 `\name`、`\address`、致謝全部匿名）、
**(3)** 真正的截止日。

## 論文主軸還沒定案

`main.tex` 檔頭寫了一個**建議**的敘事順序（構造案例集 → 參考實作的缺陷 →
規格），另一個選項是 `stage-report/ERL_thesis.md` §5 建議的「以參考實作開場」。
**這是取捨不是對錯，需要你和教授拍板**，決定之後再寫正文——
章節順序改起來很便宜，寫完的正文改起來不便宜。
