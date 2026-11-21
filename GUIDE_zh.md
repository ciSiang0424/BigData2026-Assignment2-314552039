# 把這個資料夾上傳成公開 GitHub repo（給新手、Mac）

作業要求：必須是**公開（Public）**的 GitHub repository，在截止時間 **2026-11-21 23:55（台灣時間）** 之前完成所有 commit，README 要說明怎麼跑，並且註明資料來源。

建議 repo 名稱：`BigData2026-Assignment2-314552039`

你現在手上的資料夾是 `github_repo/`（或解壓縮後的 `asg2_github_repo/github_repo/`）。下面兩種做法選一種即可。做完後把 repo 網址傳回來，投影片才能填連結。

---

## 開始之前

1. 準備一個 GitHub 帳號，並先在瀏覽器登入 https://github.com 。
2. **不要**把 `data/`、`.parquet`、`.pt`、`.venv` 上傳進去。這個資料夾已經排除這些檔案；請不要自行再加進去。
3. 建立 repo 時**不要**勾選 “Add a README file”，否則之後上傳會變成兩個 README，還要處理衝突。

---

## 方法 A：網頁拖曳上傳（不用打指令）

### A1. 在網頁建立空的公開 repo

1. 打開 https://github.com/new 。
2. Repository name 填：`BigData2026-Assignment2-314552039`（或你喜歡的名稱）。
3. 選 **Public**（一定要公開，選 Private 不符合作業規定）。
4. **不要**勾選 Add a README、Add .gitignore、Choose a license。
5. 按 **Create repository**。

### A2. 把資料夾內容拖上去

1. 在剛建好的空白 repo 頁面，點 **uploading an existing file**（或 Add file → Upload files）。
2. 打開你電腦上的 `github_repo` 資料夾，把**裡面的全部檔案和資料夾**拖進瀏覽器（要看到 `README.md`、`asg2_01_train_gnn.ipynb`、`src`、`results` 等，不是只拖一個空的外層資料夾）。
   - 如果系統一次上傳有大小或檔案數限制，可以分批傳，但最後都要在同一個 repo 裡。
3. 頁面下方 Commit message 可寫：`Add Assignment 2 source, results, and README`。
4. 確認 Commit 的分支是 `main`（或 `master`，之後可再改名）。
5. 按 **Commit changes**。

請在 **23:55 之前** 按下去。上傳中請不要關頁面。

---

## 方法 B：用 Mac 終端機（git）

先安裝 Git（若 `git --version` 已有輸出可跳過）：在終端機執行 `xcode-select --install`，或到 https://git-scm.com 安裝。

### B1. 一樣先在網頁建立空的公開 repo

步驟與 A1 相同。建好後，GitHub 會顯示一個網址，例如：

```
https://github.com/你的帳號/BigData2026-Assignment2-314552039.git
```

把它複製下來。

### B2. 在資料夾裡提交並推上去

打開「終端機」，把路徑改成你的資料夾（若你放在下載項目裡，路徑可能如下；請改成實際位置）：

```bash
cd /Users/你的帳號/Downloads/github_repo
```

確認這裡看得到 README：

```bash
ls
# 應看得到 README.md、asg2_01_train_gnn.ipynb、src、results、GUIDE_zh.md
```

然後執行：

```bash
git init
git add .
git status
# 看一下清單：應該有 README.md 和程式，不該有 .parquet、.pt、.venv、data/
git commit -m "Add Assignment 2 source, results, and README"
git branch -M main
git remote add origin https://github.com/你的帳號/BigData2026-Assignment2-314552039.git
git push -u origin main
```

### B3. 登入方式（HTTPS）

`git push` 時 GitHub 通常會要你登入，常見有兩種：

- **瀏覽器登入**：若出現 “Sign in with your browser”，照指示在瀏覽器授權即可。
- **Personal Access Token**：密碼欄請不要填 GitHub 登入密碼。到 GitHub → Settings → Developer settings → Personal access tokens，新增一個有 `repo` 權限的 token，在密碼欄貼上 token。

如果 `git` 提示 repo 已存在遠端檔案，代表建立時勾了 README，請不要強制覆蓋；比較安全的作法是刪掉那個 GitHub repo、用 A1 重建一個空白的再推一次。

請在 **23:55（台灣時間）之前** 讓 `git push` 成功。

---

## 上傳後請自己檢查（全部要通過）

打開你的 repo 網址，確認：

1. 頁面左上角或 About 附近標示 **Public**，不是 Private。
2. 首頁會自動顯示 `README.md`，標題與說明正常排版（不是一堆原始符號亂掉）。
3. README **靠近開頭** 看得到這一行（作業 1%）：

   `Data Source: GH Archive https://www.gharchive.org/`

4. 點進一個 commit，時間必須在 **2026-11-21 23:55 台灣時間** 之前。
5. 看得到 `asg2_01_train_gnn.ipynb`、`src/fetch_and_build.py`、`src/asg2_lib.py`、`results/output_314552039.csv`。
6. 沒有上傳 `.parquet` 或 `.pt` 檔。

最後把 repo 網址複製下來（例如 `https://github.com/你的帳號/BigData2026-Assignment2-314552039`），傳回來填進投影片第 1 頁與第 16 頁。

---

## 常見問題

- **不小心建成 Private：** 進 repo → Settings → 最下方 Danger Zone → Change repository visibility → Public。
- **不小心勾了 Add a README：** 不要跟遠端互推打架。刪掉 GitHub 上那個 repo，重建空白的再開傳。
- **拖曳時只傳了一個空資料夾：** 請進到 `github_repo` **裡面** 再把內容全選拖上去。
- **檔案太大傳不上去：** 這個資料夾壓縮後應遠小於 25 MB，不該碰到 GitHub 單一檔案 100 MB 限制。若有，請先確認沒有誤加 parquet 或模型權重。
