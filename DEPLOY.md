# Streamlit Cloudへのデプロイガイド

## エラー対処

### `ModuleNotFoundError: No module named 'cv2'`

このエラーが出た場合、以下を確認してください：

1. **requirements.txt**に`opencv-python-headless`が含まれているか
2. **packages.txt**にシステムパッケージが含まれているか

## 必要なファイル

### 1. requirements.txt
```
streamlit>=1.28.0
opencv-python-headless>=4.8.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
matplotlib>=3.7.0
ultralytics>=8.0.0
Pillow>=10.0.0
```

### 2. packages.txt
```
libgl1-mesa-glx
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
ffmpeg
```

### 3. ファイル構成
```
.
├── structural_analysis_app.py  # メインアプリ
├── fem_lib.py                  # FEM解析ライブラリ
├── draw_lib.py                 # 描画ライブラリ
├── requirements.txt            # Pythonパッケージ
├── packages.txt                # システムパッケージ
├── .streamlit/
│   └── config.toml            # Streamlit設定
├── models/
│   └── best.pt                # YOLOモデル（要配置）
└── templates/                  # テンプレート画像
    ├── pin.png
    ├── roller.png
    ├── fixed.png
    ├── hinge.png
    ├── beam.png
    ├── load.png
    ├── UDL.png
    ├── momentL.png
    └── momentR.png
```

## デプロイ手順

### ステップ1: GitHubリポジトリの準備

1. **モデルファイルの配置**
   ```bash
   mkdir -p models
   # best.pt を models/ にコピー
   ```

2. **テンプレート画像の配置**
   ```bash
   mkdir -p templates
   # 各テンプレート画像を templates/ にコピー
   ```

3. **Git LFSの設定（大きなファイル用）**
   ```bash
   git lfs install
   git lfs track "models/*.pt"
   git add .gitattributes
   ```

4. **GitHubにプッシュ**
   ```bash
   git add .
   git commit -m "Add model and templates for deployment"
   git push origin main
   ```

### ステップ2: Streamlit Cloudでのデプロイ

1. [Streamlit Cloud](https://streamlit.io/cloud)にアクセス
2. GitHubアカウントでログイン
3. 「New app」をクリック
4. リポジトリ、ブランチ、ファイルを選択：
   - Repository: `your-username/structural-analysis-app`
   - Branch: `main`
   - Main file path: `structural_analysis_app.py`
5. 「Advanced settings」をクリック（オプション）
   - Python version: `3.11`
6. 「Deploy!」をクリック

### ステップ3: デプロイ後の確認

デプロイが完了すると、以下のようなURLでアクセスできます：
```
https://your-app-name.streamlit.app
```

## トラブルシューティング

### エラー: `ModuleNotFoundError: No module named 'cv2'`

**原因**: OpenCVがインストールされていない

**解決方法**:
1. requirements.txtに`opencv-python-headless`があることを確認
2. packages.txtに必要なシステムパッケージがあることを確認
3. Streamlit Cloudでアプリを再起動

### エラー: `FileNotFoundError: models/best.pt`

**原因**: モデルファイルがリポジトリに含まれていない

**解決方法**:
1. モデルファイルをGitHubにプッシュ
2. Git LFSを使用（ファイルサイズが100MB以上の場合）
3. または、環境変数でパスを指定

### エラー: メモリ不足

**原因**: Streamlit Cloudの無料プランはメモリ制限がある

**解決方法**:
1. モデルを軽量化（YOLOv8n など）
2. 画像サイズを小さくする
3. 有料プランにアップグレード

### エラー: Git LFSの容量制限

**原因**: Git LFSの無料枠（1GB）を超えた

**解決方法**:
1. モデルファイルを外部ストレージ（Google Drive、Dropbox等）に配置
2. アプリ起動時にダウンロードする処理を追加
3. Git LFSの有料プランを使用

## 外部ストレージからモデルをダウンロードする方法

モデルファイルが大きすぎる場合、外部ストレージから動的にダウンロードできます：

```python
import os
import requests
from pathlib import Path

@st.cache_resource
def download_model():
    """モデルファイルをダウンロード"""
    model_path = Path("models/best.pt")
    
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Google Driveの共有リンクから取得
        url = "YOUR_GOOGLE_DRIVE_DIRECT_LINK"
        
        with st.spinner("モデルをダウンロード中..."):
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
    
    return str(model_path)

# 使用例
MODEL_PATH = download_model()
```

## パフォーマンス最適化

### 1. キャッシュの活用
```python
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
```

### 2. 画像サイズの制限
```python
# アップロード画像のサイズを制限
max_size = 1024
if img.shape[0] > max_size or img.shape[1] > max_size:
    scale = max_size / max(img.shape[:2])
    img = cv2.resize(img, None, fx=scale, fy=scale)
```

### 3. セッション状態の活用
```python
# 解析結果をセッションに保存
if 'results' not in st.session_state:
    st.session_state.results = None
```

## 環境変数の設定

Streamlit Cloudの「Settings」→「Secrets」で環境変数を設定できます：

```toml
MODEL_PATH = "/mount/src/app/models/best.pt"
TEMPLATE_DIR = "/mount/src/app/templates"
```

## 参考リンク

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [OpenCV Installation](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

## サポート

問題が解決しない場合は、以下を確認してください：

1. Streamlit Cloudのログを確認
2. requirements.txtとpackages.txtの内容を確認
3. GitHubリポジトリのファイル構成を確認
4. [Streamlit Community Forum](https://discuss.streamlit.io/)で質問

---

**開発者**: 森本遥香 (DA22340)
