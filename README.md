# 構造力学解析アプリ

手書き構造図から自動で構造解析を行い、変形図と応力図を出力するStreamlitアプリケーションです。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 機能

- **画像認識**: YOLOv8-OBBモデルで手書き構造図から要素を自動検出
- **清書処理**: 15度刻みで角度補正、支点の座標整列、自動接続
- **構造解析**: 剛性マトリクス法（FEM）による構造解析
- **結果可視化**: 変形図、軸力図、せん断力図、曲げモーメント図

## 📋 対応要素

### 支点（4種類）
- ピン支点 (pin)
- ピンローラー支点 (roller)
- 固定支点 (fixed)
- ヒンジ (hinge)

### 荷重（4種類）
- 集中荷重 (load)
- 等分布荷重 (UDL)
- モーメント荷重（左回り） (momentL)
- モーメント荷重（右回り） (momentR)

### 部材
- 梁 (beam)

## 🚀 ローカルでの実行

### 必要な環境
- Python 3.8以上
- pip

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-username/structural-analysis-app.git
cd structural-analysis-app

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 実行

```bash
streamlit run structural_analysis_app.py
```

ブラウザが自動で開き、アプリが表示されます。

## 📁 ファイル構成

```
.
├── structural_analysis_app.py  # メインアプリケーション
├── fem_lib.py                  # FEM解析ライブラリ
├── draw_lib.py                 # 描画処理ライブラリ
├── requirements.txt            # 必要なライブラリ
├── packages.txt                # システムパッケージ（Streamlit Cloud用）
├── .streamlit/
│   └── config.toml            # Streamlit設定
├── models/
│   └── best.pt                # YOLOv8-OBBモデル（要配置）
└── templates/                  # 要素テンプレート画像
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

## ☁️ Streamlit Cloudへのデプロイ

詳細なデプロイ手順は [DEPLOY.md](DEPLOY.md) を参照してください。

### クイックスタート

1. **必要なファイルを準備**
   ```bash
   mkdir -p models templates
   # モデルとテンプレート画像を配置
   ```

2. **GitHubにプッシュ**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

3. **Streamlit Cloudでデプロイ**
   - [Streamlit Cloud](https://streamlit.io/cloud)にアクセス
   - リポジトリを選択
   - `structural_analysis_app.py`を指定
   - Deploy!

### トラブルシューティング

**`ModuleNotFoundError: No module named 'cv2'`が出た場合**:
- requirements.txtに`opencv-python-headless`が含まれているか確認
- packages.txtに必要なシステムパッケージが含まれているか確認
- 詳細は [DEPLOY.md](DEPLOY.md) を参照

## 🎨 使い方

1. **画像をアップロード**: 手書きの構造図画像を選択
2. **パラメータ調整**: サイドバーで解析パラメータを調整
   - 検出信頼度
   - 高さ揃え閾値
   - 接続閾値
   - 材料特性
   - 荷重の大きさ
3. **解析実行**: 「解析実行」ボタンをクリック
4. **結果確認**: 変形図と応力図を確認

## 🔧 技術仕様

### 画像認識
- モデル: YOLOv8-OBB (Oriented Bounding Box)
- 入力画像サイズ: 640x640
- 角度補正: 15度刻みで自動補正

### 構造解析
- 手法: 剛性マトリクス法（有限要素法）
- 要素: 平面骨組要素（6自由度/節点）
- 座標系: 2次元直交座標系
- 符号規則: x右向き正、y上向き正、モーメント反時計回り正

### 清書処理
- 支点のx/y座標を自動整列
- 梁端点と支点節点の自動接続
- 荷重の作用点: 矢じりの先端
- 梁の途中に荷重がある場合は梁を分割

### 変形図
- 最大変位を部材長の1/4に自動スケール調整
- 見やすい図を出力

## 📝 注意事項

- モデルファイル（best.pt）は約100MBあります
- Git LFSを使用しない場合、GitHubの容量制限に注意
- 手書き図面は明瞭に描いてください
- 複雑な構造の場合、検出精度が低下する可能性があります

## 🐛 トラブルシューティング

### モデルが見つからない
```
エラー: モデルパスが存在しません
```
→ `models/best.pt` にモデルファイルを配置してください

### 解析エラー
```
エラー: FEM解析エラー
```
→ 構造が不安定（支点不足）または入力データに問題がある可能性があります

### Streamlit Cloudでのデプロイエラー
- requirements.txtの内容を確認
- packages.txtにシステムパッケージを追加
- モデルファイルのパスを確認

## 📄 ライセンス

このプロジェクトは教育目的で作成されています。

## 👤 開発者

森本遥香 (DA22340)

## 🙏 謝辞

- YOLOv8: Ultralytics
- Streamlit: Streamlit Inc.
