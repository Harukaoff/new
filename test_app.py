"""
構造力学解析アプリのテストスクリプト
各モジュールが正しくインポートできるか確認
"""

import sys

def test_imports():
    """必要なモジュールのインポートテスト"""
    print("=" * 50)
    print("モジュールインポートテスト")
    print("=" * 50)
    
    modules = [
        ("streamlit", "Streamlit"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("ultralytics", "Ultralytics YOLO"),
        ("PIL", "Pillow"),
    ]
    
    success_count = 0
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}: OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: FAILED - {e}")
    
    print(f"\n{success_count}/{len(modules)} モジュールが正常にインポートされました")
    return success_count == len(modules)

def test_local_modules():
    """ローカルモジュールのインポートテスト"""
    print("\n" + "=" * 50)
    print("ローカルモジュールテスト")
    print("=" * 50)
    
    modules = [
        ("fem_lib", "FEM解析ライブラリ"),
        ("draw_lib", "描画ライブラリ"),
    ]
    
    success_count = 0
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}: OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {display_name}: FAILED - {e}")
    
    print(f"\n{success_count}/{len(modules)} ローカルモジュールが正常にインポートされました")
    return success_count == len(modules)

def test_paths():
    """パスの存在確認"""
    import os
    
    print("\n" + "=" * 50)
    print("パス存在確認")
    print("=" * 50)
    
    paths = [
        (r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt", "YOLOモデル"),
        (r"C:\Users\morim\Downloads\graduation\templates", "テンプレートディレクトリ"),
    ]
    
    success_count = 0
    for path, description in paths:
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
            success_count += 1
        else:
            print(f"❌ {description}: NOT FOUND - {path}")
    
    print(f"\n{success_count}/{len(paths)} パスが存在します")
    return success_count == len(paths)

def test_template_files():
    """テンプレートファイルの存在確認"""
    import os
    
    print("\n" + "=" * 50)
    print("テンプレートファイル確認")
    print("=" * 50)
    
    template_dir = r"C:\Users\morim\Downloads\graduation\templates"
    template_files = [
        "pin.png",
        "roller.png",
        "fixed.png",
        "hinge.png",
        "beam.png",
        "load.png",
        "UDL.png",
        "momentL.png",
        "momentR.png",
    ]
    
    success_count = 0
    for filename in template_files:
        path = os.path.join(template_dir, filename)
        if os.path.exists(path):
            print(f"✅ {filename}")
            success_count += 1
        else:
            print(f"❌ {filename}: NOT FOUND")
    
    print(f"\n{success_count}/{len(template_files)} テンプレートファイルが存在します")
    return success_count == len(template_files)

def main():
    """メインテスト実行"""
    print("\n🔍 構造力学解析アプリ - 環境テスト\n")
    
    results = []
    results.append(("モジュールインポート", test_imports()))
    results.append(("ローカルモジュール", test_local_modules()))
    results.append(("パス確認", test_paths()))
    results.append(("テンプレートファイル", test_template_files()))
    
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 すべてのテストが成功しました！")
        print("アプリを起動するには以下のコマンドを実行してください:")
        print("  streamlit run structural_analysis_app.py")
    else:
        print("⚠️  一部のテストが失敗しました")
        print("失敗した項目を確認して修正してください")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
