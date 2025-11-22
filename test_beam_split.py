"""
梁の分割処理テスト
荷重が梁の途中に作用している場合の分割処理を確認
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def visualize_beam_split():
    """梁の分割処理を視覚化"""
    
    # テストデータ
    # 元の梁
    beam = {
        "node1_idx": 0,
        "node2_idx": 1,
        "node1_coord": np.array([100.0, 200.0]),
        "node2_coord": np.array([500.0, 200.0]),
    }
    
    # 荷重の投影点（梁上のt=0.3, 0.5, 0.7の位置）
    loads = [
        {"t": 0.3, "type": "load"},
        {"t": 0.5, "type": "load"},
        {"t": 0.7, "type": "load"},
    ]
    
    # 図の作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # === 分割前 ===
    ax1.set_xlim(50, 550)
    ax1.set_ylim(150, 250)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('分割前: 1本の梁', fontsize=14, fontweight='bold')
    
    # 元の梁
    ax1.plot([beam["node1_coord"][0], beam["node2_coord"][0]], 
            [beam["node1_coord"][1], beam["node2_coord"][1]], 
            'gray', linewidth=8, alpha=0.5, label='梁')
    
    # 端点
    ax1.plot(beam["node1_coord"][0], beam["node1_coord"][1], 'ro', markersize=12, label='端点')
    ax1.plot(beam["node2_coord"][0], beam["node2_coord"][1], 'ro', markersize=12)
    ax1.text(beam["node1_coord"][0], beam["node1_coord"][1] - 15, 'N0', fontsize=12, ha='center')
    ax1.text(beam["node2_coord"][0], beam["node2_coord"][1] - 15, 'N1', fontsize=12, ha='center')
    
    # 荷重の投影点
    for i, load in enumerate(loads):
        t = load["t"]
        proj = beam["node1_coord"] + t * (beam["node2_coord"] - beam["node1_coord"])
        ax1.plot(proj[0], proj[1], 'b^', markersize=10, label='荷重' if i == 0 else '')
        ax1.text(proj[0], proj[1] + 15, f't={t}', fontsize=10, ha='center')
    
    ax1.legend(loc='upper right')
    
    # === 分割後 ===
    ax2.set_xlim(50, 550)
    ax2.set_ylim(150, 250)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('分割後: 4本の梁（荷重位置で分割）', fontsize=14, fontweight='bold')
    
    # 分割点を追加
    all_nodes = [beam["node1_coord"], beam["node2_coord"]]
    split_nodes = []
    for load in loads:
        t = load["t"]
        proj = beam["node1_coord"] + t * (beam["node2_coord"] - beam["node1_coord"])
        split_nodes.append(proj)
        all_nodes.append(proj)
    
    # 分割された梁を描画
    colors = ['red', 'blue', 'green', 'orange']
    sorted_nodes = [beam["node1_coord"]] + sorted(split_nodes, key=lambda x: x[0]) + [beam["node2_coord"]]
    
    for i in range(len(sorted_nodes) - 1):
        ax2.plot([sorted_nodes[i][0], sorted_nodes[i+1][0]], 
                [sorted_nodes[i][1], sorted_nodes[i+1][1]], 
                color=colors[i % len(colors)], linewidth=8, alpha=0.7, 
                label=f'梁{i}')
    
    # すべての節点
    for i, node in enumerate(sorted_nodes):
        if i == 0 or i == len(sorted_nodes) - 1:
            ax2.plot(node[0], node[1], 'ro', markersize=12)
            ax2.text(node[0], node[1] - 15, f'N{i}', fontsize=12, ha='center', color='red')
        else:
            ax2.plot(node[0], node[1], 'go', markersize=12)
            ax2.text(node[0], node[1] - 15, f'N{i}', fontsize=12, ha='center', color='green')
    
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('beam_split_test.png', dpi=150, bbox_inches='tight')
    print("✅ 梁の分割処理の視覚化を保存しました: beam_split_test.png")
    plt.show()

def test_split_logic():
    """分割ロジックのテスト"""
    print("=" * 60)
    print("梁の分割ロジックテスト")
    print("=" * 60)
    
    # テストケース
    beam = {
        "node1_idx": 0,
        "node2_idx": 1,
        "node1_coord": np.array([100.0, 200.0]),
        "node2_coord": np.array([500.0, 200.0]),
    }
    
    # 荷重の位置（t値）
    load_positions = [0.3, 0.5, 0.7]
    
    print(f"\n元の梁: N{beam['node1_idx']} → N{beam['node2_idx']}")
    print(f"座標: ({beam['node1_coord'][0]:.1f}, {beam['node1_coord'][1]:.1f}) → "
          f"({beam['node2_coord'][0]:.1f}, {beam['node2_coord'][1]:.1f})")
    print(f"長さ: {np.linalg.norm(beam['node2_coord'] - beam['node1_coord']):.1f}px")
    
    print(f"\n荷重の位置:")
    split_nodes = []
    for i, t in enumerate(load_positions):
        proj = beam["node1_coord"] + t * (beam["node2_coord"] - beam["node1_coord"])
        split_nodes.append({"idx": 2 + i, "coord": proj, "t": t})
        print(f"  荷重{i}: t={t:.2f}, 座標=({proj[0]:.1f}, {proj[1]:.1f})")
    
    # 分割後の梁
    print(f"\n分割後の梁:")
    sorted_nodes = [{"idx": 0, "coord": beam["node1_coord"], "t": 0.0}] + \
                   sorted(split_nodes, key=lambda x: x["t"]) + \
                   [{"idx": 1, "coord": beam["node2_coord"], "t": 1.0}]
    
    for i in range(len(sorted_nodes) - 1):
        n1 = sorted_nodes[i]
        n2 = sorted_nodes[i + 1]
        length = np.linalg.norm(n2["coord"] - n1["coord"])
        print(f"  梁{i}: N{n1['idx']} → N{n2['idx']}, 長さ={length:.1f}px")
    
    # 検証
    print(f"\n検証:")
    total_length = sum([np.linalg.norm(sorted_nodes[i+1]["coord"] - sorted_nodes[i]["coord"]) 
                       for i in range(len(sorted_nodes) - 1)])
    original_length = np.linalg.norm(beam["node2_coord"] - beam["node1_coord"])
    print(f"  元の梁の長さ: {original_length:.1f}px")
    print(f"  分割後の合計長さ: {total_length:.1f}px")
    print(f"  差: {abs(total_length - original_length):.6f}px")
    
    if abs(total_length - original_length) < 0.01:
        print("  ✅ 長さが保持されています")
    else:
        print("  ❌ 長さが変わっています")

def test_edge_cases():
    """エッジケースのテスト"""
    print("\n" + "=" * 60)
    print("エッジケースのテスト")
    print("=" * 60)
    
    test_cases = [
        {"name": "端点近く (t=0.05)", "t": 0.05, "should_split": False},
        {"name": "端点近く (t=0.95)", "t": 0.95, "should_split": False},
        {"name": "梁の中央 (t=0.5)", "t": 0.5, "should_split": True},
        {"name": "分割境界 (t=0.1)", "t": 0.1, "should_split": False},
        {"name": "分割境界 (t=0.9)", "t": 0.9, "should_split": False},
        {"name": "分割範囲内 (t=0.11)", "t": 0.11, "should_split": True},
        {"name": "分割範囲内 (t=0.89)", "t": 0.89, "should_split": True},
    ]
    
    for case in test_cases:
        t = case["t"]
        should_split = case["should_split"]
        
        # 分割判定（0.1 < t < 0.9）
        will_split = 0.1 < t < 0.9
        
        status = "✅" if will_split == should_split else "❌"
        print(f"{status} {case['name']}: 分割={'する' if will_split else 'しない'} "
              f"(期待: {'する' if should_split else 'しない'})")

def main():
    """メインテスト実行"""
    print("\n🔍 梁の分割処理テスト\n")
    
    test_split_logic()
    test_edge_cases()
    visualize_beam_split()
    
    print("\n" + "=" * 60)
    print("✅ すべてのテストが完了しました")
    print("=" * 60)

if __name__ == "__main__":
    main()
