"""
接続処理の視覚的テスト
実際の接続処理をシミュレート
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

def visualize_connection_test():
    """接続処理を視覚化"""
    
    # テストデータ
    # 支点の節点
    support_nodes = [
        np.array([100.0, 300.0]),  # N0: ピン支点
        np.array([400.0, 300.0]),  # N1: ローラー支点
    ]
    
    # 梁の端点（検出された四角形から抽出）
    beam_endpoints = [
        {"pt1": np.array([105.0, 295.0]), "pt2": np.array([395.0, 298.0])},  # 梁1
    ]
    
    # 荷重の矢じり先端
    load_tips = [
        {"tip": np.array([250.0, 200.0]), "type": "load"},  # 集中荷重
    ]
    
    # 接続閾値
    threshold = 25.0
    
    # 図の作成
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.set_title('節点接続処理のシミュレーション', fontsize=16, fontweight='bold')
    
    # 1. 支点節点を描画
    all_nodes = []
    node_info = []
    
    for i, node in enumerate(support_nodes):
        all_nodes.append(node)
        node_info.append({"type": "support"})
        circle = Circle(node, 10, color='red', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.text(node[0] + 15, node[1] - 15, f'N{i}\n(支点)', fontsize=10, color='red')
        
        # 接続閾値の円
        threshold_circle = Circle(node, threshold, color='red', fill=False, 
                                 linewidth=1, linestyle='--', alpha=0.3)
        ax.add_patch(threshold_circle)
    
    # 2. 梁の端点を処理
    beam_connections = []
    for beam_idx, be in enumerate(beam_endpoints):
        pt1 = be["pt1"]
        pt2 = be["pt2"]
        
        # 端点1の処理
        min_dist1 = float('inf')
        snap_idx1 = -1
        for i, node in enumerate(all_nodes):
            dist = np.linalg.norm(pt1 - node)
            if dist < min_dist1:
                min_dist1 = dist
                snap_idx1 = i
        
        if min_dist1 < threshold and snap_idx1 >= 0:
            node1_idx = snap_idx1
            node1_coord = all_nodes[snap_idx1]
            # スナップを視覚化
            ax.plot([pt1[0], node1_coord[0]], [pt1[1], node1_coord[1]], 
                   'g--', linewidth=2, alpha=0.5)
            ax.text((pt1[0] + node1_coord[0])/2, (pt1[1] + node1_coord[1])/2 - 10,
                   f'スナップ\n{min_dist1:.1f}px', fontsize=8, color='green')
        else:
            node1_idx = len(all_nodes)
            node1_coord = pt1
            all_nodes.append(pt1)
            node_info.append({"type": "beam_endpoint"})
            circle = Circle(node1_coord, 8, color='blue', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.text(node1_coord[0] + 15, node1_coord[1] - 15, 
                   f'N{node1_idx}\n(新規)', fontsize=10, color='blue')
        
        # 端点2の処理
        min_dist2 = float('inf')
        snap_idx2 = -1
        for i, node in enumerate(all_nodes):
            dist = np.linalg.norm(pt2 - node)
            if dist < min_dist2:
                min_dist2 = dist
                snap_idx2 = i
        
        if min_dist2 < threshold and snap_idx2 >= 0:
            node2_idx = snap_idx2
            node2_coord = all_nodes[snap_idx2]
            # スナップを視覚化
            ax.plot([pt2[0], node2_coord[0]], [pt2[1], node2_coord[1]], 
                   'g--', linewidth=2, alpha=0.5)
            ax.text((pt2[0] + node2_coord[0])/2, (pt2[1] + node2_coord[1])/2 - 10,
                   f'スナップ\n{min_dist2:.1f}px', fontsize=8, color='green')
        else:
            node2_idx = len(all_nodes)
            node2_coord = pt2
            all_nodes.append(pt2)
            node_info.append({"type": "beam_endpoint"})
            circle = Circle(node2_coord, 8, color='blue', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.text(node2_coord[0] + 15, node2_coord[1] - 15, 
                   f'N{node2_idx}\n(新規)', fontsize=10, color='blue')
        
        # 梁を描画
        ax.plot([node1_coord[0], node2_coord[0]], 
               [node1_coord[1], node2_coord[1]], 
               'gray', linewidth=6, alpha=0.5)
        
        beam_connections.append({
            "node1_idx": node1_idx,
            "node2_idx": node2_idx,
            "node1_coord": node1_coord,
            "node2_coord": node2_coord
        })
    
    # 3. 荷重の処理
    for load in load_tips:
        tip = load["tip"]
        
        # 最も近い梁を探す
        best_proj = None
        best_dist = 1e9
        
        for beam in beam_connections:
            a = np.array(beam["node1_coord"])
            b = np.array(beam["node2_coord"])
            ba = b - a
            denom = np.dot(ba, ba) + 1e-12
            t = np.dot(tip - a, ba) / denom
            t = max(0.0, min(1.0, t))
            proj = a + t * ba
            dist = np.linalg.norm(tip - proj)
            if dist < best_dist:
                best_dist = dist
                best_proj = proj
        
        if best_proj is not None:
            # 矢じり先端
            circle = Circle(tip, 6, color='red', fill=True)
            ax.add_patch(circle)
            ax.text(tip[0] + 15, tip[1], '矢じり先端', fontsize=9, color='red')
            
            # 投影点
            circle = Circle(best_proj, 6, color='orange', fill=True)
            ax.add_patch(circle)
            ax.text(best_proj[0] + 15, best_proj[1], 
                   f'投影点\n({best_dist:.1f}px)', fontsize=9, color='orange')
            
            # 接続線
            ax.plot([tip[0], best_proj[0]], [tip[1], best_proj[1]], 
                   'orange', linewidth=2, linestyle='--')
            
            # 荷重節点として追加
            load_node_idx = len(all_nodes)
            all_nodes.append(best_proj)
            node_info.append({"type": "load_point"})
            ax.text(best_proj[0] - 40, best_proj[1] + 20, 
                   f'N{load_node_idx}\n(荷重)', fontsize=10, color='orange')
    
    # 凡例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='支点節点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='梁端点節点'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=10, label='荷重節点'),
        plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, 
                   label='スナップ接続'),
        plt.Line2D([0], [0], color='gray', linewidth=6, alpha=0.5, label='梁'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('connection_test.png', dpi=150, bbox_inches='tight')
    print("✅ 接続処理の視覚化を保存しました: connection_test.png")
    plt.show()

if __name__ == "__main__":
    print("🔍 接続処理の視覚的テスト\n")
    visualize_connection_test()
