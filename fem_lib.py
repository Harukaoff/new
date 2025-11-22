import numpy as np
np.set_printoptions(precision=4)
import pandas as pd
pd.set_option('display.max_rows', None)
import math
import time


def make_T3(angle): #変換マトリクス作成
    mu = math.sin(math.radians(angle))
    lamb = math.cos(math.radians(angle))

    T3 = np.array([
                [ lamb,   mu, 0,    0,    0, 0],
                [  -mu, lamb, 0,    0,    0, 0],
                [    0,    0, 1,    0,    0, 0],
                [    0,    0, 0, lamb,   mu, 0],
                [    0,    0, 0,  -mu, lamb, 0],
                [    0,    0, 0,    0,    0, 1]
                ])

    return T3

def esm(E, A, I, L, angle): #要素剛性マトリクス
    # E = 2.0*10**3 #部材ヤング係数
    # A = 6.0*10**3 #部材断面積
    # I = 2.0*10**3 #部材断面二次モーメント
    # L = 8000 #部材長さ
    # angle = 270

    matrix_L = np.array([
                        [ (E*A)/L,              0,             0, -(E*A)/L,              0,             0],
                        [       0,  (12*E*I)/L**3,  (6*E*I)/L**2,        0, -(12*E*I)/L**3,  (6*E*I)/L**2],
                        [       0,   (6*E*I)/L**2,     (4*E*I)/L,        0,  -(6*E*I)/L**2,     (2*E*I)/L],
                        [-(E*A)/L,              0,             0,  (E*A)/L,              0,             0],
                        [       0, -(12*E*I)/L**3, -(6*E*I)/L**2,        0,  (12*E*I)/L**3, -(6*E*I)/L**2],
                        [       0,   (6*E*I)/L**2,     (2*E*I)/L,        0,  -(6*E*I)/L**2,     (4*E*I)/L]
                        ])
    
    matrix_T3 = make_T3(angle)
    matrix_G = np.dot(matrix_T3.T, np.dot(matrix_L, matrix_T3))
    return matrix_G

def gsm(matrixs): #全体剛性マトリクス作成
    node = max(max([i[1] for i in matrixs])+1, max([i[2] for i in matrixs])+1)
    matrix = np.zeros((node*3, node*3))

    for i in matrixs:
        arr = np.zeros((node*3, node*3))
        arr[i[1]*3:i[1]*3+3, i[1]*3:i[1]*3+3] = i[0][:3, :3] #左上の3x3行列
        arr[i[1]*3:i[1]*3+3, i[2]*3:i[2]*3+3] = i[0][:3, 3:] #右上の3x3行列
        arr[i[2]*3:i[2]*3+3, i[1]*3:i[1]*3+3] = i[0][3:, :3] #左下の3x3行列
        arr[i[2]*3:i[2]*3+3, i[2]*3:i[2]*3+3] = i[0][3:, 3:] #右下の3x3行列
        matrix = matrix + arr

    return matrix

def d_r(e_l, n_d): #各接点の変位・反力計算
    #e_l : element_list
    #n_d : nodes_df

    matrix = gsm(e_l) #全体剛性マトリクス作成
    dl = d_l(e_l) #分布荷重考慮

    rc = n_d[['rc_x', 'rc_y', 'rc_m']].values.tolist() #拘束条件
    ef = n_d[['ef_x', 'ef_y', 'ef_m']].values.tolist() #外力条件

    matrix_ind = matrix.shape[0] #入力したマトリクスの大きさを把握する
    rc_ind = [3 * row + col for row, sublist in enumerate(rc) for col, value in enumerate(sublist) if value == 1] #拘束条件のインデックス
    ef_ind = [i for i in range(0, matrix_ind) if i not in rc_ind] #外力条件のインデックス

    aa = [dl[index] for index in ef_ind]
    ba = [dl[index] for index in rc_ind]

    mat = matrix[:, [False if i in rc_ind else True for i in range(matrix_ind)]]
    Kaa = mat[[False if i in rc_ind else True for i in range(matrix_ind)]]
    Kba = mat[[True if i in rc_ind else False for i in range(matrix_ind)]]

    Pa = [x - y for x, y in zip([sum(ef, [])[i] for i in ef_ind], aa)] #計算に必要な外力条件 >>> index= ef_ind
    Ua = np.linalg.pinv(Kaa) @ Pa #移動する節点の変位

    Pb = (Kba @ Ua) + ba #支点反力 >>> index= rc_ind

    Pa_list = [[x, y] for x, y in zip(ef_ind, Pa)] #節点インデックスと外力
    Pb_list = [[x, y] for x, y in zip(rc_ind, Pb)] #節点インデックスと支点反力
    Ua_list = [[x, y] for x, y in zip(ef_ind, Ua)] #節点インデックスと変位

    df = pd.DataFrame(index=range(int(matrix_ind/3)), columns=['Px', 'Py', 'M', 'u', 'v', 'theta']) #各節点の変位と力をdfにまとめる
    df.fillna(0, inplace=True)
    
    for i in Pa_list:
        ind, col = divmod(i[0], 3)
        df.iat[ind, col] = i[1] + dl[i[0]]
    
    for i in Pb_list:
        ind, col = divmod(i[0], 3)
        df.iat[ind, col] = i[1]

    for i in Ua_list:
        ind, col = divmod(i[0], 3)
        df.iat[ind, col+3] = i[1]

    return df

def member_stress(e_l, d_r, n_d): #部材応力計算
    # e_l : element_list
    # d_r : 各接点の変位・反力df

    step = 10 #形状関数xの増加量 1~

    node = n_d[['x', 'y']].values.tolist() #節点の座標リスト

    stress_list = []
    disp_df_list = []
    for i in e_l:
        length = i[6] #部材長
        angle = i[3] #部材角
        Ws, We = i[4], i[5]

        sin = math.sin(math.radians(angle)) #sin(theta)
        cos = math.cos(math.radians(angle)) #cos(theta)

        x = round(math.sin(math.radians(angle)),3) #力をx, yに分解
        y = round(math.cos(math.radians(angle)),3)

        if 0 <= angle < 90:
            x = -x
        elif 90 <= angle < 180:
            x, y = -x, -y
        elif 180 <= angle < 270:
            y = -y
        elif 270 <= angle:
            x = -x

        Qxs = (length/20) * (7*Ws*x + 3*We*x)
        Qxe = (length/20) * (3*Ws*x + 7*We*x)
        Qys = (length/20) * (7*Ws*y + 3*We*y)
        Qye = (length/20) * (3*Ws*y + 7*We*y)
        Ms = (length**2/60) * (3*Ws + 2*We)
        Me = -(length**2/60) * (2*Ws + 3*We)

        if Ws*x != 0 or Ws*y != 0 or We*x != 0 or We*y != 0:
            Wxy = [Ws*x, -Ws*y, 1, We*x, -We*y, 1]
        elif Ws*x == 0 and Ws*y == 0 and We*x == 0 and We*y == 0:
            Wxy = [0, 0, 0, 0, 0, 0]

        Kg = i[0]
        Ug = d_r.iloc[i[1], 3:].values.tolist() + d_r.iloc[i[2], 3:].values.tolist()

        Fg_G = (Kg @ Ug) + [Qxs, Qys, Ms, Qxe, Qye, Me]  #部材応力(基準座標)
        T3 = make_T3(i[3]) #変換マトリクス
        Fg_L = (T3 @ Fg_G).tolist() #部材応力(局所座標)
        Fg_L = [-Fg_L[0], Fg_L[1], Fg_L[2], Fg_L[3], -Fg_L[4], -Fg_L[5]] #部材応力の正負の整合を整理
        Wuv = T3 @ Wxy
        stress_list.append(Fg_L)

        # 変形・応力算出
        start = node[i[1]] #部材の始点座標[x, y]
        d_r_list = d_r.iloc[[i[1], i[2]], 3:].values.tolist() #端点節点それぞれの変位・回転角
        Us, Vs, Ts = d_r_list[0][0], d_r_list[0][1], d_r_list[0][2] #start節点の変位・回転角(標準座標)
        Ue, Ve, Te = d_r_list[1][0], d_r_list[1][1], d_r_list[1][2] #end節点の変位・回転角(標準座標)

        Us_l, Vs_l = Us*cos + Vs*sin, -Us*sin + Vs*cos #start節点の変位(局所座標)
        Ue_l, Ve_l = Ue*cos + Ve*sin, -Ue*sin + Ve*cos #end節点の変位(局所座標)

        disp_list = []
        for x in range(0,int(length)+1,step):
            Ux = (1 - x/length)*Us_l + (x/length)*Ue_l #材軸方向の変位
            Vx = np.array([(1 -3*x**2/length**2 + 2*x**3/length**3), (x - 2*x**2/length + x**3/length**2), (3*x**2/length**2 -2*x**3/length**3), (-1*x**2/length + x**3/length**2)]) @ [Vs_l, Ts, Ve_l, Te] #材軸直行方向の変位
            Tx = np.array([(-6*x/length**2 + 6*x**2/length**3), (1 - 4*x/length + 3*x**2/length**2), (6*x/length**2 - 6*x**2/length**2), (-2*x/length + 3*x**2/length**2)]) @ [Vs_l, Ts, Ve_l, Te] #各点の回転角

            dx = Ux * cos - Vx * sin #x方向変位(標準座標)
            dy = Ux * sin + Vx * cos #y方向変位(標準座標)

            c_x, c_y = x * cos - 0 * sin + start[0], x * sin + 0 * cos + start[1] #移動前地点x座標(x, y)

            N = -(Wuv[3] -Wuv[0])/2 * x ** 2/length -Wuv[0] * x + Fg_L[0] # 軸方向力
            Q = (Wuv[4] -Wuv[1])/2 * x ** 2/length +Wuv[1] * x + Fg_L[1] # せん断力
            M = (Wuv[4] -Wuv[1])/6 * x ** 3/length +Wuv[1]/2 * x ** 2 +Fg_L[1] * x -Fg_L[2] # モーメント

            disp_list.append([x, c_x, c_y, dx, dy, N, Q, M])

        disp_df = pd.DataFrame(data=np.array(disp_list), columns=['delta', 'x', 'y', 'dx', 'dy', 'N', 'Q', 'M']) #部材の始点からの距離がxである任意の点における変位

        if len(disp_df[(disp_df['x'] == node[i[2]][0]) & (disp_df['y'] == node[i[2]][1])]) == 0:
            disp_df.iloc[-1] = [length, node[i[2]][0], node[i[2]][1], Ue, Ve, Fg_L[3], Fg_L[4], -Fg_L[5]] #dfに終点情報を追加

        disp_df_list.append(disp_df)

    return disp_df_list

def d_l(e_l): #分布荷重のために加算する要素を作成
    # e_l : element_list

    qm_l = []
    for i in e_l:
        length = i[6]
        angle = i[3]
        Ws, We = i[4], i[5]
        start, end = i[1], i[2]

        x = round(math.sin(math.radians(angle)),3) #力をx, yに分解
        y = round(math.cos(math.radians(angle)),3)

        if 0 <= angle < 90:
            x = -x
        elif 90 <= angle < 180:
            x, y = -x, -y
        elif 180 <= angle < 270:
            y = -y
        elif 270 <= angle:
            x = -x

        Qxs = (length/20) * (7*Ws*x + 3*We*x) #各要素の応力算出
        Qxe = (length/20) * (3*Ws*x + 7*We*x)
        Qys = (length/20) * (7*Ws*y + 3*We*y)
        Qye = (length/20) * (3*Ws*y + 7*We*y)
        Ms = (length**2/60) * (3*Ws + 2*We)
        Me = -(length**2/60) * (2*Ws + 3*We)

        qm_l.append([start, [Qxs, Qys, Ms]])
        qm_l.append([end, [Qxe, Qye, Me]])

    dl_dict = {}

    for index, values in qm_l:
        if index in dl_dict:
            dl_dict[index] = [x + y for x, y in zip(dl_dict[index], values)]
        else:
            dl_dict[index] = values

    dist_load_list = [[index, values] for index, values in dl_dict.items()]

    flat_list = [item for sublist in dist_load_list for item in sublist[1]]

    return flat_list

def fem_calc(elements_df, nodes_df): #FEM解析プログラム
    #element_df : 部材df
    #nodes_df : 節点df

    elements_list = [[esm(i[1], i[2], i[3], i[4], i[5]), i[6], i[7], i[5], i[8], i[9], i[4]] for i in elements_df.itertuples()] #[esm, start, end, angle, Ws, We, length] <--- リストの中身

    D_R = d_r(elements_list, nodes_df) #各接点の変位・反力計算
    M_S = member_stress(elements_list, D_R, nodes_df) #部材応力計算

    return D_R, M_S




# モーメント:反時計回り正
# x:右向き正
# y:上向き正







