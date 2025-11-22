import numpy as np
from numpy.linalg import solve
import math
import os
import itertools
import pandas as pd
import copy


def request_angle(p1, p2): #2点の角度を求める---point1:[x1, y1]
    radian = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    degree = radian * (180 / math.pi)

    if degree >= 0:
        degree = degree
    elif degree < 0:
        degree = 360 + degree
    
    return degree

def length_line(p1, p2): #2点間の距離を求める p:[x, y] 
    length = math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    return length

def dis_p_l(line, point): #線と点の距離　(line=[[x1,y1], [x2,y2]], point=[x,y])
    fx = make_fx(line[0][0], line[0][1], line[1][0], line[1][1])
    dis = (abs(fx[0]*point[0] + fx[1]*point[1] + fx[2]))/math.sqrt(fx[0]**2 + fx[1]**2)
    return dis

def judge_p_s(line, point): #線分と点が範囲内かどうか
    Range = 30 #探索範囲

    start = line[0] #始点
    end = line[1] #終点

    if length_line(start, point) <= Range: #端点からの距離で判断
        return 's'
    
    elif length_line(end, point) <= Range: #端点からの距離で判断
        return 'e'
    
    elif dis_p_l(line, point) <= Range:
        ase = abs(request_angle(start, point)-request_angle(start, end))
        aes = abs(request_angle(end, point)-request_angle(end, start))
        if ase <= 90 and aes <= 90:
            return False
    
    else:
        return None

def detective_cropo(line1, line2): #2線分の交点を特定-------line1:[[x1,y1],[x2,y2]] /line2:[[x1,y1],[x2,y2]]
    fx1 = make_fx(line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    fx2 = make_fx(line2[0][0], line2[0][1], line2[1][0], line2[1][1])

    if (fx1[0] == 0) and (fx2[0] == 0): #2直線はX軸に対して平行に直線状に並ぶ

        if line1[1][0] < line2[0][0]:
            cross_point = [(line1[1][0]+line2[0][0])/2, line1[0][1]]
            return cross_point

        elif line1[0][0] > line2[1][0]:
            cross_point = [(line1[0][0]+line2[1][0])/2, line1[0][1]]
            return cross_point

    elif (fx1[1] == 0) and (fx2[1] == 0): #2直線はY軸に対して平行に直線状に並ぶ

        if line1[1][1] < line2[0][1]:
            cross_point = [line1[0][0], (line1[1][1]+line2[0][1])/2]
            return cross_point

        elif line1[0][1] > line2[1][1]:
            cross_point = [line1[0][0], (line1[0][1]+line2[1][1])/2]
            return cross_point

    elif (fx1[2] == fx2[2]) and (fx1[0]/fx1[1] == fx2[0]/fx2[1]): #2直線は直線状に並ぶ

        if line1[1][1] < line2[0][1]:
            cross_point = [(line1[1][0]+line2[0][0])/2, (line1[1][1]+line2[0][1])/2]
            return cross_point

        elif line1[0][1] > line2[1][1]:
            cross_point = [(line1[0][0]+line2[1][0])/2, (line1[0][1]+line2[1][1])/2]
            return cross_point

    else: #2直線は交点を有する

        left = [[fx1[0], fx1[1]],
                [fx2[0], fx2[1]]]
        right = [-fx1[2], -fx2[2]]

        points = solve(left, right)
        xp = points[0]
        yp = points[1]

        cross_point = [xp,yp] #交点座標

        return cross_point

def find_point_M(point_s, point_e, point, d): # 応力図(M)を出力するためのポイントを作成する
    # point_s, _e : 部材端点座標
    # point : 部材上の任意点座標
    # d : 応力

    c1 = [0, d] # 原点から長さdの座標を作成する
    angle = request_angle(point_s, point_e) # 部材角度
    c2 = rotate_point(c1, [0, 0], angle) #部材の角度に合わせて回転
    c3 = [c2[0]+point[0], c2[1]+point[1]]

    return c3

def find_point_NQ(point_s, point_e, point, d): # 応力図(N, Q)を出力するためのポイントを作成する
    # point_s, _e : 部材端点座標
    # point : 部材上の任意点座標
    # d : 応力

    c1 = [0, abs(d)] # 原点から長さdの座標を作成する
    angle = request_angle(point_s, point_e) # 部材角度
    # if 180 < angle: # 部材角度が180以上だった場合, 0<=x<180の域に変換する
    #     angle = angle-180

    # 部材角度による応力正負の分岐
    if angle < 45:
        if d < 0:
            c2 = rotate_point(c1, [0, 0], angle+180) #部材の角度に合わせて回転
        elif 0 <= d:
            c2 = rotate_point(c1, [0, 0], angle) #部材の角度に合わせて回転
    elif 45 <= angle <=225:
        if d < 0:
            c2 = rotate_point(c1, [0, 0], angle) #部材の角度に合わせて回転
        elif 0 <= d:
            c2 = rotate_point(c1, [0, 0], angle+180) #部材の角度に合わせて回転
    elif 225 < angle:
        if d < 0:
            c2 = rotate_point(c1, [0, 0], angle+180) #部材の角度に合わせて回転
        elif 0 <= d:
            c2 = rotate_point(c1, [0, 0], angle) #部材の角度に合わせて回転

    c3 = [c2[0]+point[0], c2[1]+point[1]]

    return c3

def rotate_point(point, center, angle): #点(point:[x, y])を(center:[x, y])を中心としてn°(angle)回転する
    sin, cos = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    x = (point[0] - center[0])*cos - (point[1] - center[1])*sin + center[0]
    y = (point[0] - center[0])*sin + (point[1] - center[1])*cos + center[1]

    return [x, y]

def make_fx(x1, y1, x2, y2): #ax+by+c=0の一次関数作成
    a=y2-y1
    b=x1-x2
    c=y1*x2-x1*y2

    return a,b,c

def make_line_list(point_list): #line_list生成
    line_list = [[point_list[i], point_list[i+1]] for i in range(len(point_list)-1)]
    return line_list

def find_feature(point_list, threshold): #特徴点探索
    se_len = length_line(point_list[0], point_list[-1]) #端点間距離
    p_len = [dis_p_l([point_list[0], point_list[-1]], i) for i in point_list[1:-1]] #各点と線の距離リスト

    try:
        f_p, d = p_len.index(max(p_len)), max(p_len) #特徴点のインデックス, 線との距離
        if d / se_len > threshold: #距離と端点長さの比を確認
            return [point_list[0:f_p+1], point_list[f_p:]]
        else:
            return None #比が設定値よりも小さかった場合Noneを返す
    
    except ValueError: #ポイントリストの中に2つしかポイントがなかった場合[p_len]に値がないため, ValueErrorになる
        return None

def find_fs(point_list, repeat): #特徴点を必要数探索する
    first = find_feature(point_list, 0.1)

    if first != None: #最初の判定でNoneではなかった場合1つ以上の特徴点を持つため,df化して特徴点の範囲ごとに分けて特徴点を抽出する
        df = pd.DataFrame(data=[first])
        n = 0
        while len(df.iloc[n].values.tolist()) < repeat: #whileを抜け出す条件 : 設定した特徴点数(repeat) or 抽出ができなくなった場合
            l = []
            for i in df.iloc[n].values.tolist():
                x = find_feature(i, 0.1)
                if x != None:
                    l.append(x[0]), l.append(x[1])
                else:
                    l.append(i)

            df = pd.concat([df, pd.DataFrame(data=[l])])
            n += 1

            if df.iloc[n].values.tolist() == df.iloc[n-1].values.tolist(): #抽出ができなくなった場合
                break
    
        features = [i[0] for i in df.tail(1).values.tolist()[0]] #始終点と特徴点のリスト
        features.append(df.tail(1).values.tolist()[0][-1][-1])

    else:
        features = [point_list[0], point_list[-1]] #始終点
    
    return features

def judg_poly(point_list, d): #多角形か多角線分か単線分かを判定する
    c_point_list = copy.deepcopy(point_list)

    adjust = 15 #角度調整のパラメータ

    if (length_line(c_point_list[0], c_point_list[-1]) < d) and (len(c_point_list) > 3): #線分端点間距離が短く, ポイント数が4以上の場合, 多角形である
        # print('Polygon')

        #ポリゴンの場合, 始点と終点の座標をそれぞれの延長線の交点に変更する
        cross_point = detective_cropo([c_point_list[0], c_point_list[1]], [c_point_list[-2], c_point_list[-1]])
        c_point_list[0], c_point_list[-1] = cross_point, cross_point

        #その後, 始点・終点の位置を固定したまま設定した角度に直す
        lines = make_line_list(c_point_list) #一旦, ライン変換して座標整理を行いやすくする

        for i in lines[1:-1]:
            angle = request_angle(i[0], i[1]) #現単線分の角度
            dif = [abs(i - angle) for i in range(0, 361, adjust)]
            aor = list(range(0, 361, adjust))[dif.index(min(dif))] - angle #必要な回転角(Angle of rotation の略)

            i[0] = rotate_point(i[0], [(i[0][0] + i[1][0]) / 2, (i[0][1] + i[1][1]) / 2], aor)
            i[-1] = rotate_point(i[-1], [(i[0][0] + i[1][0]) / 2, (i[0][1] + i[1][1]) / 2], aor)
        
        angle = request_angle(lines[0][0], lines[0][1]) #始線分の角度
        dif = [abs(i - angle) for i in range(0, 361, adjust)]
        aor = list(range(0, 361, adjust))[dif.index(min(dif))] - angle #必要な回転角(Angle of rotation の略)

        lines[0][1] = rotate_point(lines[0][1], lines[0][0], aor)

        angle = request_angle(lines[-1][0], lines[-1][1]) #終線分の角度
        dif = [abs(i - angle) for i in range(0, 361, adjust)]
        aor = list(range(0, 361, adjust))[dif.index(min(dif))] - angle #必要な回転角(Angle of rotation の略)

        lines[-1][0] = rotate_point(lines[-1][0], lines[-1][1], aor)

        #角度調整が終了した後, 隣合う線分の交点を求めてそれを特徴点として[c_point_list]を書き換える
        c_point_list[0] = lines[0][0] #ポリラインの始点
        for i in range(len(lines)-1):
            c_point_list[i+1] = detective_cropo(lines[i], lines[i+1])
        c_point_list[-1] = lines[-1][1] #ポリラインの終点
    
    elif len(c_point_list) > 2: #ポイント数が3以上の場合, 多角線分である
        # print('Polyline')

        #ポリラインの場合, 端部の線分は始点と終点を中心に, 間の線分は中点を中心に設定した角度に直す
        lines = make_line_list(c_point_list) #一旦, ライン変換して座標整理を行いやすくする

        for i in lines[1:-1]:
            angle = request_angle(i[0], i[1]) #現単線分の角度
            dif = [abs(i - angle) for i in range(0, 361, adjust)]
            aor = list(range(0, 361, adjust))[dif.index(min(dif))] - angle #必要な回転角(Angle of rotation の略)

            i[0] = rotate_point(i[0], [(i[0][0] + i[1][0]) / 2, (i[0][1] + i[1][1]) / 2], aor)
            i[-1] = rotate_point(i[-1], [(i[0][0] + i[1][0]) / 2, (i[0][1] + i[1][1]) / 2], aor)
        
        angle = request_angle(lines[0][0], lines[0][1]) #始線分の角度
        dif = [abs(i - angle) for i in range(0, 361, adjust)]
        aor = list(range(0, 361, adjust))[dif.index(min(dif))] - angle #必要な回転角(Angle of rotation の略)

        lines[0][1] = rotate_point(lines[0][1], lines[0][0], aor)

        angle = request_angle(lines[-1][0], lines[-1][1]) #終線分の角度
        dif = [abs(i - angle) for i in range(0, 361, adjust)]
        aor = list(range(0, 361, adjust))[dif.index(min(dif))] - angle #必要な回転角(Angle of rotation の略)

        lines[-1][0] = rotate_point(lines[-1][0], lines[-1][1], aor)

        #角度調整が終了した後, 隣合う線分の交点を求めてそれを特徴点として[c_point_list]を書き換える
        c_point_list[0] = lines[0][0] #ポリラインの始点
        for i in range(len(lines)-1):
            c_point_list[i+1] = detective_cropo(lines[i], lines[i+1])
        c_point_list[-1] = lines[-1][1] #ポリラインの終点

    elif len(c_point_list) == 2: #ポイント数が2の場合, 単線分である
        # print('line')

        #単線分の場合, 線分の中点を中心に設定した角度に直す
        angle = request_angle(c_point_list[0], c_point_list[1]) #現単線分の角度
        dif = [abs(i - angle) for i in range(0, 361, adjust)]
        aor = list(range(0, 361, adjust))[dif.index(min(dif))] - angle #必要な回転角(Angle of rotation の略)

        c_point_list[0] = rotate_point(c_point_list[0], [(c_point_list[0][0] + c_point_list[1][0]) / 2, (c_point_list[0][1] + c_point_list[1][1]) / 2], aor)
        c_point_list[-1] = rotate_point(c_point_list[-1], [(c_point_list[0][0] + c_point_list[1][0]) / 2, (c_point_list[0][1] + c_point_list[1][1]) / 2], aor)

    return c_point_list

def make_line(node_df, line_df): #node_dfとline_dfからlineのリストを作成する
    lines_list = []

    for i, line in line_df.iterrows(): #line_dfから行ごとに始点・終点を取得する
        start_node = node_df[node_df['No'] == line['s_node']][['x', 'y']].values.tolist()
        end_node = node_df[node_df['No'] == line['e_node']][['x', 'y']].values.tolist()
        lines_list.append(start_node+end_node)
    
    return lines_list

def add_df(points): #入力した線分をnode_dfとline_dfに追加する
    #points : 追加入力した線分の節点データ

    global node_df, line_df  # グローバル変数として宣言

    judges = []
    #入力した線分に近接する節点を探索するプログラム
    for l in range(len(points)-1):
        line = points[l:l+2]
        judge = []
        for index, row in node_df.iterrows():
            point = row[['x', 'y']].values.tolist()
            judge.append(judge_p_s(line, point)) #線分と点が範囲内かどうか

        judges.append(judge)

    new_points = judg_poly(points, 50)

    for j in range(len(judges)):

        for r in range(len(judges[j])): #先に端点の座標を確定させるためfor文を繰り返す, ****改善の余地あり****
            if judges[j][r] == 's':
                p = node_df[['x', 'y']].iloc[r].values.tolist()
                new_points[j] = p
            elif judges[j][r] == 'e':
                p = node_df[['x', 'y']].iloc[r].values.tolist()
                new_points[j+1] = p
        
        for r in range(len(judges[j])):
            if judges[j][r] == False:
                p = node_df[['x', 'y']].iloc[r].values.tolist()
                n = node_df[['No']].iloc[r].values.tolist()
                px = [p[0]-(new_points[j][1]-new_points[j+1][1]), p[1]+(new_points[j][0]-new_points[j+1][0])] #点から垂線になるように移動した座標 
                cross = detective_cropo(new_points[j:j+2], [p, px]) #点からの垂線の交点
                node_df.loc[node_df['No'] == n[0], 'x'] = cross[0] #交点へ移動
                node_df.loc[node_df['No'] == n[0], 'y'] = cross[1]


    #入力した線分をnode_df,line_dfに追加するプログラム
    if len(node_df) == 0:
        tail_No = -1 #現在のnode_dfの最終行節点番号
    
    elif len(node_df) > 0:
        tail_No = node_df['No'].iloc[-1] #現在のnode_dfの最終行節点番号

    cnt = 1
    for p in new_points: #node_df追加
        if len(node_df[(node_df['x'] == p[0]) & (node_df['y'] == p[1])]) == 0: #節点の既存を確認
            new_row = {'No': tail_No+cnt, 'x': p[0], 'y': p[1]}
            node_df = pd.concat([node_df, pd.DataFrame([new_row])], ignore_index=True)
            cnt += 1

    for l in range(len(new_points)-1): #line_df追加
        start = new_points[l] #始点座標
        end = new_points[l+1] #終点座標

        s_node = node_df[(node_df['x'] == start[0]) & (node_df['y'] == start[1])]['No'].values[0]
        e_node = node_df[(node_df['x'] == end[0]) & (node_df['y'] == end[1])]['No'].values[0]

        new_row = {'s_node': s_node, 'e_node': e_node}
        line_df = pd.concat([line_df, pd.DataFrame([new_row])], ignore_index=True)

    return node_df, line_df

def input_condition(point, button_stats): #トグルボタンの状態と入力点から各条件の入力を判定する
    #point: [x, y]
    threshold_length = 20 #入力点検知の閾値

    global node_df, line_df, input_df  # グローバル変数として宣言

    if button_stats == 'ピン支点': #ピン支点の場合
        input_nodes, Len = [], [] #入力節点の候補座標
        for p in range(len(node_df)):
            length = length_line(point, node_df.iloc[p].values[1:3]) #入力点とnode_dfに存在する各接点との距離を算出
            if length < threshold_length:
                input_nodes.append(node_df.iloc[p].values[0])
                Len.append(length)

        if len(Len) != 0:
            node = input_nodes[Len.index(min(Len))] #入力する節点番号
            change_row = input_df[input_df['No'] == node]
            if change_row.empty:
                new_row = {'No': node, 'stick': 0, 's_pos' : 1}
                input_df = pd.concat([input_df, pd.DataFrame([new_row])], ignore_index=True)
            elif ~change_row.empty:
                if change_row['stick'].iloc[0] == 0:
                    if change_row['s_pos'].iloc[0] == 1:
                        change_row['s_pos'] = 0
                    else:
                        change_row['s_pos'] = change_row['s_pos'] + 1
                else:
                    change_row['stick'] = 0
                    change_row['s_pos'] = 1
                input_df.update(change_row)

    if button_stats == 'ピンローラー支点': #ピンローラー支点の場合
        input_nodes, Len = [], [] #入力節点の候補座標
        for p in range(len(node_df)):
            length = length_line(point, node_df.iloc[p].values[1:3]) #入力点とnode_dfに存在する各接点との距離を算出
            if length < threshold_length:
                input_nodes.append(node_df.iloc[p].values[0])
                Len.append(length)

        if len(Len) != 0:
            node = input_nodes[Len.index(min(Len))] #入力する節点番号
            change_row = input_df[input_df['No'] == node]
            if change_row.empty:
                new_row = {'No': node, 'stick': 1, 's_pos' : 1}
                input_df = pd.concat([input_df, pd.DataFrame([new_row])], ignore_index=True)
            elif ~change_row.empty:
                if change_row['stick'].iloc[0] == 1:
                    if change_row['s_pos'].iloc[0] == 2:
                        change_row['s_pos'] = 0
                    else:
                        change_row['s_pos'] = change_row['s_pos'] + 1
                else:
                    change_row['stick'] = 1
                    change_row['s_pos'] = 1
                input_df.update(change_row)

    if button_stats == '固定端': #固定支点の場合
        input_nodes, Len = [], [] #入力節点の候補座標
        for p in range(len(node_df)):
            length = length_line(point, node_df.iloc[p].values[1:3]) #入力点とnode_dfに存在する各接点との距離を算出
            if length < threshold_length:
                input_nodes.append(node_df.iloc[p].values[0])
                Len.append(length)

        if len(Len) != 0:
            node = input_nodes[Len.index(min(Len))] #入力する節点番号
            change_row = input_df[input_df['No'] == node]
            if change_row.empty:
                new_row = {'No': node, 'stick': 2, 's_pos' : 1}
                input_df = pd.concat([input_df, pd.DataFrame([new_row])], ignore_index=True)
            elif ~change_row.empty:
                if change_row['stick'].iloc[0] == 2:
                    if change_row['s_pos'].iloc[0] == 1:
                        change_row['s_pos'] = 0
                    else:
                        change_row['s_pos'] = change_row['s_pos'] + 1
                else:
                    change_row['stick'] = 2
                    change_row['s_pos'] = 1
                input_df.update(change_row)

    if button_stats == '集中荷重': #集中荷重の場合
        input_nodes, Len = [], [] #入力節点の候補座標
        for p in range(len(node_df)):
            length = length_line(point, node_df.iloc[p].values[1:3]) #入力点とnode_dfに存在する各接点との距離を算出
            if length < threshold_length:
                input_nodes.append(node_df.iloc[p].values[0])
                Len.append(length)

        if len(Len) != 0:
            node = input_nodes[Len.index(min(Len))] #入力する節点番号
            change_row = input_df[input_df['No'] == node]
            if change_row.empty:
                new_row = {'No': node, 'load': 5, 'l_pos' : 1}
                input_df = pd.concat([input_df, pd.DataFrame([new_row])], ignore_index=True)
            elif ~change_row.empty:
                if change_row['load'].iloc[0] == 5:
                    if change_row['l_pos'].iloc[0] == 4:
                        change_row['l_pos'] = 0
                    else:
                        change_row['l_pos'] = change_row['l_pos'] + 1
                else:
                    change_row['load'] = 5
                    change_row['l_pos'] = 1
                input_df.update(change_row)

    if button_stats == 'モーメント荷重': #モーメント荷重の場合
        input_nodes, Len = [], [] #入力節点の候補座標
        for p in range(len(node_df)):
            length = length_line(point, node_df.iloc[p].values[1:3]) #入力点とnode_dfに存在する各接点との距離を算出
            if length < threshold_length:
                input_nodes.append(node_df.iloc[p].values[0])
                Len.append(length)

        if len(Len) != 0:
            node = input_nodes[Len.index(min(Len))] #入力する節点番号
            change_row = input_df[input_df['No'] == node]
            if change_row.empty:
                new_row = {'No': node, 'load': 7, 'l_pos' : 1}
                input_df = pd.concat([input_df, pd.DataFrame([new_row])], ignore_index=True)
            elif ~change_row.empty:
                if change_row['load'].iloc[0] == 7:
                    if change_row['l_pos'].iloc[0] == 2:
                        change_row['l_pos'] = 0
                    else:
                        change_row['l_pos'] = change_row['l_pos'] + 1
                else:
                    change_row['load'] = 7
                    change_row['l_pos'] = 1
                input_df.update(change_row)

    return input_df

def draw_condition(): #input_dfの情報から条件入力を可視化する為の線分リストを作成する

    global node_df, line_df, input_df  # グローバル変数として宣言

    dic = {0:[[[0, 0], [25, -45], [-25, -45], [0, 0]]],
            1:[[[0, 0], [25, -45], [-25, -45], [0, 0]], [[25, -50], [-25, -50]]],
            2:[[[-25, 0], [25, 0], [25, -25], [-25, -25], [-25, 0]]],
            5:[[[-10, 15], [0, 0], [10, 15]], [[0, 0], [0, 40]]],
            7:[[[0, -10], [15, -20], [0, -30]], [[0, -10], [-15, -20], [0, -30]]]}

    element_df = input_df.replace({'s_pos': 0, 'l_pos': 0}, np.nan).copy() #posに0を含むデータに対してnanへの置換を実行,追加入力時に支障が生じないようにcopyとする
    stick_df = element_df[['No', 'stick', 's_pos']].dropna() #拘束条件DF
    load_df = element_df[['No', 'load', 'l_pos']].dropna() #荷重条件DF

    draw_list = []
    if len(stick_df) != 0:
        for index, row in stick_df.iterrows():
            draw_list.append([row['No'], dic[row['stick']], 'Line', row['s_pos']])
    if len(load_df) != 0:
        for index, row in load_df.iterrows():
            if row['load'] == 7:
                draw_list.append([row['No'], dic[row['load']], '7', row['l_pos']])
            else:
                draw_list.append([row['No'], dic[row['load']], 'Line', row['l_pos']])

    lines = [] #条件描写のための線分リストを作成
    for d in draw_list:
        if d[2] == 'Line':
            coordinate = node_df[node_df['No'] == d[0]].values.tolist()[0][1:]
            lines.extend([['line']+[[rotate_point(i2, [0, 0], -90*(d[3]-1))[0]+ coordinate[0], rotate_point(i2, [0, 0], -90*(d[3]-1))[1]+ coordinate[1]] for i2 in i1] for i1 in d[1]])
        elif d[2] == '7':
            coordinate = node_df[node_df['No'] == d[0]].values.tolist()[0][1:]
            r = 40
            if d[3] == 1:
                lines.extend([['circle', coordinate, r, 90, -180]])
                lines.extend([['triangle']+[[i2[0]+ coordinate[0], i2[1]+ coordinate[1]] for i2 in i1] for i1 in d[1][:1]])
            if d[3] == 2:
                lines.extend([['circle', coordinate, r, -90, 180]])
                lines.extend([['triangle']+[[i2[0]+ coordinate[0], i2[1]+ coordinate[1]] for i2 in i1] for i1 in d[1][1:]])

    return lines

def make_dfs(): # FEMプログラムへ渡すためのデータ作成
    global node_df, line_df, input_df  # グローバル変数として宣言

    elements_df = pd.DataFrame(columns=['young', 'area', 's_moment', 'length', 'angle', 'start', 'end', 'Ws', 'We'])
    elements_df[['start', 'end']] = line_df[['s_node', 'e_node']]
    for i, r in elements_df.iterrows():
        elements_df['length'][i] = length_line(node_df[node_df['No'] == r['start']][['x', 'y']].values[0].tolist(), node_df[node_df['No'] == r['end']][['x', 'y']].values[0].tolist())
        elements_df['angle'][i] = request_angle(node_df[node_df['No'] == r['start']][['x', 'y']].values[0].tolist(), node_df[node_df['No'] == r['end']][['x', 'y']].values[0].tolist())
        elements_df['young'][i] = 2.0*10**2
        elements_df['area'][i] = 9.0*10**2
        elements_df['s_moment'][i] = 6.75*10**4

    elements_df = elements_df.fillna(0) #FEMプログラムへ渡すためのelementデータ

    nodes_df = pd.DataFrame(columns=['x', 'y', 'rc_x', 'rc_y', 'rc_m', 'ef_x', 'ef_y', 'ef_m'])
    nodes_df[['x', 'y']] = node_df[['x', 'y']]
    for i, r in nodes_df.iterrows():

        #拘束条件置換
        if len(input_df[input_df['No'] == i]) != 0 and not pd.isnull(input_df[input_df['No'] == i]['stick'].values[0]):
            if input_df[input_df['No'] == i]['stick'].values[0] == 0: #ピン支点
                nodes_df['rc_x'][i] = 1
                nodes_df['rc_y'][i] = 1
            elif input_df[input_df['No'] == i]['stick'].values[0] == 1: #ピンローラー支点
                if input_df[input_df['No'] == i]['s_pos'].values[0] == 1:
                    nodes_df['rc_y'][i] = 1
                elif input_df[input_df['No'] == i]['s_pos'].values[0] == 2:
                    nodes_df['rc_x'][i] = 1
            elif input_df[input_df['No'] == i]['stick'].values[0] == 2: #固定端
                nodes_df['rc_x'][i] = 1
                nodes_df['rc_y'][i] = 1
                nodes_df['rc_m'][i] = 1

        #荷重条件置換
        if len(input_df[input_df['No'] == i]) != 0 and not pd.isnull(input_df[input_df['No'] == i]['load'].values[0]):
            if input_df[input_df['No'] == i]['load'].values[0] == 5: #集中荷重
                if input_df[input_df['No'] == i]['l_pos'].values[0] == 1:
                    nodes_df['ef_y'][i] = -10
                elif input_df[input_df['No'] == i]['l_pos'].values[0] == 2:
                    nodes_df['ef_x'][i] = -10
                elif input_df[input_df['No'] == i]['l_pos'].values[0] == 3:
                    nodes_df['ef_y'][i] = 10
                elif input_df[input_df['No'] == i]['l_pos'].values[0] == 4:
                    nodes_df['ef_x'][i] = 10
            elif input_df[input_df['No'] == i]['load'].values[0] == 7: #モーメント荷重
                if input_df[input_df['No'] == i]['l_pos'].values[0] == 1:
                    nodes_df['ef_m'][i] = 10
                elif input_df[input_df['No'] == i]['l_pos'].values[0] == 2:
                    nodes_df['ef_m'][i] = -10

    nodes_df = nodes_df.fillna(0) #FEMプログラムへ渡すためのnodesデータ
    return elements_df, nodes_df

def make_figure(dfs): # 応力図作図のためのデータ作成
    #element_df : 部材df
    #nodes_df : 節点df
    #dfs : 応力・変位dfのリスト群
    #position : 0,1,2,3

    mag = 25 #表示倍率

    # それぞれの値の最大値を決定する
    sample_dt, sample_N, sample_Q, sample_M = [], [], [], []

    for i in dfs:
        i = i.round(4)
        sample_dt = sample_dt + sum(i[['dx', 'dy']].values.tolist(), [])
        sample_N = sample_N + i['N'].values.tolist()
        sample_Q = sample_Q + i['Q'].values.tolist()
        sample_M = sample_M + i['M'].values.tolist()

    max_dt = max(list(map(abs, sample_dt))) #スケーリング用のmax値(変位)
    max_N = max(list(map(abs, sample_N))) #スケーリング用のmax値(軸方向)
    max_Q = max(list(map(abs, sample_Q))) #スケーリング用のmax値(軸直交方向)
    max_M = max(list(map(abs, sample_M))) #スケーリング用のmax値(回転)

    df_list = []
    for d in dfs:
        dfc = d.copy().round(4)
        dfc['dx'] = (dfc['dx']/max_dt)*mag
        dfc['dy'] = (dfc['dy']/max_dt)*mag
        dfc['N'] = (dfc['N']/max_N)*mag
        dfc['Q'] = (dfc['Q']/max_Q)*mag
        dfc['M'] = (dfc['M']/max_M)*mag

        dfc = dfc.fillna(0)

        dfc['ax'], dfc['ay'] = dfc['x']+dfc['dx'], dfc['y']+dfc['dy']
        dfc['Nx'], dfc['Ny'], dfc['Qx'], dfc['Qy'], dfc['Mx'], dfc['My'] = 0, 0, 0, 0, 0, 0

        for i, r in dfc.iterrows():
            Np = find_point_NQ([dfc.iloc[0]['x'], dfc.iloc[0]['y']], [dfc.iloc[-1]['x'], dfc.iloc[-1]['y']], [r['x'], r['y']], r['N'])
            Qp = find_point_NQ([dfc.iloc[0]['x'], dfc.iloc[0]['y']], [dfc.iloc[-1]['x'], dfc.iloc[-1]['y']], [r['x'], r['y']], r['Q'])
            Mp = find_point_M([dfc.iloc[0]['x'], dfc.iloc[0]['y']], [dfc.iloc[-1]['x'], dfc.iloc[-1]['y']], [r['x'], r['y']], -r['M'])
            dfc['Nx'].iloc[i], dfc['Ny'].iloc[i] = Np[0], Np[1]
            dfc['Qx'].iloc[i], dfc['Qy'].iloc[i] = Qp[0], Qp[1]
            dfc['Mx'].iloc[i], dfc['My'].iloc[i] = Mp[0], Mp[1]

        df_list.append(dfc)

    return df_list

def draw_fig(df_list): # 作図用データから作図データ形式への変換
    deform_list, N_list_p, N_list_m, Q_list_p, Q_list_m, M_list_p, M_list_m = [], [], [], [], [], [], []

    for df in df_list:
        deform_list.append(sum(df[['ax', 'ay']].values.tolist(), [])) # 変形図のポイントリスト

        N_p = df[df['N'] >= 0]
        if len(N_p) > 0:
            N_list_p.append(sum(N_p[['Nx', 'Ny']].values.tolist(), []))
            N_list_p.append(N_p.iloc[0][['x', 'y', 'Nx', 'Ny']].values.tolist())
            for i, r in N_p[2::2].iterrows():
                N_list_p.append(r[['x', 'y', 'Nx', 'Ny']].values.tolist())
            N_list_p.append(N_p.iloc[-1][['x', 'y', 'Nx', 'Ny']].values.tolist())

        N_m = df[df['N'] < 0]
        if len(N_m) > 0:
            N_list_m.append(sum(N_m[['Nx', 'Ny']].values.tolist(), []))
            N_list_m.append(N_m.iloc[0][['x', 'y', 'Nx', 'Ny']].values.tolist())
            for i, r in N_m[2::2].iterrows():
                N_list_m.append(r[['x', 'y', 'Nx', 'Ny']].values.tolist())
            N_list_m.append(N_m.iloc[-1][['x', 'y', 'Nx', 'Ny']].values.tolist())

        Q_p = df[df['Q'] >= 0]
        if len(Q_p) > 0:
            Q_list_p.append(sum(Q_p[['Qx', 'Qy']].values.tolist(), []))
            Q_list_p.append(Q_p.iloc[0][['x', 'y', 'Qx', 'Qy']].values.tolist())
            for i, r in Q_p[2::2].iterrows():
                Q_list_p.append(r[['x', 'y', 'Qx', 'Qy']].values.tolist())
            Q_list_p.append(Q_p.iloc[-1][['x', 'y', 'Qx', 'Qy']].values.tolist())

        Q_m = df[df['Q'] < 0]
        if len(Q_m) > 0:
            Q_list_m.append(sum(Q_m[['Qx', 'Qy']].values.tolist(), []))
            Q_list_m.append(Q_m.iloc[0][['x', 'y', 'Qx', 'Qy']].values.tolist())
            for i, r in Q_m[2::2].iterrows():
                Q_list_m.append(r[['x', 'y', 'Qx', 'Qy']].values.tolist())
            Q_list_m.append(Q_m.iloc[-1][['x', 'y', 'Qx', 'Qy']].values.tolist())

        M_p = df[df['M'] >= 0]
        if len(M_p) > 0:
            M_list_p.append(sum(M_p[['Mx', 'My']].values.tolist(), []))
            M_list_p.append(M_p.iloc[0][['x', 'y', 'Mx', 'My']].values.tolist())
            for i, r in M_p[2::2].iterrows():
                M_list_p.append(r[['x', 'y', 'Mx', 'My']].values.tolist())
            M_list_p.append(M_p.iloc[-1][['x', 'y', 'Mx', 'My']].values.tolist())

        M_m = df[df['M'] < 0]
        if len(M_m) > 0:
            M_list_m.append(sum(M_m[['Mx', 'My']].values.tolist(), []))
            M_list_m.append(M_m.iloc[0][['x', 'y', 'Mx', 'My']].values.tolist())
            for i, r in M_m[2::2].iterrows():
                M_list_m.append(r[['x', 'y', 'Mx', 'My']].values.tolist())
            M_list_m.append(M_m.iloc[-1][['x', 'y', 'Mx', 'My']].values.tolist())

    return [deform_list, N_list_p, N_list_m, Q_list_p, Q_list_m, M_list_p, M_list_m]





