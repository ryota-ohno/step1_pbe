##tetracene層内計算
import os
os.environ['HOME'] ='/home/ohno'
import pandas as pd
import time
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.environ['HOME'],'Working/interaction/'))
from make_8_xyz_new import exec_gjf##計算した点のxyzfileを出す
from utils import get_E
import argparse
import numpy as np
from scipy import signal
import scipy.spatial.distance as distance
import random

def main_process(args):
    auto_dir = args.auto_dir
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussian'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)
    
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['a','b','theta','E','E1','E2','E3','status'])##いじる
        df_E.to_csv(auto_csv_path,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    auto_csv_path1 = os.path.join(auto_dir,'step1_1.csv')
    if not os.path.exists(auto_csv_path1):        
        df_E_1 = pd.DataFrame(columns = ['a','theta','E1','machine_type','status','file_name'])##いじる
        df_E_1.to_csv(auto_csv_path1,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    auto_csv_path2 = os.path.join(auto_dir,'step1_2.csv')
    if not os.path.exists(auto_csv_path2):        
        df_E_2 = pd.DataFrame(columns = ['b','theta','E2','machine_type','status','file_name'])##いじる
        df_E_2.to_csv(auto_csv_path2,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    auto_csv_path3 = os.path.join(auto_dir,'step1_3.csv')
    if not os.path.exists(auto_csv_path3):        
        df_E_3 = pd.DataFrame(columns = ['a','b','theta','E3','machine_type','status','file_name'])##いじる
        df_E_3.to_csv(auto_csv_path3,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    os.chdir(os.path.join(args.auto_dir,'gaussian'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(args.auto_dir,args.monomer_name,args.num_nodes,args.max_nodes,args.isTest)##argsの中身を取る
        time.sleep(1)

def listen(auto_dir,monomer_name,num_nodes,max_nodes,isTest):##args自体を引数に取るか中身をばらして取るかの違い
    maxnum_machine2 = 3#int(num_nodes/2) ##多分俺のために空けていてくださったので2 3にする
    fixed_param_keys = ['theta'];opt_param_keys_1 = ['a'];opt_param_keys_2 = ['b']
    
    auto_csv_1 = os.path.join(auto_dir,'step1_1.csv');df_E_1 = pd.read_csv(auto_csv_1)
    df_prg_1 = df_E_1.loc[df_E_1['status']=='InProgress',fixed_param_keys+opt_param_keys_1+['machine_type']]
    machine_type_list_1 = df_prg_1['machine_type'].values.tolist();len_prg_1=len(df_prg_1)
    for idx,row in df_prg_1.iterrows():
        params_dict1_ = row[fixed_param_keys + opt_param_keys_1].to_dict()
        file_name1=make_file(monomer_name,params_dict1_,structure_type=1)##辞書をつくってそこにopt_1とopt_2でファイル名作成
        log_filepath1 = os.path.join(*[auto_dir,'gaussian',file_name1])
        if not(os.path.exists(log_filepath1)):#logファイルが生成される直前だとまずいので
            continue
        E_list1=get_E(log_filepath1)
        if len(E_list1)!=1 :##get Eの長さは計算した分子の数
            continue
        else:
            len_prg_1-=1;machine_type_list_1.remove(machine_type)
            E1=float(E_list1[0])##8分子に向けてep1,ep2作成　ep1:b ep2:a
            df_E_1.loc[idx, ['E1','status']] = [E1,'Done']
            df_E_1.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    
    
    auto_csv_2 = os.path.join(auto_dir,'step1_2.csv')
    df_E_2 = pd.read_csv(auto_csv_2)
    df_prg_2 = df_E_2.loc[df_E_2['status']=='InProgress', fixed_param_keys+opt_param_keys_2+['machine_type']]
    machine_type_list_2 = df_prg_2['machine_type'].values.tolist()
    len_prg_2 = len(df_prg_2)

    for idx, row in df_prg_2.iterrows():
        params_dict2_ = row[fixed_param_keys + opt_param_keys_2].to_dict()
        file_name2=make_file(monomer_name,params_dict2_,structure_type=2)##辞書をつくってそこにopt_1とopt_2でファイル名作成
        log_filepath2 = os.path.join(*[auto_dir, 'gaussian', file_name2])
        if not(os.path.exists(log_filepath2)):
            continue
        E_list2 = get_E(log_filepath2)
        if len(E_list2) != 1:
            continue
        else:
            len_prg_2 -= 1
            machine_type_list_2.remove(machine_type)
            E2 = float(E_list2[0])  # Updated to E2
            df_E_2.loc[idx, ['E2', 'status']] = [E2, 'Done']
            df_E_2.to_csv(auto_csv_2, index=False)  # Updated to auto_csv_2
            break  # Break after one iteration

    
    auto_csv_3 = os.path.join(auto_dir, 'step1_3.csv')
    df_E_3 = pd.read_csv(auto_csv_3)
    df_prg_3 = df_E_3.loc[df_E_3['status'] == 'InProgress', fixed_param_keys+opt_param_keys_1 + opt_param_keys_2 + ['machine_type']]
    machine_type_list_3 = df_prg_3['machine_type'].values.tolist()
    len_prg_3 = len(df_prg_3)

    for idx, row in df_prg_3.iterrows():
        params_dict3_ = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2].to_dict()
        file_name3=make_file(monomer_name,params_dict3_,structure_type=3)##辞書をつくってそこにopt_1とopt_2でファイル名作成
        log_filepath3 = os.path.join(*[auto_dir, 'gaussian', file_name3])
        if not (os.path.exists(log_filepath3)):
            continue
        E_list3 = get_E(log_filepath3)
        if len(E_list3) != 1:
            continue
        else:
            len_prg_3 -= 1
            machine_type_list_3.remove(machine_type)
            E3 = float(E_list3[0])  # Updated to E3
            df_E_3.loc[idx, ['E3', 'status']] = [E3, 'Done']
            df_E_3.to_csv(auto_csv_3, index=False)  # Updated to auto_csv_3
            break  # Break after one iteration


    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E = pd.read_csv(auto_csv)
    df_prg = df_E.loc[df_E['status']=='InProgress',fixed_param_keys+opt_param_keys_1+opt_param_keys_2]
    
    for idx,row in df_prg.iterrows():
        params_dict1_ = row[fixed_param_keys + opt_param_keys_1].to_dict()
        params_dict2_ = row[fixed_param_keys + opt_param_keys_2].to_dict()
        params_dict3_ = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2].to_dict()
        file_name1=make_file(monomer_name,params_dict1_,structure_type=3)##辞書をつくってそこにopt_1とopt_2でファイル名作成
        file_name2=make_file(monomer_name,params_dict2_,structure_type=3)
        file_name3=make_file(monomer_name,params_dict3_,structure_type=3)
        
        log_filepath1 = os.path.join(*[auto_dir,'gaussian',file_name1])
        if not(os.path.exists(log_filepath1)):#logファイルが生成される直前だとまずいので
            continue
        E_list1=get_E(log_filepath1)
        
        log_filepath2 = os.path.join(*[auto_dir,'gaussian',file_name2])
        if not(os.path.exists(log_filepath2)):#logファイルが生成される直前だとまずいので
            continue
        E_list2=get_E(log_filepath2)
        
        log_filepath3 = os.path.join(*[auto_dir,'gaussian',file_name3])
        if not(os.path.exists(log_filepath3)):#logファイルが生成される直前だとまずいので
            continue
        E_list3=get_E(log_filepath3)
        
        if (len(E_list1)!=1) or (len(E_list2)!=1) or (len(E_list3)!=1):##get Eの長さは計算した分子の数
            continue
        else:
            E1=float(E_list1[0]);E2=float(E_list2[0]);E3=float(E_list3[0])##8分子に向けてep1,ep2作成　ep1:b ep2:a
            E=E1+E2+E3
            df_E.loc[idx, ['E','E1','E2','E3','status']] = [E,E1,E2,E3,'Done']
            df_E.to_csv(auto_csv,index=False)
            break#2つ同時に計算終わったりしたらまずいので一個で切る
    
    df_qw_1 = df_E_1[df_E_1['status'] == 'qw'];df_qw_2 = df_E_2[df_E_2['status'] == 'qw'];df_qw_3 = df_E_3[df_E_3['status'] == 'qw']
    len_queue = len_prg_1 + len_prg_2 + len_prg_3
    len_qw_1 = len(df_qw_1);len_qw_2 = len(df_qw_2);len_qw_3 = len(df_qw_3)
    margin = max_nodes - len_queue

    df_inpr_1 = df_E_1.loc[df_E_1['status']=='InProgress'];df_inpr_2 = df_E_2.loc[df_E_2['status']=='InProgress'];df_inpr_3 = df_E_3.loc[df_E_3['status']=='InProgress']
    machine_counts_1 = df_inpr_1['machine_type'].value_counts().to_dict();machine_counts_1.setdefault(1, 0);machine_counts_1.setdefault(2, 0)
    machine_counts_2 = df_inpr_2['machine_type'].value_counts().to_dict();machine_counts_2.setdefault(1, 0);machine_counts_2.setdefault(2, 0)
    machine_counts_3 = df_inpr_3['machine_type'].value_counts().to_dict();machine_counts_3.setdefault(1, 0);machine_counts_3.setdefault(2, 0)
    num_machine2 = machine_counts_1.get(2, 0) + machine_counts_2.get(2, 0) + machine_counts_3.get(2, 0)

    if len_qw_1 > 0 and margin > 0:# 進行中ジョブのマシンタイプをカウント
        for index, row in df_qw_1.iterrows():
            if margin == 0:
                break
            params_dict = row[fixed_param_keys + opt_param_keys_1].to_dict()# パラメータの辞書を作成
            if num_machine2 >= maxnum_machine2:
                machine_type = 1             
            else:
                machine_type = 2# マシンタイプの決定
                num_machine2 += 1
            file_name = exec_gjf(auto_dir, monomer_name, {**params_dict}, machine_type, structure_type=1, isTest=isTest)# ジョブの実行 structure type
            len_queue += 1
            df_E_1.at[index, 'machine_type'] = machine_type
            df_E_1.at[index, 'status'] = 'InProgress'
            df_E_1.at[index, 'file_name'] = file_name
            margin -= 1
        df_E_1.to_csv(auto_csv_1, index=False)# データフレームをCSVに保存
    
    if len_qw_2 > 0 and margin > 0:# 進行中ジョブのマシンタイプをカウント
        for index, row in df_qw_2.iterrows():
            if margin == 0:
                break
            params_dict = row[fixed_param_keys + opt_param_keys_2].to_dict()# パラメータの辞書を作成
            if num_machine2 >= maxnum_machine2:
                machine_type = 1             
            else:
                machine_type = 2# マシンタイプの決定
                num_machine2 += 1
            
            file_name = exec_gjf(auto_dir, monomer_name, {**params_dict}, machine_type, structure_type=2,isTest=isTest)# ジョブの実行
            len_queue += 1
            df_E_2.at[index, 'machine_type'] = machine_type
            df_E_2.at[index, 'status'] = 'InProgress'
            df_E_2.at[index, 'file_name'] = file_name
            margin -= 1
        df_E_2.to_csv(auto_csv_2, index=False)# データフレームをCSVに保存
    
    if len_qw_3 > 0 and margin > 0:# 進行中ジョブのマシンタイプをカウント
        for index, row in df_qw_3.iterrows():
            if margin == 0:
                break
            params_dict = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2].to_dict()# パラメータの辞書を作成
            if num_machine2 >= maxnum_machine2:
                machine_type = 1             
            else:
                machine_type = 2# マシンタイプの決定
                num_machine2 += 1
            
            file_name = exec_gjf(auto_dir, monomer_name, {**params_dict}, machine_type, structure_type=3,isTest=isTest)# ジョブの実行
            len_queue += 1
            df_E_3.at[index, 'machine_type'] = machine_type
            df_E_3.at[index, 'status'] = 'InProgress'
            df_E_3.at[index, 'file_name'] = file_name
            margin -= 1
        df_E_3.to_csv(auto_csv_3, index=False)# データフレームをCSVに保存
    
    dict_matrix = get_params_dict(auto_dir,num_nodes)##更新分を流す x1~z2まで取得
    if len(dict_matrix)!=0:#終わりがまだ見えないなら
        for i in range(len(dict_matrix)):

            params_dict=dict_matrix[i]#print(params_dict)
            params_dict1 = {k: v for k, v in params_dict.items() if (k in fixed_param_keys) or (k in opt_param_keys_1)}
            params_dict2 = {k: v for k, v in params_dict.items() if (k in fixed_param_keys) or (k in opt_param_keys_2)}
            params_dict3 = params_dict
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):##ここで各点について1~3を実行しつつ余ったものをqwにぶち込む

                df_newline = pd.Series({**params_dict,'E':0.,'E1':0.,'E2':0.,'E3':0.,'status':'InProgress'})
                df_E=df_E.append(df_newline,ignore_index=True)
                df_E.to_csv(auto_csv,index=False)
                
                ## 1の実行　##
                df_done_1_ = filter_df(df_E_1, {**params_dict1,'status':'Done'});df_inpr_1_ = filter_df(df_E_1, {**params_dict1,'status':'InProgress'});df_qw_1_ = filter_df(df_E_1, {**params_dict1,'status':'qw'})
                if (len(df_done_1_)>=1) or(len(df_inpr_1_)>=1) or (len(df_qw_1_)>=1):
                    continue
                else:
                    isAvailable = len_queue < max_nodes
                    if isAvailable:
                        machine2IsFull = num_machine2 >= maxnum_machine2
                        if machine2IsFull:
                            machine_type = 1
                        else:
                            machine_type = 2;num_machine2 += 1
                        file_name = exec_gjf(auto_dir, monomer_name, {**params_dict1}, machine_type, structure_type=1,isTest=isTest);len_queue +=1
                        df_newline_1 = pd.Series({**params_dict1,'E1':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                        df_E_new_1=pd.concat([df_E_1,df_newline_1.to_frame().T],axis=0,ignore_index=True);df_E_new_1.to_csv(auto_csv_1,index=False)
                    else:
                        file_name = exec_gjf(auto_dir, monomer_name, {**params_dict1}, machine_type=1, structure_type=1,isTest=True)
                        df_newline_1 = pd.Series({**params_dict1,'E1':0.,'machine_type':1,'status':'qw','file_name':file_name})
                        df_E_new_1=pd.concat([df_E_1,df_newline_1.to_frame().T],axis=0,ignore_index=True);df_E_new_1.to_csv(auto_csv_1,index=False)

                ## 2の実行　##
                df_done_2_ = filter_df(df_E_2, {**params_dict2,'status':'Done'});df_inpr_2_ = filter_df(df_E_2, {**params_dict2,'status':'InProgress'});df_qw_2_ = filter_df(df_E_2, {**params_dict2,'status':'qw'})
                if (len(df_done_2_)>=1) or(len(df_inpr_2_)>=1) or (len(df_qw_2_)>=1):
                    continue
                else:
                    isAvailable = len_queue < max_nodes
                    if isAvailable:
                        machine2IsFull = num_machine2 >= maxnum_machine2
                        if machine2IsFull:
                            machine_type = 1
                        else:
                            machine_type = 2;num_machine2 += 1
                        file_name = exec_gjf(auto_dir, monomer_name, {**params_dict2}, machine_type, structure_type=2,isTest=isTest);len_queue +=1
                        df_newline_2 = pd.Series({**params_dict2,'E2':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                        df_E_new_2=pd.concat([df_E_2,df_newline_2.to_frame().T],axis=0,ignore_index=True);df_E_new_2.to_csv(auto_csv_2,index=False)
                    else:
                        file_name = exec_gjf(auto_dir, monomer_name, {**params_dict2}, machine_type=1, structure_type=2,isTest=True)
                        df_newline_2 = pd.Series({**params_dict2,'E3':0.,'machine_type':1,'status':'qw','file_name':file_name})
                        df_E_new_2=pd.concat([df_E_2,df_newline_2.to_frame().T],axis=0,ignore_index=True);df_E_new_2.to_csv(auto_csv_2,index=False)

                ## 3の実行　##
                df_done_3_ = filter_df(df_E_3, {**params_dict3,'status':'Done'});df_inpr_3_ = filter_df(df_E_3, {**params_dict3,'status':'InProgress'});df_qw_3_ = filter_df(df_E_3, {**params_dict3,'status':'qw'})
                if (len(df_done_3_)>=1) or (len(df_inpr_3_)>=1) or (len(df_qw_3_)>=1):
                    continue
                else:
                    isAvailable = len_queue < max_nodes
                    if isAvailable:
                        machine2IsFull = num_machine2 >= maxnum_machine2
                        if machine2IsFull:
                            machine_type = 1
                        else:
                            machine_type = 2;num_machine2 += 1
                        file_name = exec_gjf(auto_dir, monomer_name, {**params_dict3}, machine_type, structure_type=3,isTest=isTest);len_queue +=1
                        df_newline_3 = pd.Series({**params_dict3,'E3':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                        df_E_new_3=pd.concat([df_E_3,df_newline_3.to_frame().T],axis=0,ignore_index=True);df_E_new_3.to_csv(auto_csv_3,index=False)
                    else:
                        file_name = exec_gjf(auto_dir, monomer_name, {**params_dict3}, machine_type=1, structure_type=3,isTest=True)
                        df_newline_3 = pd.Series({**params_dict3,'E3':0.,'machine_type':1,'status':'qw','file_name':file_name})
                        df_E_new_3=pd.concat([df_E_3,df_newline_3.to_frame().T],axis=0,ignore_index=True);df_E_new_3.to_csv(auto_csv_3,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step1.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes):
    """
    前提:
        step1_init_params.csvとstep1.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    fixed_param_keys = ['theta'];opt_param_keys_1 = ['a'];opt_param_keys_2 = ['b']

    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        #print(1)
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys_1+opt_param_keys_2].to_dict()
            return [params_dict]
    dict_matrix=[]
    for index in df_init_params_inprogress.index:##こちら側はinit_params内のある業に関する探索が終わった際の新しい行での探索を開始するもの ###ここを改良すればよさそう
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys_1+opt_param_keys_2].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)
        if isDone:
            opt_params_dict={'a':np.round(opt_params_matrix[0][0],1),'b':np.round(opt_params_matrix[0][1],1)}
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:##もうこれ以上は新しい計算は進まない
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':##計算が始まっていないものがあったらこの時点で開始する　ここでダメでもまた直にlistenでgrt_params_dictまでいけば新しいのが始まる            
                opt_params_dict = get_values_from_df(df_init_params,index+1,opt_param_keys_1+opt_param_keys_2)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                dict_matrix.append({**fixed_params_dict,**opt_params_dict})
            else:
                continue

        else:
            for i in range(len(opt_params_matrix)):
                opt_params_dict={'a':np.round(opt_params_matrix[i][0],1),'b':np.round(opt_params_matrix[i][1],1)}
                df_inprogress = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'InProgress'})
                df_qw = filter_df(df_cur, {**fixed_params_dict,**opt_params_dict,'status':'qw'})
                if (len(df_inprogress)>=1) or (len(df_qw)>=1):
                    continue
                else:
                    d={**fixed_params_dict,**opt_params_dict}
                    dict_matrix.append(d)
                    #print(d)
    return dict_matrix
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    a_init_prev = init_params_dict['a'];b_init_prev = init_params_dict['b'];theta = init_params_dict['theta']
    while True:
        E_list=[];xyz_list=[]
        para_list=[]
        for a in [a_init_prev-0.1,a_init_prev,a_init_prev+0.1]:
            for b in [b_init_prev-0.1,b_init_prev,b_init_prev+0.1]:
                a = np.round(a,1);b = np.round(b,1)
                df_val_xyz = df_val[(df_val['a']==a)&(df_val['b']==b)&(df_val['theta']==theta)&(df_val['status']=='Done')]
                if len(df_val_xyz)==0:
                    para_list.append([a,b])
                    continue
                xyz_list.append([a,b]);E_list.append(df_val_xyz['E'].values[0])
        if len(para_list) != 0:
            print(para_list)
            return False,para_list
        a_init,b_init = xyz_list[np.argmin(np.array(E_list))]
        if a_init==a_init_prev and b_init==b_init_prev:
            return True,[[a_init,b_init]]
        else:
            a_init_prev=a_init;b_init_prev=b_init

def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def make_file(monomer_name,params_dict,structure_type):
    xyzfile_name = ''
    xyzfile_name += monomer_name
    for key,val in params_dict.items():
        if key in ['a','b','cx','cy','cz','theta']:
            val = np.round(val,2)
        elif key in ['A1','A2']:#,'theta']:
            val = int(val)
        xyzfile_name += '_{}={}'.format(key,val)
    return xyzfile_name + f'_{structure_type}.log'

def filter_df(df, dict_filter):
    for k, v in dict_filter.items():
        if type(v)==str:
            df=df[df[k]==v]
        else:
            df=df[df[k]==v]
    df_filtered=df
    return df_filtered

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes gaussian, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    parser.add_argument('--max-nodes',type=int,help='max nodes')
    ##maxnum-machine2 がない
    args = parser.parse_args()

    print("----main process----")
    main_process(args)
    print("----finish process----")
    