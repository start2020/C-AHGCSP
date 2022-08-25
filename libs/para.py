# coding: utf-8
import argparse
import numpy as np
import datetime

######################################  模型参数  #################################################
def ANN(parser):
    parser.add_argument('--ANN_Units', default="256,256", type=str)
    parser.add_argument('--ANN_Activations', default="relu,relu", type=str)

def CANN(parser):
    parser.add_argument('--ANN_Units', default="256,256", type=str)
    parser.add_argument('--ANN_Activations', default="relu,relu", type=str)

def GCN(parser):
    parser.add_argument('--GCN_Units', default="256,256", type=str)
    parser.add_argument('--GCN_Activations', default="relu,relu", type=str)
    parser.add_argument('--GCN_Ks', default="1,1", type=str)

def CGCN(parser):
    parser.add_argument('--GCN_Units', default="256,256", type=str)
    parser.add_argument('--GCN_Activations', default="relu,relu", type=str)
    parser.add_argument('--GCN_Ks', default="1,1", type=str)

def RNN(parser):
    parser.add_argument('--RNN_Units', default="256,256", type=str)
    parser.add_argument('--RNN_Type', default='RNN', type=str, help="RNN, GRU, LSTM")

def CRNN(parser):
    parser.add_argument('--RNN_Units', default="256,256", type=str)
    parser.add_argument('--RNN_Type', default='RNN', type=str, help="RNN, GRU, LSTM")

def ConvLSTM(parser):
    parser.add_argument('--Kernel', default="3,3", type=str)
    parser.add_argument('--Filters', default="8,8,1", type=str)

def CConvLSTM(parser):
    parser.add_argument('--Kernel', default="3,3", type=str)
    parser.add_argument('--Filters', default="8,8,1", type=str)

def GEML(parser):
    parser.add_argument('--GEML_GCN_Units', default="256,256", type=str)
    parser.add_argument('--GEML_GCN_Activations', default="relu,relu", type=str) # sigmoid
    parser.add_argument('--GEML_RNN_Units', default="256", type=str)
    parser.add_argument('--Ks', default=[None,None,None], type=list)
    parser.add_argument('--GEML_RNN_Type', default="GRU", type=str)
    parser.add_argument('--GEML_Weights', default=[0.5, 0.25, 0.25], type=list)

def CGEML(parser):
    parser.add_argument('--GEML_GCN_Units', default="256,256", type=str)
    parser.add_argument('--GEML_GCN_Activations', default="relu,relu", type=str) # sigmoid
    parser.add_argument('--GEML_RNN_Units', default="256", type=str)
    parser.add_argument('--Ks', default=[None,None,None], type=list)
    parser.add_argument('--GEML_RNN_Type', default="GRU", type=str)
    parser.add_argument('--GEML_Weights', default=[0.5, 0.25, 0.25], type=list)

def AHGCSP(parser):
    parser.add_argument('--Single', default="1", type=str)#"all" "complete" "Long" "GCN"  "LSTM" "Dym" "KL" "Geo"
    parser.add_argument('--ablation', default="all", type=str)#"all" "complete" "Long" "GCN"  "LSTM" "Dym" "KL" "Geo"
    parser.add_argument('--AHGCSP_GCN_Units', default="64, 64", type=str)
    parser.add_argument('--AHGCSP_GCN_Activations', default="relu,relu", type=str)
    parser.add_argument('--AHGCSP_GCN_Ks', default="1,1", type=str)
    parser.add_argument('--AHGCSP_RNN_Units', default="256,256", type=str)
    parser.add_argument('--AHGCSP_RNN_Type', default="LSTM", type=str)
    parser.add_argument('--AHGCSP_Dynamic_Units', default="64", type=str)
    parser.add_argument('--AHGCSP_Dynamic_Activations', default="relu", type=str)
    parser.add_argument('--AHGCSP_Fusion_Units', default="64", type=str)
    parser.add_argument('--AHGCSP_Fusion_Activations', default="relu", type=str)

def common(parser):
    parser.add_argument('--debug', default="0", type=str, help="run expriment in parallel")
    parser.add_argument('--Repeat', default = 1, type=int)
    parser.add_argument('--Base_Lr', type=float, default=0.01,help='initial learning rate')
    parser.add_argument('--Max_Epoch', type=int, default=1000, help='the max epoch to run')
    parser.add_argument('--Val_Loss', default=0, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--GPU', default="0", type=str)
    parser.add_argument('--Q', default=4, type=int)

def choose_model(args, parser):
    if args.Model == "ANN":
        ANN(parser)
    elif args.Model == "GCN":
        GCN(parser)
    elif args.Model == "RNN":
        RNN(parser)
    elif args.Model == "ConvLSTM":
        ConvLSTM(parser)
    elif args.Model == "GEML":
        GEML(parser)
    elif args.Model == "AHGCSP":
        AHGCSP(parser)
    elif args.Model == "CGCN":
        CGCN(parser)
    elif args.Model == "CANN":
        CANN(parser)
    elif args.Model == "CGEML":
        CGEML(parser)
    elif args.Model == "CRNN":
        CRNN(parser)
    elif args.Model == "CConvLSTM":
        CConvLSTM(parser)
    elif args.Model in ["HA","Ridge"]:
        parser.add_argument('--nonlinear', default="HA or Ridge", type=str)
    else:
        raise ValueError

def Parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', default="ANN")  # "AHGCSP","HA","GEML", "GCN", "CASCNN", "ConvLSTM", "GRU"
    # (1)可变参数 => 数据文件
    parser.add_argument('--Note', default='', type=str)
    parser.add_argument('--City', default='SH', type=str)
    parser.add_argument('--M', default="04", type=str, help="Month") # 15, 30
    common(parser)
    # (1)可变参数 => 模型调试
    args =  parser.parse_args()
    choose_model(args, parser)
    # ANN(parser)
    # RNN(parser)
    # GCN(parser)
    # ConvLSTM(parser)
    # GEML(parser)
    # AHGCSP(parser)
    args = parser.parse_args()
    parser.add_argument('--hyper_para', default=str(args), type=str, help="printable parameter")

    args = parser.parse_args()
    # Unchange
    if args.City == "SH":
        parser.add_argument('--N', default=133, type=int)
    elif args.City == "SZ":
        parser.add_argument('--N', default=118, type=int)
    elif args.City == "HZ":
        parser.add_argument('--N', default=80, type=int)
    else:
        print("No such Dataset!")

    # (2)可变参数 => 非模型调试
    parser.add_argument('--T', default=15, type=int, help="Time Granularity") # 15, 30
    parser.add_argument('--P', default = 8, type=int, help="Previous P steps")
    parser.add_argument('--D', default = 1, type=int, help="Previous D steps")
    parser.add_argument('--W', default = 1, type=int, help="Previous W steps")
    parser.add_argument('--Batch_Size', default=32, type=int)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--Null_Val', type=float, default=np.nan, help='value for missing data')
    parser.add_argument('--Train_Shuffle', type=int, default=0, help='train shuffle')

    # (2)可变参数 => 优化有关参数
    parser.add_argument('--MIN_LR', type=float, default=2.0e-06, help="min learning rate")
    parser.add_argument('--LR_Decay_Ratio', type=float, default=0.9, help="decay per time")
    parser.add_argument('--Steps', type=int, default=1, help="learning rate decay time")
    parser.add_argument('--Continue_Train', type=int, default=0, help='initial withou old model')
    parser.add_argument('--Patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--Opt', type=str, default='adam', help="the optimizer")
    parser.add_argument('--Max_Grad_Norm', type=float, default=5., help="the clip of grad value")
    parser.add_argument('--Loss_Type', type=int, default=1, help="1-OD, 2-OD/Inflow/Outflow")

    # (3)不可变参数
    args = parser.parse_args()
    parser.add_argument('--Dataset', default='{}OD{}_{}.npz'.format(args.N, args.T, args.M)) # OD15_01.npz
    parser.add_argument('--day_index', default="day_index_{}.txt".format(args.M)) # day_index_01.txt
    time = datetime.datetime.strftime(datetime.datetime.now(), "%m%d_%H%M")
    parser.add_argument('--Time', default = time, type=str)

    # 目录
    args = parser.parse_args()
    S1 = 'Data/Results/' + args.City +  '/'
    parser.add_argument('--Original_Path', default= 'Data/Original/'+args.City+'/', type=str)
    parser.add_argument('--Input_Path', default='Data/Inputs/'+args.City+'/', type=str)
    parser.add_argument('--Results_Path', default=S1, type=str)
    parser.add_argument('--Results_Loss_Path', default=S1+'Loss/', type=str)
    parser.add_argument('--Results_Models_Path', default=S1+'Models/', type=str)
    parser.add_argument('--Results_Logs_Path', default=S1+'Logs/', type=str)
    parser.add_argument('--Results_Metrics_Path', default=S1+'Metrics/', type=str)
    parser.add_argument('--Results_Preds_Save_Path', default=S1+'Preds/', type=str)
    parser.add_argument('--Results_Preds_Inputs_Path', default=S1+'Preds/Inputs/', type=str)

    # 文件
    args = parser.parse_args()
    Model = args.Model
    if args.Model == "AHGCSP":
        Model = Model + "_" + args.ablation
    S2 = 'T'+str(args.T)+'P'+str(args.P)+'M'+str(args.M)+"_"
    parser.add_argument('--Data_Save_Path', default= args.Input_Path+S2, type=str)
    parser.add_argument('--Data_Load_Path', default= args.Original_Path + args.Dataset, type=str)
    parser.add_argument('--Mean_Std_Path', default= args.Input_Path+S2+"Mean_Std.npz", type=str)
    parser.add_argument('--Metrics_Save_Path', default=args.Results_Metrics_Path+S2+Model+'.txt', type=str)
    parser.add_argument('--Loss_Save_Path', default=args.Results_Loss_Path+S2+Model+'_', type=str)
    parser.add_argument('--Model_Save_Path', default=args.Results_Models_Path+S2+Model+'_', type=str)
    parser.add_argument('--Test_Data_Save_Path', default= args.Results_Preds_Inputs_Path+S2, type=str)

    parser.add_argument('--Log_Inputs_Path', default=args.Results_Logs_Path + 'inputs.txt', type=str)
    parser.add_argument('--Log_Main_Path', default=args.Results_Logs_Path + 'exp_{}_{}_{}.txt'.format(Model, args.Time, args.debug), type=str)

    parser.add_argument('--GCN_A_Path', default= args.Input_Path + 'graph_connection.npz', type=str)
    parser.add_argument('--GEML_Geo_Path', default= args.Input_Path + 'graph_geo3.npz', type=str)
    parser.add_argument('--AHGCSP_Geo_Path', default= args.Input_Path + 'AHGCSP_Geo.npz', type=str)
    parser.add_argument('--AHGCSP_KL_Path', default= args.Input_Path + 'AHGCSP_KL.npz', type=str)
    parser.add_argument('--AHGCSP_S_Path', default= args.Input_Path + 'AHGCSP_S.npz', type=str)
    parser.add_argument('--AHGCSP_D_Path', default= args.Input_Path + 'AHGCSP_D.npz', type=str)
    parser.add_argument('--AHGCSP_W_Path', default= args.Input_Path + 'AHGCSP_W.npz', type=str)
    args = parser.parse_args()
    return args
