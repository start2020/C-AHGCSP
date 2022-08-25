import numpy as np
import os
from libs import utils
import datetime

################################  GEML
'''
功能：构建semantic graph (GEML)
      度向量+邻居矩阵，按行归一化，加入自环
输入：矩阵(...,N,N)
'''
def one_graph(A):
    N,N = A.shape
    Ms = np.zeros_like(A).astype(np.float32)
    D_In = np.sum(A, axis=-1)  # (N,)
    D_Out = np.sum(A, axis=-2)  # (N,)
    for i in range(N):
        for j in range(i,N):
            if i == j: continue
            if A[i, j] > 0.0 or A[j, i] > 0.0:
                Ms[j, i] = Ms[i, j] = D_In[j] + D_Out[j]
    for i in range(N):
        row_sum = np.sum(Ms[i,...])
        if row_sum==0.0:continue
        else:
            Ms[i, ...] = Ms[i,...]/row_sum
    return Ms

def semantic_graph_GEML(A):
    M, P, N, N = A.shape
    Ms = np.copy(A).astype(np.float32)
    for i in range(M):
        for j in range(P):
            Ms[i,j] = one_graph(Ms[i,j])
    return Ms

'''
3- 主模型装载数据
装载训练集/测试集/验证集，标准差/均值
'''
def load_data(args, data_dir, yes=True):
    print('loading data...')
    types = ['train','val','test']
    results = []
    for type in types:
        path = os.path.join(data_dir, '%s.npz'%(type))
        data = np.load(path)
        dict = []
        for j in range(len(data)):
            dict.append(data['arr_{}'.format(j)])
            print('%s=>%s shape: %s,type:%s' % (type, j, dict[j].shape,dict[j].dtype))
        results.append(dict)
    # 装载标准差和均值
    if yes:
        ms = np.load("ANN/mean_std_OD.npz")
        # mean, std = np.load(args.Mean_Std_Path)['mean'], np.load(args.Mean_Std_Path)['std']
        mean, std = ms['mean'], ms['std']
    else:
        mean, std = 0.0, 1.0 # 数据不标准化
    print('mean:{},std:{}'.format(mean, std))
    return results, mean, std

# 产生周特征
def week_transform(args):
    station_txt_file = os.path.join(args.Original_Path, args.day_index)
    day_list = []
    with open(station_txt_file, "r") as fin:
        for line in fin.readlines():
            line = line[:-1]  # 去掉\n
            day_list.append(line)
    dayofweek = np.array([int(datetime.datetime.strptime(day_list[i], '%Y-%m-%d').strftime("%w")) for i in
                 range(len(day_list))])
    print('week features shape:{}, dtype:{}'.format(dayofweek.shape, dayofweek.dtype))
    return dayofweek

# 产生 time attributes
def add_time(T1, N, dayofweek):
    Days = len(dayofweek)
    Dayoftime = np.tile(np.reshape(np.array([t for t in range(T1)]), (1, T1, 1, 1)),(Days, 1, N, 1)) #(Days,T1,N,1)
    Dayofweek = np.tile(np.reshape(np.array(dayofweek), (Days, 1, 1, 1)),(1, T1, N, 1)) # (Days,T1,N,1)
    Dayofyear = np.tile(np.reshape(np.arange(len(dayofweek)), (Days, 1, 1, 1)),(1, T1, N, 1)) # (Days,T1,N,1)
    output = np.concatenate([Dayoftime, Dayofweek, Dayofyear], axis=-1) # (Days,T1,N,3)
    return output

# 输入一个list，list中每个元素是一个数据集，形状是多维数组，按照第一个维度将数据集打乱
def shuffle(data):
    results = []
    sample_num = data[0].shape[0]
    per = list(np.random.RandomState(seed=42).permutation(sample_num)) # 固定seed
    #per = list(np.random.permutation(sample_num)) # 随机划分
    for i in range(len(data)):
        results.append(data[i][per,...])
    return results

# 划分batch data列表中每个元素的第一个维度进行batch划分，drop=1不够一个batch的丢弃掉， drop=0 够一个batch的保留
def batch_split(args, data, drop=True):
    sample_num = data[0].shape[0]
    results=[]
    for i in range(len(data)):
        remainder = sample_num % args.Batch_Size # [0, args.Batch_Size-1]
        if remainder == 0:
            t = data[i]
        else:
            if drop:
                t = data[i][:-remainder] # 去掉最后 remainder 个
            else:
                L = [-1 for i in range(args.Batch_Size - remainder)]
                t = np.concatenate([data[i], data[i][L]]) # 增加 args.Batch_Size - remainder 个
        batch_num = int(t.shape[0] / args.Batch_Size)
        t = np.stack(np.split(t, batch_num, axis=0), axis=0)
        results.append(t)
    return results

# 划分、保存 训练集/验证集/测试集
def data_split(args, data, data_names):
    batch_num = data[0].shape[0]
    train_num = round(args.train_ratio * batch_num)
    val_num = round(args.val_ratio * batch_num)
    for i in range(len(data)):
        d = []
        train = data[i][:train_num, ...]
        val = data[i][train_num:train_num+val_num, ...]
        test = data[i][train_num+val_num:, ...]
        d.extend([train, val, test])
        print('{}=>dtype:{}, train: {}\tval: {}\ttest: {}'.format(data_names[i],data[i].dtype, train.shape, val.shape, test.shape))
        data[i] = d
        if train.shape[2] != 0: #(M,B,D,N,N)
            path = args.Data_Save_Path + data_names[i] + '.npz'
            np.savez_compressed(path, train, val, test) # 保存

# 检查多维数组中是否存在nan,inf
def check_inf_nan(Ms):
    nan_num = np.sum(np.isnan(Ms).astype(np.float32))
    inf_num = np.sum(np.isinf(Ms).astype(np.float32))
    print("Number of nan",nan_num,"Number of inf",inf_num)

def generate_samples_analysis(args, Train=True):
    # 载入原始数据
    print("Loading Data!")
    start_time = datetime.datetime.now() # 开始记录时间
    OD = np.load(args.Data_Load_Path)['matrix'].astype(np.float32) #(D,T,N,N,T),是否转为浮点数？
    # D, T, N = 21, 30, 5
    # OD = np.random.randint(0,8,size=(D,T,N,N,T))
    print("Data shape:{}, Datatype:{}".format(OD.shape, OD.dtype))
    end_time_1 = datetime.datetime.now()
    print("Done! Loading Time:{}\n".format(end_time_1-start_time))

    print("Generate Samples")
    # 产生并保存好std,mean
    mean, std = np.mean(np.sum(OD, axis=-1)), np.std(np.sum(OD, axis=-1)) #(M,T,N,N)
    print("OtD => mean:{:.2f},std:{:.2f}".format(mean, std))
    np.savez_compressed(args.Mean_Std_Path, mean=mean, std=std)

    #dayofweek = week_transform(args)
    #Time = add_time(T1,N,dayofweek)  # (D,T1,N,T) # 时间特征
    Days, T1, N, N, T2 = OD.shape

    # 前P个时间段,前D天,前W周
    P = int(args.P)
    D = int(args.D)
    W = int(args.W)

    All_Data = [[], []]
    for j in range(Days):
        all_data = [[], []]
        if j - 7 * W < 0: continue
        weeks = [j - 7 * w for w in range(1, W+1)] #[j-7,...,j-7W]
        if j - D < 0: continue
        for i in range(T1):
            if i - P < 0: continue
            #t = Time[j,i,...]# (D,T1,N,T)=>(N,T) y对应的时间特征，仅在画图的时候需要
            x_P = np.sum(OD[j,i-P:i, ...,:],axis=-1) # 同一天，前P个时间段，全天候出闸，(P,N,N)
            x_P_before = np.sum(OD[j,i-P:i,...,:i], axis=-1)  # 同一天，前P个时间段，预测时间段前出闸，(P,N,N)
            all_data[0].append(x_P_before) # (M,P,N,N)
            all_data[1].append(x_P)  # (M,P,N,N)

        for a in range(len(all_data)):
            all_data[a] = np.stack(all_data[a], axis=0)  # (M,P,N,N)
            All_Data[a].append(all_data[a])

    for A in range(len(All_Data)):
        All_Data[A] = np.squeeze(np.stack(All_Data[A], axis=0))  # (D,M,P,N,N)
        if Train:
            All_Data[A] = np.squeeze(np.concatenate(np.split(All_Data[A], All_Data[A].shape[0], axis=0),axis=1)) #(DM,P,N,N)

    # 检查多维数组是否有nan,inf
    for i in range(len(All_Data)):
        check_inf_nan(All_Data[i])
    data_names = ['OtD_b_analysis', 'OtD_analysis']

    for l in range(len(data_names)):
        path = args.Data_Save_Path + data_names[l] + '.npz'
        np.savez_compressed(path, All_Data[l])
    print("Test Data Done!")

def generate_samples(args, Train=True):
    # 载入原始数据
    print("Loading Data!")
    start_time = datetime.datetime.now() # 开始记录时间
    OD = np.load(args.Data_Load_Path)['matrix'].astype(np.float32) #(D,T,N,N,T),是否转为浮点数？
    # D, T, N = 21, 30, 5
    # OD = np.random.randint(0,8,size=(D,T,N,N,T))
    print("Data shape:{}, Datatype:{}".format(OD.shape, OD.dtype))
    end_time_1 = datetime.datetime.now()
    print("Done! Loading Time:{}\n".format(end_time_1-start_time))

    print("Generate Samples")
    # 产生并保存好std,mean
    mean, std = np.mean(np.sum(OD, axis=-1)), np.std(np.sum(OD, axis=-1)) #(M,T,N,N)
    print("OtD => mean:{:.2f},std:{:.2f}".format(mean, std))
    np.savez_compressed(args.Mean_Std_Path, mean=mean, std=std)

    #dayofweek = week_transform(args)
    #Time = add_time(T1,N,dayofweek)  # (D,T1,N,T) # 时间特征
    Days, T1, N, N, T2 = OD.shape

    # 前P个时间段,前D天,前W周
    P = int(args.P)
    D = int(args.D)
    W = int(args.W)

    All_Data = [[], [], [], [], [], [], [],[]]
    for j in range(Days):
        all_data = [[], [], [], [], [], [], [],[]]
        if j - 7 * W < 0: continue
        weeks = [j - 7 * w for w in range(1, W+1)] #[j-7,...,j-7W]
        if j - D < 0: continue
        for i in range(T1):
            if i - P < 0: continue
            y = np.sum(OD[j,i,...,:], axis=-1)  # 第j天第i个时间段，(D,T1,N,N)=>(N,N)
            #t = Time[j,i,...]# (D,T1,N,T)=>(N,T) y对应的时间特征，仅在画图的时候需要
            x_D = np.sum(OD[j-D:j,i,...,:], axis=-1) # 前D天第i个时间段，(D,N,N)
            x_W = np.sum(OD[weeks,i-P:i+1,...,:], axis=-1)# 前W周同一天第i个时间段(W,P+1,N,N)
            x_P = np.sum(OD[j,i-P:i, ...,:],axis=-1) # 同一天，前P个时间段，全天候出闸，(P,N,N)
            x_P_before = np.sum(OD[j,i-P:i,...,:i], axis=-1)  # 同一天，前P个时间段，预测时间段前出闸，(P,N,N)
            gs = []
            for p in range(P):
                gs.append(one_graph(x_P_before[p,:]))
            gs = np.stack(gs, axis=0) #(P,N,N) GEML:产生上下文图

            x_P_W = np.sum(OD[weeks,i-P:i, ...,:],axis=-1) # 前一周，前P个时间段，全天候出闸，(P,N,N)
            x_P_W_a = np.sum(OD[weeks,i-P:i, ...,i:],axis=-1) # 前一周，前P个时间段，全天候出闸，(P,N,N)
            x_P_W[x_P_W==0] = 1 # 0的位置用1代替，防止除零错误
            x_P_W_a_r = x_P_W_a / x_P_W

            xt_P = np.sum(OD[j,:, ...,i-P:i],axis=0) #(D,T,N,N,T)=>(T,N,N,T)=>(N,N,P)
            xt_P =np.transpose(xt_P, (2,1,0) ) # 同一天，前P个时间段，全天候入闸，(N_o,N_d,P)=>(P,N_d,N_o)
            all_data[0].append(y)  # (M,N,N)
            all_data[1].append(x_P_before) # (M,P,N,N)
            all_data[2].append(x_P)  # (M,P,N,N)
            all_data[3].append(x_D)  # (M,D,N,N)
            all_data[4].append(x_W)  # (M,W,P+1,N,N)
            all_data[5].append(xt_P) # (M,P,N,N)
            all_data[6].append(x_P_W_a_r) # (M,P,N,N)
            all_data[7].append(gs) # (M,P,N,N)

        for a in range(len(all_data)):
            all_data[a] = np.stack(all_data[a], axis=0)  # (M,P,N,N)
            All_Data[a].append(all_data[a])

    for A in range(len(All_Data)):
        All_Data[A] = np.squeeze(np.stack(All_Data[A], axis=0))  # (D,M,P,N,N)
        if Train:
            All_Data[A] = np.squeeze(np.concatenate(np.split(All_Data[A], All_Data[A].shape[0], axis=0),axis=1)) #(DM,P,N,N)

    # 标准化
    L = [1, 2, 3, 4, 5]
    for i in range(len(L)):
        All_Data[L[i]] = (All_Data[L[i]] - mean)/std #(M,P,N,N)

    # graph_sem = semantic_graph_GEML(All_Data[1])  # (M,P,N,N)
    # All_Data.append(graph_sem)

    Inflow = np.sum(All_Data[2], axis=-1) #((D-7)M, P, N_o)  【输入 CASCNN、AHGCSP补全】
    Finished_Inflow = np.sum(All_Data[1], axis=-1) #((D-7)M, P, N_o,)
    Delayed_Inflow = Inflow - Finished_Inflow #((D-7)M, P, N_o,) 【输入 AHGCSP补全】
    del Finished_Inflow
    Outflow = np.sum(All_Data[5], axis=-1) # ((D-7)M, P, N_d, N_o) => ((D-7)M, P, N_d) 【输入 CASCNN】
    All_Data.extend([Inflow, Outflow, Delayed_Inflow])

    # 检查多维数组是否有nan,inf
    for i in range(len(All_Data)):
        check_inf_nan(All_Data[i])
    data_names = ['labels', 'OtD_b', 'OtD', 'labels_D', 'labels_W', 'ODt', 'OtD_ar', 'graph_sem', 'Inflow', 'Outflow',
                  'Delayed_Inflow']
    if Train:
        # 打乱数据
        All_Data = shuffle(All_Data)
        # 划分batch
        results = batch_split(args, All_Data, drop=True)
        # 划分、保存 训练集/验证集/测试集
        data_split(args, results, data_names)
        print("Train Data Done!")

    else:
        for l in range(len(data_names)):
            path = args.Test_Data_Save_Path + data_names[l] + '.npz'
            np.savez_compressed(path, All_Data[l])
            print(data_names[l], All_Data[l].shape)
        print("Test Data Done!")

'''
HA/GCN/GRU/ConvLSTM: 当天前p个时刻的不全OtD矩阵 => OtD_b
CASCNN: 当前天p个时刻的Inflow和Outflow向量 + 前M天同一个时刻的OtD矩阵 => Outflow(ODt_P) + Inflow(OtD_P) + ODt_M
GEML: 当天前p个时刻的不全OtD矩阵 + 预测的Inflow/Outflow => OtD_b + 目标矩阵产生就可以
AHGCSP:  当天前p个时刻的不全OtD矩阵 + 当天前p个时刻的ODt矩阵 + 目标的前一周的同一个时刻 + 当前天p个时刻的Inflow +  ST=>
=> OtD_b + (ODt_P) + OtD_W

All Models: 需要同样的预测目标 => Label
'''
