import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import datetime
import requests

def send_notice(title, content):
    token = "15fe03f24be3487faa7026f09896ebd3"
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=txt"
    response = requests.request("GET", url)
    #print(response.text)


# 给出一个路径（单级目录或多级别目录），若它不存在，则创建；若存在，则跳过
def create_path(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def create_paths(args):
    # 自动创建有关文件夹:
    # Inputs/City/所有的模型的输入数据 => args.Input_Path
    # Results/City/Figure(Logs)(Models)(Metrics) => Results_Figures/Models/Logs_Path
    print(args.Input_Path)
    dirs = [args.Input_Path, args.Results_Loss_Path, args.Results_Models_Path, args.Results_Logs_Path,
            args.Results_Metrics_Path,args.Results_Preds_Inputs_Path]
    create_path(dirs)

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def print_parameters(log):
    parameters = 0
    for variable in tf.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    log_string(log, 'trainable parameters: {:,}'.format(parameters))




'''
功能：产生输入模式的字符串，如 P0D3W0
'''
def input_name(args):
    num = len(args.input_types)
    name = ''
    for i in range(num):
        name += args.input_types[i] + str(args.input_steps[i])
    return name

'''
数据集产生序列后，存放有关的数据/日志/模型
'''
def create_dir_PDW(args):
    dir_name = input_name(args)  # 文件夹名，按输入模式命名
    PDW_dir = os.path.join(args.dataset_dir, dir_name) # 总路径
    log_dir = os.path.join(PDW_dir, 'log')  # args.dataset_dir/PDW/log，日志路径
    save_dir = os.path.join(PDW_dir, 'save', args.model)  # args.dataset_dir/PDW/save/model，模型保存路径
    data_dir = os.path.join(PDW_dir, 'data', args.model)  # args.dataset_dir/PDW/data/model, 数据保存路径
    create_path(save_dir)
    create_path(log_dir)
    create_path(data_dir)
    return log_dir, save_dir, data_dir, dir_name

'''
打开日志文件（追加模式），提示开始，打印参数
'''
def create_log(args, type):
    dir_name = input_name(args)  # 文件夹名，按输入模式命名
    log_file = os.path.join(args.dataset_dir, dir_name, 'log', args.model+'_'+type)  # args.dataset_dir/PDW/log/ANN_data_log，日志路径
    log = open(log_file, 'a') # append the log
    return log

def path(args,data_mode):
    dir_name = input_name(args) # 文件夹名，以输入模式命名
    data_path = os.path.join(args.dataset_dir, args.model, data_mode, dir_name) # 路径
    create_path(data_path) # 创建
    log_file = os.path.join(data_path, 'data_log')
    log = open(log_file, 'a') # append the log
    return dir_name, data_path, log


# mat; 矩阵(D,T,N,N),nd.array类型,元素数据类型是int or float32
# start_val: 区间的起始值,区间是闭区间[start,end]
# end_val: 区间的结束值
# 返回值: 这个区间在整个矩阵中占得比例

def get_pro(mat, start_val, end_val):
    mat = np.around(mat)
    days, T, N, _ = mat.shape
    sum = 0
    for i in range(start_val, end_val+1):
        sum += np.sum(mat == i)
    total = days*T*N*N
    proportion = sum/total
    return proportion

# 检查多维数组中是否存在nan,inf
def check_inf_nan(Ms):
    nan_num = np.sum(np.isnan(Ms).astype(np.float32))
    inf_num = np.sum(np.isinf(Ms).astype(np.float32))
    print("Number of nan",nan_num,"Number of inf",inf_num)