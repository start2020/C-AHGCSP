import os,sys
curPath = os.path.abspath(os.path.dirname(__file__)) # 当前上级目录，如E:\RUN-NEW
rootPath = os.path.split(curPath)[0] # 当前最上级目录，如 E:\
sys.path.append(rootPath)
import argparse
from libs import para, utils, data_common

if __name__ == "__main__":
    # 引入所需参数
    args = para.Parameter()
    print(args)
    utils.create_paths(args) # 自动创建所有路径
    # 打开日志文件：a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加

    f = open(args.Log_Inputs_Path, "a+")
    # 控制台所有信息输入日志文件
    # sys.stdout = f
    # sys.stderr = f
    # 产生并保存样本
    data_common.generate_samples(args,Train=False)
    #data_common.generate_samples_analysis(args,Train=True)
    # 关闭日志文件
    f.close()