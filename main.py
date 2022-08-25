# import os,sys
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import para, main_common, utils
import sys

if __name__ == '__main__':
    args = para.Parameter()
    utils.create_paths(args) # 自动创建所有路径
    f = open(args.Log_Main_Path, 'a+')
    print("############################################################")
    print("Begin!")
    print("debug", args.debug)
    # sys.stdout = f
    # sys.stderr = f
    print("############################################################")
    print("Begin!")
    print(args)  # 打印所有参数
    #main_common.experiment(args)
    main_common.test(args)
    # try:
    #     main_common.experiment(args)
    #     main_common.test(args)
    # except Exception as err:
    #     title = "Error!"
    #     content = str(err) + '\n' + str(args.hyper_para)
    #     utils.send_notice(title, content) # content f"训练正确率:55%\n测试正确率:96.5%"
    #     print(err)
    # print("End!" + '\n')
    # f.close()