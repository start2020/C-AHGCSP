# conding=utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import time
from libs import metrics, para
import datetime
import numpy as np
import joblib
from sklearn import linear_model

def choose_model(args):
    Data_Names = ['labels','OtD_b']
    Checkpoint_File = ""
    if args.Model == "HA":
        import libs.HA as model
    elif args.Model == "Ridge":
        import libs.Ridge as model
    elif args.Model == "ANN":
        import libs.ANN as model
        #Checkpoint_File = 'T15P8M05_ANN_0629_0709_6_checkpoint'
        Checkpoint_File = 'T15P8M04_ANN_0629_0344_3_checkpoint'
    elif args.Model == "GCN":
        import libs.GCN as model
        #Checkpoint_File = 'T15P8M05_GCN_0629_0440_7_checkpoint'
        Checkpoint_File = 'T15P8M04_GCN_0629_0416_3_checkpoint'
    elif args.Model == "RNN":
        import libs.RNN as model
        #Checkpoint_File = 'T15P8M05_RNN_0629_0722_0_checkpoint'
        Checkpoint_File = 'T15P8M04_RNN_0629_0356_1_checkpoint'
    elif args.Model == "ConvLSTM":
        import libs.ConvLSTM as model
        #Checkpoint_File = 'T15P8M05_ConvLSTM_0629_0403_3_checkpoint'
        Checkpoint_File = "T15P8M04_ConvLSTM_0629_0347_7_checkpoint"
    elif args.Model == "GEML":
        import libs.GEML as model
        Data_Names = ['labels', 'OtD_b','graph_sem']
        #Checkpoint_File = 'T15P8M05_GEML_0630_1005_6_checkpoint'
        Checkpoint_File = "T15P8M04_GEML_0630_0910_0_checkpoint"
    elif args.Model == "AHGCSP":
        import libs.AHGCSP as model
        Data_Names = ['labels', 'OtD_b','ODt', 'labels_W', 'OtD_ar', 'Delayed_Inflow','OtD']
        #Checkpoint_File = "T15P8M05_AHGCSP_all_0801_0633_3_0checkpoint"
        Checkpoint_File = "T15P8M04_AHGCSP_all_0704_0840_8_0checkpoint"
    elif args.Model == "CGCN":
        import libs.CGCN as model
        Data_Names = ['labels', 'OtD_b','ODt', 'labels_W', 'OtD_ar', 'Delayed_Inflow','OtD']
    elif args.Model == "CANN":
        import libs.CANN as model
        Data_Names = ['labels', 'OtD_b','ODt', 'labels_W', 'OtD_ar', 'Delayed_Inflow','OtD']
    elif args.Model == "CGEML":
        import libs.CGEML as model
        Data_Names = ['labels','graph_sem', 'OtD_b','ODt', 'labels_W', 'OtD_ar', 'Delayed_Inflow','OtD']
    elif args.Model == "CRNN":
        import libs.CRNN as model
        Data_Names = ['labels', 'OtD_b','ODt', 'labels_W', 'OtD_ar', 'Delayed_Inflow','OtD']
    elif args.Model == "CConvLSTM":
        import libs.CConvLSTM as model
        Data_Names = ['labels', 'OtD_b','ODt', 'labels_W', 'OtD_ar', 'Delayed_Inflow','OtD']
    else:
        raise ValueError
    return model, Data_Names, Checkpoint_File


def test(args):
    # 选择模型
    model, Data_Names, Checkpoint_File = choose_model(args)
    # 数据装载(D,M,P,N,N),(D,M,N,N)
    # 载入均值和标准差
    mean, std = np.load(args.Mean_Std_Path)['mean'], np.load(args.Mean_Std_Path)['std']
    print("OtD => mean:{:.2f},std:{:.2f}".format(mean, std))
    # 载入placeholder的数据
    Data = []
    for i in range(len(Data_Names)):
        path = args.Test_Data_Save_Path + Data_Names[i] + '.npz'
        data = np.load(path)['arr_0']
        #print(f"{Data_Names[i]}:{data.shape}")
        Data.append(data)
    print('Data Loaded Finish...')
    print(args.Model)
    if args.Model == "HA":
        Samples = Data[1] #(D,M,P,N,N)
        Preds = np.mean(Samples, axis=2) #(D,M,N,N)
        Preds = Preds * std + mean
        print(Preds.shape)
        preds_save = args.Results_Preds_Save_Path + 'HA_preds.npz'
    elif args.Model == "Ridge":
        #Model_File = args.Results_Models_Path + "T15P8M05_Ridge_0811_1729_0_0ridge.pkl"
        Model_File = args.Results_Models_Path + "T15P8M04_Ridge_0812_0915_0_0ridge.pkl"
        print(f"Model_File:{Model_File}")
        clf = joblib.load(Model_File)
        # 数据变形:X=(M,B,T,N,N)=>(M,B,N,N,T)=>(M*B*N*N,T),Y=(M,B,N,N)=>(M*B*N*N,1)
        Samples = np.reshape(np.transpose(Data[1], (0, 1, 3, 4, 2)), (-1, args.P)) * std + mean
        # linear_model.RidgeClassifier
        Preds = clf.predict(Samples)
        shape = Data[0].shape
        print(f"shape:{shape}")
        Preds = np.reshape(Preds, newshape=shape)
        print(f"Preds.shape{Preds.shape}")
        preds_save = args.Results_Preds_Save_Path + 'Ridge_preds.npz'
        print(f"preds_save{preds_save}")
    else:
        # 配置GPU
        sess = GPU(args)
        # 恢复模型
        # Model_Path: 去到这里找，找 checkpoint；
        print(args.Results_Models_Path, Checkpoint_File)
        ckpt = tf.train.get_checkpoint_state(args.Results_Models_Path, latest_filename=Checkpoint_File)
        print("ckpt", ckpt) # ckpt model_checkpoint_path: "Data/Results/HZ/Models/T15P8M01_ANN_0628_1811_0_model-21"
                             #all_model_checkpoint_paths: "Data/Results/HZ/Models/T15P8M01_ANN_0628_1811_0_model-21"
        print("model_checkpoint_path", ckpt.model_checkpoint_path) # Data/Results/HZ/Models/T15P8M01_ANN_0628_1811_0_model-21
        preds_save = args.Results_Preds_Save_Path + ckpt.model_checkpoint_path.split('/')[-1] + '_preds.npz'
        label_save = args.Results_Preds_Save_Path + ckpt.model_checkpoint_path.split('/')[-1] + '_labels.npz'
        mae_save = args.Results_Preds_Save_Path + ckpt.model_checkpoint_path.split('/')[-1] + '_mae.npz'

        if ckpt and ckpt.model_checkpoint_path:
            saver2 = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
            saver2.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError

        graph = tf.get_default_graph()  # 获取当前默认计算图
        if args.Model in ["ANN", "RNN", "GCN", "ConvLSTM",]:
            samples = graph.get_tensor_by_name("samples:0")
            placeholders = [samples]
        if args.Model in ["GEML"]:
            samples = graph.get_tensor_by_name("samples:0")
            graph_sem = graph.get_tensor_by_name("graph_sem:0")
            placeholders = [samples, graph_sem]
        if args.Model in ["AHGCSP"]:
            OtD_b = graph.get_tensor_by_name("samples-1:0")
            OtD_all = graph.get_tensor_by_name("samples-2:0")
            ODt = graph.get_tensor_by_name("samples-2_1:0")
            labels_W = graph.get_tensor_by_name("samples-3:0")
            OtD_ar = graph.get_tensor_by_name("samples-4:0")
            Delayed_Inflow = graph.get_tensor_by_name("samples-5:0")
            placeholders = [OtD_b, ODt, labels_W, OtD_ar, Delayed_Inflow, OtD_all]
        preds = graph.get_tensor_by_name("preds:0")

        Days = Data[0].shape[0]
        Preds = []
        for d in range(Days):
            # 输入数据
            Feed_Dict = {}
            for f in range(1, len(Data)):
                Feed_Dict[placeholders[f-1]] = Data[f][d,...] # 第f个输入的第d天的数据，(M,P,N,N)
            # 预测结果
            output = sess.run(preds, feed_dict=Feed_Dict)
            #print("output:{},{}".format(output.shape, type(output)))
            Preds.append(output)
        Preds = np.stack(Preds, axis=0) #(D,M,N,N)

    Preds = np.round(Preds).astype(np.float32)
    np.savez_compressed(preds_save, Preds)

    # 计算numpy - array
    Labels = Data[0]
    # if args.Model == "Ridge":
    #     Labels = np.reshape(Data[0], (-1, 1))
    #     print(f"Labels.shape{Labels.shape}")
    #np.savez_compressed(label_save, Labels)

    mae, rmse, wmape, smape = metrics.calculate_metrics(Preds, Labels, null_val=args.Null_Val)
    message = "Test=> MAE:{:.4f} RMSE:{:.4f} WMAPE:{:.4f} SMAPE:{:.4f}".format(mae, rmse, wmape, smape)
    message_res = "MAE\t%.4f\tRMSE\t%.4f\tWMAPE\t%.4f\tSMAPE\t%.4f\ttest" % (mae, rmse, wmape, smape)
    print(message)
    write_metrics(args, message_res)
    mae = np.abs(np.subtract(Preds, Labels)) #(D,M,N,N)
    #np.savez_compressed(mae_save, mae)


def test_old(args):
    # 选择模型
    model, Data_Names, Checkpoint_File = choose_model(args)
    # 数据装载(D,M,P,N,N),(D,M,N,N)
    # 载入均值和标准差
    mean, std = np.load(args.Mean_Std_Path)['mean'], np.load(args.Mean_Std_Path)['std']
    print("OtD => mean:{:.2f},std:{:.2f}".format(mean, std))
    # 载入placeholder的数据
    data = [[], [], []]  # 训练集,验证集, 测试集
    for i in range(len(Data_Names)):
        path = args.Data_Save_Path + Data_Names[i] + '.npz'
        Data = np.load(path)
        for j in range(3):
            data[j].append(Data['arr_{}'.format(j)].astype(np.float32))
        print("{}".format(Data_Names[i]), data[0][i].shape, data[0][i].dtype, data[1][i].shape, data[1][i].dtype,
              data[2][i].shape, data[2][i].dtype)
    print('Data Loaded Finish...')
    Data = data[2] # 0是label，剩余是其他的

    # 配置GPU
    sess = GPU(args)
    # 恢复模型
    # Model_Path: 去到这里找，找 checkpoint；
    print(args.Results_Models_Path, Checkpoint_File)
    ckpt = tf.train.get_checkpoint_state(args.Results_Models_Path, latest_filename=Checkpoint_File)
    print("ckpt", ckpt) # ckpt model_checkpoint_path: "Data/Results/HZ/Models/T15P8M01_ANN_0628_1811_0_model-21"
                         #all_model_checkpoint_paths: "Data/Results/HZ/Models/T15P8M01_ANN_0628_1811_0_model-21"
    print("model_checkpoint_path", ckpt.model_checkpoint_path) # Data/Results/HZ/Models/T15P8M01_ANN_0628_1811_0_model-21
    preds_save = args.Results_Preds_Save_Path + ckpt.model_checkpoint_path.split('/')[-1] + '_preds.npz'
    label_save = args.Results_Preds_Save_Path + ckpt.model_checkpoint_path.split('/')[-1] + '_labels.npz'
    mae_save = args.Results_Preds_Save_Path + ckpt.model_checkpoint_path.split('/')[-1] + '_mae.npz'

    if ckpt and ckpt.model_checkpoint_path:
        saver2 = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        saver2.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError

    graph = tf.get_default_graph()  # 获取当前默认计算图
    if args.Model in ["ANN", "RNN", "GCN", "ConvLSTM",]:
        samples = graph.get_tensor_by_name("samples:0")
        placeholders = [samples]
    if args.Model in ["AHGCSP"]:
        OtD_b = graph.get_tensor_by_name("samples-1:0")
        OtD_all = graph.get_tensor_by_name("samples-2:0")
        ODt = graph.get_tensor_by_name("samples-2_1:0")
        labels_W = graph.get_tensor_by_name("samples-3:0")
        OtD_ar = graph.get_tensor_by_name("samples-4:0")
        Delayed_Inflow = graph.get_tensor_by_name("samples-5:0")
        placeholders = [OtD_b, ODt, labels_W, OtD_ar, Delayed_Inflow, OtD_all]
    preds = graph.get_tensor_by_name("preds:0")

    Days = Data[0].shape[0]
    print(Data[0].shape, Data[1].shape)
    Preds = []

    for d in range(Days):
        # 输入数据
        Feed_Dict = {}
        for f in range(1, len(Data)):
            Feed_Dict[placeholders[f-1]] = Data[f][d,...] # 第f个输入的第d天的数据，(M,P,N,N)
        # 预测结果
        output = sess.run(preds, feed_dict=Feed_Dict)
        print("output:{},{}".format(output.shape, type(output)))
        Preds.append(output)
    Preds = np.stack(Preds, axis=0) #(D,M,N,N)
    Preds = np.round(Preds).astype(np.float32)
    np.savez_compressed(preds_save, Preds)

    # 计算numpy - array
    Labels = Data[0]
    np.savez_compressed(label_save, Labels)

    mae, rmse, wmape, smape = metrics.calculate_metrics(Preds, Labels, null_val=args.Null_Val)
    message = "Test=> MAE:{:.4f} RMSE:{:.4f} WMAPE:{:.4f} SMAPE:{:.4f}".format(mae, rmse, wmape, smape)
    message_res = "MAE\t%.4f\tRMSE\t%.4f\tWMAPE\t%.4f\tSMAPE\t%.4f\ttest" % (mae, rmse, wmape, smape)
    print(message)
    write_metrics(args, message_res)
    mae = np.abs(np.subtract(Preds, Labels)) #(D,M,N,N)
    np.savez_compressed(mae_save, mae)

def experiment(args):
    model, Data_Names, Checkpoint_File = choose_model(args)
    # 载入均值和标准差
    mean, std = np.load(args.Mean_Std_Path)['mean'], np.load(args.Mean_Std_Path)['std']
    print("OtD => mean:{:.2f},std:{:.2f}".format(mean, std))
    # 载入placeholder的数据
    data = [[], [], []]  # 训练集,验证集, 测试集
    for i in range(len(Data_Names)):
        path = args.Data_Save_Path + Data_Names[i] + '.npz'
        Data = np.load(path)
        for j in range(3):
            data[j].append(Data['arr_{}'.format(j)].astype(np.float32))
        print("{}".format(Data_Names[i]), data[0][i].shape, data[0][i].dtype, data[1][i].shape, data[1][i].dtype,
              data[2][i].shape, data[2][i].dtype)
    print('Data Loaded Finish...')

    for r in range(args.Repeat):
        print(args.Repeat)
        # data_dir = 'ANN/'
        # data, mean, std = data_common.load_data(args, data_dir, yes=True)
        # F_in, F_out = data[0][1].shape[-1], args.N
        if args.Model in ["HA","Ridge"]:
            model.Model(args, r, mean, std, data)
            continue
        else:
            outputs, placeholders = model.Model(args, mean, std)
            labels = placeholders[0]
            preds = outputs[0]
            if args.Model == "AHGCSP" and args.Single == "0":
                inputs_preds= outputs[1]
                inputs = placeholders[-1]
                inputs_preds = tf.identity(inputs_preds, name="inputs_preds")
        preds = tf.identity(preds, name='preds')

        # 编译模型
        print('compiling model...')
        loss = metrics.masked_mse_tf(preds, labels, null_val=args.Null_Val)  # 损失

        if args.Model == "AHGCSP"and args.Single == "0":
            loss_inputs = metrics.masked_mse_tf(inputs_preds, inputs, null_val=args.Null_Val)  # 损失
            loss += loss_inputs
            print("double target!")

        lr, new_lr, lr_update, train_op = optimization(args, loss)  # 优化
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # 保存模型
        # 总的可训练参数
        TP = print_parameters()
        print(TP)

        # 配置GPU
        sess = GPU(args)

        # 初始化模型
        val_loss_min, WMAPE = np.inf, np.inf
        Epoch, wait, step, epoch = 0, 0, 0, 0
        Message, Metrics = '', ''
        save_loss = [[], [], []]
        sess.run(tf.global_variables_initializer())
        print("initializer successfully")
        
        print('**** training model ****')
        sign = '{}_{}_{}'.format(args.Time, r, args.debug)
        Model_File = args.Model_Save_Path + sign + 'model'
        Checkpoint_File = (args.Model_Save_Path + sign + 'checkpoint').split('/')[-1]
        Val_Loss_File = args.Loss_Save_Path + sign + '_val_loss_min.npz'
        Losses_File = args.Loss_Save_Path + sign + '_losses.npz' # 保存好损失,跟log相对应

        print("Model_File",Model_File)
        print("Checkpoint_File", Checkpoint_File)
        print("args.Results_Models_Path", args.Results_Models_Path)

        while (epoch < args.Max_Epoch):
            # 降低学习率
            if wait >= args.Patience:
                val_loss_min, epoch = restore(sess, saver, args.Results_Models_Path, Val_Loss_File, Checkpoint_File)
                step += 1
                wait = 0
                New_Lr = max(args.MIN_LR, args.Base_Lr * (args.LR_Decay_Ratio ** step))
                sess.run(lr_update, feed_dict={new_lr: New_Lr})
                # 删除多余的loss
                if epoch > args.Patience:
                    for k in range(len(save_loss)):
                        save_loss[k] = save_loss[k][:-args.Patience]
                if step > args.Steps:
                    print('early stop at epoch: %04d' % (epoch))
                    break

            # 打印当前时间/训练轮数/lr
            print('%s | epoch: %04d/%d, lr: %.4f' %
                             (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.Max_Epoch, sess.run(lr)))

            # 计算训练集/验证集/测试集损失
            types = ['train', 'val', 'test']
            results = []
            for i in range(len(types)):
                feed_dicts = get_feed_dicts(data[i], placeholders, args.Train_Shuffle)
                result, message, WMAPE = caculation(args,  sess, feed_dicts, labels, preds, train_op, loss,type=types[i])
                if i == 2:  # 获取测试集的性能指标
                    message_cur = message
                results.append(result)
            message = "loss=> train:%.4f val:%.4f test:%.4f time=> train:%.1f val:%.1f test:%.1f" % (
            results[0][0], results[1][0], results[2][0], results[0][1], results[1][1], results[2][1])
            print(message)

            # 存储损失
            save_loss[0].append(results[0][0])
            save_loss[1].append(results[1][0])
            save_loss[2].append(results[2][0])

            # 更新最小损失
            if args.Val_Loss == 1:
                val_loss = results[1][0]
            else:
                val_loss = WMAPE # test_loss 没用 val_loss = results[2][0]

            if val_loss < val_loss_min:
                wait = 0
                val_loss_min = val_loss
                saver.save(sess, Model_File, epoch, Checkpoint_File)
                Metrics = message_cur
                np.savez(Val_Loss_File, loss=val_loss_min)
                print("save %02d" % epoch)
            else:
                wait += 1
            epoch += 1

        # 写入log
        write_metrics(args, Metrics)
        # 存储好损失
        np.savez_compressed(Losses_File, np.array(save_loss))
        print(Message)
        sess.close()
        tf.reset_default_graph()

# Write the Metrics
def write_metrics(args, Metrics):
    # 可训练参数
    TP = print_parameters()
    # 评估指标
    S = Metrics + "\t{:<13} ".format(args.Time) + "\t{:<13} ".format(TP) + "\t{:<13} ".format(str(args.hyper_para)) + '\n'
    File_Name = args.Metrics_Save_Path
    if args.Model == "AHGCSP":
        File_Name = File_Name[:-4] + '_' + args.ablation + '.txt'
    f = open(File_Name, mode = 'a+')
    f.writelines(S)
    print(S)
    f.close()

def print_parameters():
    parameters = 0
    for variable in tf.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    Message = 'TP: {:,}'.format(parameters)
    return Message

def get_feed_dicts(data, placeholders, shuffle=0):
    num_batch = data[0].shape[0]
    feed_dicts = []
    if shuffle:
        per = list(np.random.permutation(num_batch))  # 随机划分
    else:
        per = range(num_batch)
    for j in per:
        feed_dict = {}
        for i in range(len(placeholders)):
            feed_dict[placeholders[i]] = data[i][j, ...]
        feed_dicts.append(feed_dict)
    return feed_dicts

# 配置优化器/学习率/剪裁梯度/反向更新/
def optimization(args, loss):
    lr = tf.Variable(tf.constant_initializer(args.Base_Lr)(shape=[]),
                     dtype=tf.float32, trainable=False, name='learning_rate')  # (F, F1)
    new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
    lr_update = tf.assign(lr, new_lr)
    if args.Opt == 'adam':
        optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-3)
    elif args.Opt == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif args.Opt == 'amsgrad':
        optimizer = tf.train.AMSGrad(lr, epsilon=1e-3)
    # clip
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, args.Max_Grad_Norm)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
    return lr, new_lr, lr_update, train_op

# 配置GPU
def GPU(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

# 恢复模型或初始化模型
def restore(sess, saver, Model_Path, Val_Loss_File, Checkpoint_File):
    print(Model_Path)
    ckpt = tf.train.get_checkpoint_state(Model_Path, latest_filename=Checkpoint_File)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        Epoch = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
        val_loss_min = np.load(Val_Loss_File)['loss']
        message = "restore successfully, path:%s, Epoch:%d" % (ckpt.model_checkpoint_path, Epoch)
        print( message)
        return val_loss_min, Epoch
    else:
        print("ckpt", ckpt)
        raise ValueError

# 计算一个epoch，训练时反向更新，测试时进行预测
def caculation(args,  sess, feed_dicts, labels, preds, train_op, loss, type="train"):
    start = time.time()
    loss_all = []
    preds_all = []
    labels_all = []
    message_res = ''
    wmape = np.inf
    for feed_dict in feed_dicts:
        if type == "train":
            sess.run([train_op], feed_dict=feed_dict)
        batch_loss = sess.run([loss], feed_dict=feed_dict)
        loss_all.append(batch_loss)
        if type == "test":
            batch_labels, batch_preds = sess.run([labels, preds], feed_dict=feed_dict)
            preds_all.append(batch_preds)
            labels_all.append(batch_labels)
    loss_mean = np.mean(loss_all)
    Time = time.time() - start

    if type == "test":
        preds_all = np.stack(preds_all, axis=0)
        preds_all = np.round(preds_all).astype(np.float32)
        labels_all = np.stack(labels_all, axis=0)
        mae, rmse, wmape, smape = metrics.calculate_metrics(preds_all, labels_all, null_val=args.Null_Val)
        message = "Test=> MAE:{:.4f} RMSE:{:.4f} WMAPE:{:.4f} SMAPE:{:.4f}".format(mae, rmse, wmape, smape)
        message_res = "MAE\t%.4f\tRMSE\t%.4f\tWMAPE\t%.4f\tSMAPE\t%.4f" % (mae, rmse, wmape, smape)
        print( message)
    return [loss_mean, Time], message_res, wmape