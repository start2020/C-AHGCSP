import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import model_common, main_common
import numpy as np

# "all" "complete" "Long" "GCN"  "LSTM" "Dym" "KL" "Geo"
def Model(args, mean, std):
    if args.ablation == "all":
        outputs, placeholders = Model_all(args, mean, std)
    elif args.ablation == "complete":
        outputs, placeholders = Model_delcomplete(args, mean, std)
    elif args.ablation == "Long":
        outputs, placeholders = Model_delLong(args, mean, std)
    elif args.ablation == "GCN":
        outputs, placeholders = Model_delGCN(args, mean, std)
    elif args.ablation == "LSTM":
        outputs, placeholders = Model_delLSTM(args, mean, std)
    elif args.ablation == "Dym":
        outputs, placeholders = Model_delDym(args, mean, std)
    elif args.ablation == "KL":
        outputs, placeholders = Model_delKL(args, mean, std)
    elif args.ablation == "Geo":
        outputs, placeholders = Model_delGeo(args, mean, std)
    elif args.ablation == "Prior":
        outputs, placeholders = Model_delPrior(args, mean, std)
    else:
        raise ValueError
    return outputs, placeholders

def Model_delPrior(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:,0:args.Q,...] # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P-args.Q):
                OtDQ = OtD[:,i:args.Q+i,...] # (None,Q,N,N)
                ODtQ = ODt[:,i:args.Q+i,...] # (None,Q,N,N)
                attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
                FusionQ = fusion(attentionQ, Geo, KL)
                XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks, repeat=True) #(B,P,N,F)
                XQ = tf.transpose(XQ, perm=(0,2,1,3)) # X:(B,Q,N,F)=>(B,N,Q,F)
                XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1])) # (B,N,Q,F)=>(B*N,P,F)
                XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
                XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1])) #(B,N,F)
                #XQ = long_short_fusion(args, labels_W[:,args.Q+i,...], XQ, repeat=True) # 融合 #(B,N,F)
                XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True, Scope="inverse") # (B,N,N)

                # #加入先验信息
                # XQ_r = tf.nn.softmax(XQ, axis=-1) #(B,N,N)
                # OtD_ar_w = OtD_ar[:,args.Q+i,...] #(B,P,N,N) =>(B,N,N)
                # delay_r = XQ_r * OtD_ar_w #(B,N,N)*(B,N,N)
                # delay_OD = tf.expand_dims(Delayed_Inflow[:,args.Q+i,...], axis=-1)*delay_r #(B,N,1)*(B,N,N)
                # XQ = delay_OD + OtD_b[:,args.Q+i,...] #(B,N,N)+(B,N,N)
                # XQ = tf.reshape(XQ, (-1,1,args.N, args.N)) #(B,1,N,N)

                OtD = tf.concat([OtD, XQ],axis=1) #(None,Q+1,N,N)

    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks) #(B,P,N,F)
    X = tf.transpose(X, perm=(0,2,1,3)) # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1])) # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1])) #(B,N,F)
    #X = long_short_fusion(args, Label_W, X) # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders

def Model_delLSTM(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:,0:args.Q,...] # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P-args.Q):
                OtDQ = OtD[:,i:args.Q+i,...] # (None,Q,N,N)
                ODtQ = ODt[:,i:args.Q+i,...] # (None,Q,N,N)
                attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
                FusionQ = fusion(attentionQ, Geo, KL)
                XQ = OtDQ
                #XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks, repeat=True) #(B,P,N,F)
                XQ = tf.transpose(XQ, perm=(0,2,1,3)) # X:(B,Q,N,F)=>(B,N,Q,F)
                XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1])) # (B,N,Q,F)=>(B*N,P,F)
                XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
                XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1])) #(B,N,F)
                #XQ = long_short_fusion(args, labels_W[:,args.Q+i,...], XQ, repeat=True) # 融合 #(B,N,F)
                XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True, Scope="inverse") # (B,N,N)

                # 加入先验信息
                XQ_r = tf.nn.softmax(XQ, axis=-1) #(B,N,N)
                OtD_ar_w = OtD_ar[:,args.Q+i,...] #(B,P,N,N) =>(B,N,N)
                delay_r = XQ_r * OtD_ar_w #(B,N,N)*(B,N,N)
                delay_OD = tf.expand_dims(Delayed_Inflow[:,args.Q+i,...], axis=-1)*delay_r #(B,N,1)*(B,N,N)
                XQ = delay_OD + OtD_b[:,args.Q+i,...] #(B,N,N)+(B,N,N)
                XQ = tf.reshape(XQ, (-1,1,args.N, args.N)) #(B,1,N,N)
                OtD = tf.concat([OtD, XQ],axis=1) #(None,Q+1,N,N)
    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks) #(B,P,N,F)
    X = tf.transpose(X, perm=(0,2,1,3)) # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1])) # (B,N,P,F)=>(B*N,P,F)
    #X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
    X = tf.reduce_mean(X, axis=1)  # (B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1])) #(B,N,F)
    #X = long_short_fusion(args, Label_W, X) # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders

def Model_delGCN(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:,0:args.Q,...] # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P-args.Q):
                OtDQ = OtD[:,i:args.Q+i,...] # (None,Q,N,N)
                ODtQ = ODt[:,i:args.Q+i,...] # (None,Q,N,N)
                attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
                FusionQ = fusion(attentionQ, Geo, KL)
                XQ = OtDQ
                #XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks, repeat=True) #(B,P,N,F)
                XQ = tf.transpose(XQ, perm=(0,2,1,3)) # X:(B,Q,N,F)=>(B,N,Q,F)
                XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1])) # (B,N,Q,F)=>(B*N,P,F)
                XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
                XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1])) #(B,N,F)
                #XQ = long_short_fusion(args, labels_W[:,args.Q+i,...], XQ, repeat=True) # 融合 #(B,N,F)
                XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True, Scope="inverse") # (B,N,N)

                # 加入先验信息
                XQ_r = tf.nn.softmax(XQ, axis=-1) #(B,N,N)
                OtD_ar_w = OtD_ar[:,args.Q+i,...] #(B,P,N,N) =>(B,N,N)
                delay_r = XQ_r * OtD_ar_w #(B,N,N)*(B,N,N)
                delay_OD = tf.expand_dims(Delayed_Inflow[:,args.Q+i,...], axis=-1)*delay_r #(B,N,1)*(B,N,N)
                XQ = delay_OD + OtD_b[:,args.Q+i,...] #(B,N,N)+(B,N,N)
                XQ = tf.reshape(XQ, (-1,1,args.N, args.N)) #(B,1,N,N)
                OtD = tf.concat([OtD, XQ],axis=1) #(None,Q+1,N,N)

    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL)
    X = OtD
    #X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks) #(B,P,N,F)
    X = tf.transpose(X, perm=(0,2,1,3)) # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1])) # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1])) #(B,N,F)
    #X = long_short_fusion(args, Label_W, X) # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders

def Model_delLong(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:,0:args.Q,...] # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P-args.Q):
                OtDQ = OtD[:,i:args.Q+i,...] # (None,Q,N,N)
                ODtQ = ODt[:,i:args.Q+i,...] # (None,Q,N,N)
                attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
                FusionQ = fusion(attentionQ, Geo, KL)
                XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks, repeat=True) #(B,P,N,F)
                XQ = tf.transpose(XQ, perm=(0,2,1,3)) # X:(B,Q,N,F)=>(B,N,Q,F)
                XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1])) # (B,N,Q,F)=>(B*N,P,F)
                XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
                XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1])) #(B,N,F)
                #XQ = long_short_fusion(args, labels_W[:,args.Q+i,...], XQ, repeat=True) # 融合 #(B,N,F)
                XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True, Scope="inverse") # (B,N,N)

                # 加入先验信息
                XQ_r = tf.nn.softmax(XQ, axis=-1) #(B,N,N)
                OtD_ar_w = OtD_ar[:,args.Q+i,...] #(B,P,N,N) =>(B,N,N)
                delay_r = XQ_r * OtD_ar_w #(B,N,N)*(B,N,N)
                delay_OD = tf.expand_dims(Delayed_Inflow[:,args.Q+i,...], axis=-1)*delay_r #(B,N,1)*(B,N,N)
                XQ = delay_OD + OtD_b[:,args.Q+i,...] #(B,N,N)+(B,N,N)
                XQ = tf.reshape(XQ, (-1,1,args.N, args.N)) #(B,1,N,N)
                OtD = tf.concat([OtD, XQ],axis=1) #(None,Q+1,N,N)

    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks) #(B,P,N,F)
    X = tf.transpose(X, perm=(0,2,1,3)) # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1])) # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1])) #(B,N,F)
    #X = long_short_fusion(args, Label_W, X) # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders

def Model_delDym(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:, 0:args.Q, ...]  # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P - args.Q):
            OtDQ = OtD[:, i:args.Q + i, ...]  # (None,Q,N,N)
            ODtQ = ODt[:, i:args.Q + i, ...]  # (None,Q,N,N)
            attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
            FusionQ = fusion(attentionQ, Geo, KL, num=2)
            XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units,
                                        Ks=AHGCSP_GCN_Ks, repeat=True)  # (B,P,N,F)
            XQ = tf.transpose(XQ, perm=(0, 2, 1, 3))  # X:(B,Q,N,F)=>(B,N,Q,F)
            XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1]))  # (B,N,Q,F)=>(B*N,P,F)
            XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type)  # (B*N,F)
            XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1]))  # (B,N,F)
            #XQ = long_short_fusion(args, labels_W[:, args.Q + i, ...], XQ, repeat=True)  # 融合 #(B,N,F)
            XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True,
                                       Scope="inverse")  # (B,N,N)

            # 加入先验信息
            XQ_r = tf.nn.softmax(XQ, axis=-1)  # (B,N,N)
            OtD_ar_w = OtD_ar[:, args.Q + i, ...]  # (B,P,N,N) =>(B,N,N)
            delay_r = XQ_r * OtD_ar_w  # (B,N,N)*(B,N,N)
            delay_OD = tf.expand_dims(Delayed_Inflow[:, args.Q + i, ...], axis=-1) * delay_r  # (B,N,1)*(B,N,N)
            XQ = delay_OD + OtD_b[:, args.Q + i, ...]  # (B,N,N)+(B,N,N)
            XQ = tf.reshape(XQ, (-1, 1, args.N, args.N))  # (B,1,N,N)
            OtD = tf.concat([OtD, XQ], axis=1)  # (None,Q+1,N,N)

    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL, num=2)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units,
                               Ks=AHGCSP_GCN_Ks)  # (B,P,N,F)
    X = tf.transpose(X, perm=(0, 2, 1, 3))  # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1]))  # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type)  # (B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1]))  # (B,N,F)
    #X = long_short_fusion(args, Label_W, X)  # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders

def Model_delKL(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:, 0:args.Q, ...]  # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P - args.Q):
            OtDQ = OtD[:, i:args.Q + i, ...]  # (None,Q,N,N)
            ODtQ = ODt[:, i:args.Q + i, ...]  # (None,Q,N,N)
            attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
            FusionQ = fusion(attentionQ, Geo, KL, num=1)
            XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units,
                                        Ks=AHGCSP_GCN_Ks, repeat=True)  # (B,P,N,F)
            XQ = tf.transpose(XQ, perm=(0, 2, 1, 3))  # X:(B,Q,N,F)=>(B,N,Q,F)
            XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1]))  # (B,N,Q,F)=>(B*N,P,F)
            XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type)  # (B*N,F)
            XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1]))  # (B,N,F)
            #XQ = long_short_fusion(args, labels_W[:, args.Q + i, ...], XQ, repeat=True)  # 融合 #(B,N,F)
            XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True,
                                       Scope="inverse")  # (B,N,N)

            # 加入先验信息
            XQ_r = tf.nn.softmax(XQ, axis=-1)  # (B,N,N)
            OtD_ar_w = OtD_ar[:, args.Q + i, ...]  # (B,P,N,N) =>(B,N,N)
            delay_r = XQ_r * OtD_ar_w  # (B,N,N)*(B,N,N)
            delay_OD = tf.expand_dims(Delayed_Inflow[:, args.Q + i, ...], axis=-1) * delay_r  # (B,N,1)*(B,N,N)
            XQ = delay_OD + OtD_b[:, args.Q + i, ...]  # (B,N,N)+(B,N,N)
            XQ = tf.reshape(XQ, (-1, 1, args.N, args.N))  # (B,1,N,N)
            OtD = tf.concat([OtD, XQ], axis=1)  # (None,Q+1,N,N)

    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL, num=1)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units,
                               Ks=AHGCSP_GCN_Ks)  # (B,P,N,F)
    X = tf.transpose(X, perm=(0, 2, 1, 3))  # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1]))  # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type)  # (B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1]))  # (B,N,F)
    #X = long_short_fusion(args, Label_W, X)  # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders


def Model_delGeo(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:, 0:args.Q, ...]  # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P - args.Q):
            OtDQ = OtD[:, i:args.Q + i, ...]  # (None,Q,N,N)
            ODtQ = ODt[:, i:args.Q + i, ...]  # (None,Q,N,N)
            attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
            FusionQ = fusion(attentionQ, Geo, KL, num=0)
            XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units,
                                        Ks=AHGCSP_GCN_Ks, repeat=True)  # (B,P,N,F)
            XQ = tf.transpose(XQ, perm=(0, 2, 1, 3))  # X:(B,Q,N,F)=>(B,N,Q,F)
            XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1]))  # (B,N,Q,F)=>(B*N,P,F)
            XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type)  # (B*N,F)
            XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1]))  # (B,N,F)
            #XQ = long_short_fusion(args, labels_W[:, args.Q + i, ...], XQ, repeat=True)  # 融合 #(B,N,F)
            XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True,
                                       Scope="inverse")  # (B,N,N)

            # 加入先验信息
            XQ_r = tf.nn.softmax(XQ, axis=-1)  # (B,N,N)
            OtD_ar_w = OtD_ar[:, args.Q + i, ...]  # (B,P,N,N) =>(B,N,N)
            delay_r = XQ_r * OtD_ar_w  # (B,N,N)*(B,N,N)
            delay_OD = tf.expand_dims(Delayed_Inflow[:, args.Q + i, ...], axis=-1) * delay_r  # (B,N,1)*(B,N,N)
            XQ = delay_OD + OtD_b[:, args.Q + i, ...]  # (B,N,N)+(B,N,N)
            XQ = tf.reshape(XQ, (-1, 1, args.N, args.N))  # (B,1,N,N)
            OtD = tf.concat([OtD, XQ], axis=1)  # (None,Q+1,N,N)

    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL, num=0)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units,
                               Ks=AHGCSP_GCN_Ks)  # (B,P,N,F)
    X = tf.transpose(X, perm=(0, 2, 1, 3))  # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1]))  # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type)  # (B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1]))  # (B,N,F)
    #X = long_short_fusion(args, Label_W, X)  # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders

def Model_delcomplete(args, mean, std):
    elements(args)
    OtD = OtD_b
    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units,
                               Ks=AHGCSP_GCN_Ks)  # (B,P,N,F)
    X = tf.transpose(X, perm=(0, 2, 1, 3))  # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1]))  # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type)  # (B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1]))  # (B,N,F)
    #X = long_short_fusion(args, Label_W, X)  # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    return outputs, placeholders

def elements(args):
    # 载入有关数据
    global Geo, KL, OtD_b, OtD_all, ODt, labels_W, OtD_ar, Delayed_Inflow, labels,placeholders,Label_W
    global AHGCSP_GCN_Units, AHGCSP_GCN_activations, AHGCSP_GCN_Ks, AHGCSP_RNN_units
    Geo = np.load(args.AHGCSP_Geo_Path)['arr_0'].astype(np.float32)
    KL = np.load(args.AHGCSP_KL_Path)['arr_0'].astype(np.float32)

    # Placeholders + 数据变形
    OtD_b = tf.compat.v1.placeholder(shape=(None, args.P, args.N, args.N), dtype=tf.float32, name="samples-1")
    OtD_all = tf.compat.v1.placeholder(shape=(None, args.P, args.N, args.N), dtype=tf.float32, name="samples-2")
    ODt = tf.compat.v1.placeholder(shape=(None, args.P, args.N, args.N), dtype=tf.float32, name="samples-2")
    labels_W = tf.compat.v1.placeholder(shape=(None, args.P + 1, args.N, args.N), dtype=tf.float32, name="samples-3")
    OtD_ar = tf.compat.v1.placeholder(shape=(None, args.P, args.N, args.N), dtype=tf.float32, name="samples-4")
    Delayed_Inflow = tf.compat.v1.placeholder(shape=(None, args.P, args.N), dtype=tf.float32, name="samples-5")
    labels = tf.compat.v1.placeholder(shape=(None, args.N, args.N), dtype=tf.float32, name="lables")
    placeholders = [labels, OtD_b, ODt,labels_W, OtD_ar, Delayed_Inflow, OtD_all]
    Label_W = labels_W[:, -1, ...]  # (None,N,N)
    # Check the tensor name
    for p in range(len(placeholders)):
        print(placeholders[p])
    #exit()

    # 超参数
    AHGCSP_GCN_Units = [int(unit) for unit in args.AHGCSP_GCN_Units.split(",")]
    AHGCSP_GCN_activations = args.AHGCSP_GCN_Activations.split(",")
    AHGCSP_RNN_units = [int(i) for i in args.AHGCSP_RNN_Units.split(",")]
    AHGCSP_GCN_Ks = [int(i) for i in args.AHGCSP_GCN_Ks.split(",")]

def single_target(args, X, std, mean):
    #反标准化-单目标
    outputs_1 = model_common.multi_targets(X, std, mean, args.N)
    return [outputs_1]

def double_target(args, X, std, mean, OtD):
    # 反标准化-双目标
    outputs_1 = model_common.multi_targets(X, std, mean, args.N)
    outputs_P = model_common.multi_targets(OtD, std, mean, args.N)
    return [outputs_1, outputs_P]

def Model_all(args, mean, std):
    elements(args)
    # 数据补全
    OtD = OtD_b[:,0:args.Q,...] # (None,Q,N,N)
    with tf.variable_scope("complete", reuse=tf.AUTO_REUSE):
        for i in range(args.P-args.Q):

                OtDQ = OtD[:,i:args.Q+i,...] # (None,Q,N,N)
                ODtQ = ODt[:,i:args.Q+i,...] # (None,Q,N,N)
                attentionQ = Dynamic_Matrix(args, OtDQ, ODtQ, repeat=True)
                FusionQ = fusion(attentionQ, Geo, KL)
                XQ = model_common.multi_gcn(FusionQ, OtDQ, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks, repeat=True) #(B,P,N,F)
                XQ = tf.transpose(XQ, perm=(0,2,1,3)) # X:(B,Q,N,F)=>(B,N,Q,F)
                XQ = tf.reshape(XQ, shape=(-1, args.Q, XQ.shape[-1])) # (B,N,Q,F)=>(B*N,P,F)
                XQ = model_common.multi_lstm(XQ, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
                XQ = tf.reshape(XQ, shape=(-1, args.N, XQ.shape[-1])) #(B,N,F)
                #XQ = long_short_fusion(args, labels_W[:,args.Q+i,...], XQ, repeat=True) # 融合 #(B,N,F)

                XQ = model_common.multi_fc(XQ, activations=['relu'], units=[args.N], repeat=True, Scope="inverse") # (B,N,N)

                # 加入先验信息
                XQ_r = tf.nn.softmax(XQ, axis=-1) #(B,N,N)
                OtD_ar_w = OtD_ar[:,args.Q+i,...] #(B,P,N,N) =>(B,N,N)
                delay_r = XQ_r * OtD_ar_w #(B,N,N)*(B,N,N)
                delay_OD = tf.expand_dims(Delayed_Inflow[:,args.Q+i,...], axis=-1)*delay_r #(B,N,1)*(B,N,N)
                XQ = delay_OD + OtD_b[:,args.Q+i,...] #(B,N,N)+(B,N,N)
                XQ = tf.reshape(XQ, (-1,1,args.N, args.N)) #(B,1,N,N)
                OtD = tf.concat([OtD, XQ],axis=1) #(None,Q+1,N,N)

    # 模型逻辑
    attention = Dynamic_Matrix(args, OtD, ODt)
    Fusion = fusion(attention, Geo, KL)
    X = model_common.multi_gcn(Fusion, OtD, activations=AHGCSP_GCN_activations, units=AHGCSP_GCN_Units, Ks=AHGCSP_GCN_Ks) #(B,P,N,F)
    X = tf.transpose(X, perm=(0,2,1,3)) # X:(B,P,N,F)=>(B,N,P,F)
    X = tf.reshape(X, shape=(-1, args.P, X.shape[-1])) # (B,N,P,F)=>(B*N,P,F)
    X = model_common.multi_lstm(X, AHGCSP_RNN_units, type=args.AHGCSP_RNN_Type) #(B*N,F)
    X = tf.reshape(X, shape=(-1, args.N, X.shape[-1])) #(B,N,F)
    #X = long_short_fusion(args, Label_W, X) # 融合 #(B,N,F)

    if args.Single == "1":
        outputs = single_target(args, X, std, mean)
    else:
        outputs = double_target(args, X, std, mean, OtD)
    # check output name
    print(outputs)
    return outputs, placeholders

def long_short_fusion(args, Label_W, Short_X, repeat=False):
    AHGCSP_Fusion_Units = [int(unit) for unit in args.AHGCSP_Fusion_Units.split(",")]
    AHGCSP_Fusion_activations = args.AHGCSP_Fusion_Activations.split(",")
    x1 = model_common.multi_fc(Label_W, activations=AHGCSP_Fusion_activations, units=AHGCSP_Fusion_Units, repeat=repeat, Scope="long_term")
    x2 = model_common.multi_fc(Short_X, activations=AHGCSP_Fusion_activations, units=AHGCSP_Fusion_Units, repeat=repeat, Scope="short_term")
    output = x1 + x2
    return output

def Dynamic_Matrix(args, OtD, ODt, repeat=False):
    AHGCSP_Dynamic_Units = [int(unit) for unit in args.AHGCSP_Dynamic_Units.split(",")]
    AHGCSP_Dynamic_activations = args.AHGCSP_Dynamic_Activations.split(",")
    # (B, P, N, N) => (B, P, N, F)
    query_OtD = model_common.multi_fc(OtD, activations=AHGCSP_Dynamic_activations, units=AHGCSP_Dynamic_Units, repeat=repeat, Scope="query_OtD")
    key_OtD = model_common.multi_fc(OtD, activations=AHGCSP_Dynamic_activations, units=AHGCSP_Dynamic_Units,repeat=repeat, Scope="key_OtD")
    value_OtD = model_common.multi_fc(OtD, activations=AHGCSP_Dynamic_activations, units=AHGCSP_Dynamic_Units, repeat=repeat, Scope="value_OtD")
    attention_OtD = tf.matmul(query_OtD, key_OtD, transpose_b=True) # (B, P, N, F)*(B, P, F, N)=>(B, P, N, N)
    attention_OtD /= (AHGCSP_Dynamic_Units[-1] ** 0.5)
    attention_OtD = tf.nn.softmax(attention_OtD, axis = -1)
    #X_OtD = tf.matmul( attention_OtD, value_OtD) # (B, P, N, N)* (B, P, N, F) =>

    # (B, P, N, N) => (B, P, N, F)
    query_ODt = model_common.multi_fc(ODt, activations=AHGCSP_Dynamic_activations, units=AHGCSP_Dynamic_Units, repeat=repeat, Scope="query_ODt")
    key_ODt = model_common.multi_fc(ODt, activations=AHGCSP_Dynamic_activations, units=AHGCSP_Dynamic_Units, repeat=repeat, Scope="query_ODt")
    value_ODt = model_common.multi_fc(ODt, activations=AHGCSP_Dynamic_activations, units=AHGCSP_Dynamic_Units, repeat=repeat, Scope="value_ODt")
    attention_ODt = tf.matmul(query_ODt, key_ODt, transpose_b=True) # (B, P, N, F)*(B, P, F, N)=>(B, P, N, N)
    attention_ODt /= (AHGCSP_Dynamic_Units[-1] ** 0.5)
    attention_ODt = tf.nn.softmax(attention_ODt, axis = -1)
    #X_ODt = tf.matmul( attention_ODt, value_ODt) # (B, P, N, N)* (B, P, N, F) =>

    W_ODt = tf.Variable(tf.glorot_uniform_initializer()(shape = [1,]),dtype = tf.float32, trainable = True, name='W_ODt') #(F, F1)
    W_OtD = tf.Variable(tf.glorot_uniform_initializer()(shape = [1,]),dtype = tf.float32, trainable = True, name='W_OtD') #(F, F1)
    X = attention_ODt * W_ODt + attention_OtD * W_OtD
    return X

#
def fusion(attention, Geo, KL, num=3):
    W_geo = tf.Variable(tf.glorot_uniform_initializer()(shape = [1,]),dtype = tf.float32, trainable = True, name='W_geo')
    W_kl = tf.Variable(tf.glorot_uniform_initializer()(shape = [1,]),dtype = tf.float32, trainable = True, name='W_kl')
    W_att = tf.Variable(tf.glorot_uniform_initializer()(shape = [1,]),dtype = tf.float32, trainable = True, name='W_att')
    if num == 0:
        W_geo = 0 # delete Geo
    elif num == "1":
        W_kl = 0 # delete KL
    elif num == 2:
        W_att = 0  # delete dynamic
    Fusion = attention * W_att + Geo * W_geo + KL * W_kl  # (B, N, N)
    Fusion = tf.nn.softmax(Fusion, axis=-1)
    return Fusion