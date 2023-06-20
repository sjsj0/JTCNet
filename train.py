# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as io
from sklearn.preprocessing import LabelBinarizer
import torch
import MyDataset
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import utils
from timesformer import TimeSformer
import logging
import gc
from linearmodel import MLP
from combined import *
from pytorch_model_summary import summary
# from torchsummary import summary

# DEFINING LOGGER
logging.basicConfig(filename="Train.log",
                format='%(asctime)s %(message)s',
                filemode='a')

logger = logging.getLogger()
logger.setLevel(logging.INFO)   

if __name__ == '__main__':

    args = utils.get_args()
    # 参数
    fileRoot = r'F:\RP_Abhijit_sir\Saved_94'
    saveRoot = r'F:\RP_Abhijit_sir\Saved_94' + str(args.fold_num) + str(args.fold_index)
    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()
    resize = transforms.Resize(size=(64, frames_num))
    best_mae = 20
    

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    tf_bvp_name = 'tf_bvp' + input_form + 'n' + str(frames_num) + 'fn' + str(fold_num) + 'fi' + str(fold_index)


    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu') #
        print('on GPU')
    else:
        print('on CPU')

    ##################################################################################################

    # 数据集
    if args.reData == 1:
        # indexes for which stmap will go to train or test
        test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=fold_num, fold_index=fold_index)
        print(len(test_index))
        print(len(train_index))

        #below two lines create MAT files
        Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'img_mvavg_full.png', 5, frames_num)
        Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test','img_mvavg_full.png', 5, frames_num)

    #make train and test objects
    train_db = MyDataset.Data_VIPL(root_dir=(saveRoot + '_Train'), frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    test_db = MyDataset.Data_VIPL(root_dir=(saveRoot + '_Test'), frames_num=frames_num, transform=transforms.Compose([resize, toTensor, normalize]))
    
    #converting DB objects to dataloader type
    train_loader = DataLoader(train_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)
    print('trainLen:', len(train_db), 'testLen:', len(test_db))
    print('fold_num:', fold_num, 'fold_index', fold_index)

    ########### COMBINED #########################
    #########################################################################
    # DEFINING TRANSFORMER PARAMETERS
    DIM = 128
    IMAGE_SIZE = 64
    PATCH_SIZE = 8
    NUM_FRAMES = 20
    DEPTH = 12
    HEADS = 8
    DIM_HEAD = 64
    ATTN_DROPOUT = 0.1
    FF_DROPOUT = 0.1
    ITERATIONS = 20
    tf_learning_rate = 0.001

    # DEFINING BVP MODEL PARAMETERS
    N_CHANNELS = 3
    N_CLASSES = 1

    tf_bvp_model = TimeSformer_BVP(dim=DIM, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_classes=1, num_frames=NUM_FRAMES, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD, attn_dropout=ATTN_DROPOUT, ff_dropout=FF_DROPOUT, n_channels=N_CHANNELS, n_classes=N_CLASSES)
    tf_bvp_optimizer = torch.optim.Adam(tf_bvp_model.parameters(), lr=tf_learning_rate)
    tf_bvp_model.to(device=device)
    #################################################################################################
    # print(tf_bvp_model)
    ############ ARCH PRINT###################
    # import hiddenlayer as hl
    # from torchviz import make_dot, make_dot_from_trace
    # x = torch.zeros((16, 1, 3, 64, 64))
    # y = torch.zeros((16, 3, 64, 256))
    # make_dot(tf_bvp_model(x, y),  show_attrs=True, params=dict(tf_bvp_model.named_parameters())).render("attached", format="pdf")

    ### Works, open it on Netron........
    # x = torch.zeros((16, 1, 3, 64, 64), requires_grad=True)
    # y = torch.zeros((16, 3, 64, 256), requires_grad=True)
    # torch.onnx.export(tf_bvp_model, (x,y), 'run.onnx', opset_version=12)





    # x = torch.randn(1, 8)
    # hl.build_graph(tf_bvp_model, [torch.zeros([32, 3, 64, 64],]))

    # print(summary(tf_bvp_model, torch.zeros((16, 1, 64, 64, 3)), torch.zeros((16, 64, 256, 3)), show_hierarchical=True))

    # # print(summary(rPPGNet, torch.zeros((32, 3, 64, 64)), show_input=True, show_hierarchical=True))
    # exit()

    #################################

    loss_func_rPPG = utils.P_loss3().to(device)
    loss_func_L1 = nn.L1Loss().to(device)
    loss_func_SP = utils.SP_loss(device, clip_length=frames_num).to(device)
    # noises为生成网络的输入
    noises = torch.randn(batch_size_num, 1, 4, int(frames_num/16)).to(device)

    ### EPOCHS STARTING HERE ###

    for epoch in range(epoch_num):
        tf_bvp_model.train()

        for step, (image, data, bvp, HR_rel, idx) in enumerate(train_loader):
            # print('Step:',step)
            # print("data shape:" ,data.shape)
            # print("image shape:" ,image.shape)
            # print("bvp shape:" ,bvp.shape)
            # print("hr_rel:" ,HR_rel.shape)
            image = np.expand_dims(image,axis=1)   # for transformer only 
            image = torch.tensor(image).float().to(device=device) 
            image = Variable(image).float().to(device=device)

            data = Variable(data).float().to(device=device)
            bvp = Variable(bvp).float().to(device=device)
            HR_rel = Variable(HR_rel).float().to(device=device)
            HR_rel = torch.unsqueeze(HR_rel, 1)
            bvp = bvp.unsqueeze(dim=1)
            STMap = data[:, :, :, 0:frames_num]
            Wave = bvp[:, :, 0:frames_num]
            b, _, _ = Wave.size()

            # print(image.shape)
            # print(STMap.shape)
        
            ########################################################
            # TRANSFORMER_BVP
            ########################################################
            tf_bvp_optimizer.zero_grad()
            Wave_pr, HR_pr = tf_bvp_model(video=image, x_Unet=STMap)
            # print(Wave_pr.shape)
            # print(HR_pr.shape)
            loss0 = loss_func_L1(HR_rel, HR_pr)
            loss1 = torch.zeros(1).to(device)
            loss2 = torch.zeros(1).to(device)
            whole_max_idx = []
            for width in range(64):
                loss1 = loss1 + loss_func_rPPG(Wave_pr[:, :, width, :], Wave)
                loss2_temp, whole_max_idx_temp = loss_func_SP(Wave_pr[:, :, width, :], HR_rel)
                loss2 = loss2_temp + loss2
                whole_max_idx.append(whole_max_idx_temp.data.cpu().numpy())
            HR_Droped = utils.Drop_HR(np.array(whole_max_idx))
            loss1 = loss1/64
            loss2 = loss2/64
            loss = loss0 + loss1 + loss2
            loss.backward()
            tf_bvp_optimizer.step()


            ##############################################################

            if step % 50 == 0:
                print('Train Epoch: ', epoch,
                      '| loss: %.4f' % loss.data.cpu().numpy(),
                      '| loss0: %.4f' % loss0.data.cpu().numpy(),
                      '| loss1: %.4f' % loss1.data.cpu().numpy(),
                      '| loss2: %.4f' % loss2.data.cpu().numpy()
                      )

            # logger.info("TRAIN     %d  %d    %.4f    %.4f    %.4f    %.4f    %.4f",epoch,step,loss.data.cpu().numpy()
            # ,loss0.data.cpu().numpy(),loss1.data.cpu().numpy(),loss2.data.cpu().numpy(),tf_loss0.data.cpu().numpy())

            del HR_pr

        #CLEAR MEMORY
        torch.cuda.empty_cache()
        gc.collect()

        # 测试
        tf_bvp_model.eval()

        HR_pr_temp = []
        HR_rel_temp = []
        HR_pr2_temp = []

        
        for step, (image,data, bvp, HR_rel, idx) in enumerate(test_loader):
            image = np.expand_dims(image,axis=1)   # for transformer only 
            image = torch.tensor(image).float().to(device=device)
            image = Variable(image).float().to(device=device)
            
            data = Variable(data).float().to(device=device)
            bvp = Variable(bvp).float().to(device=device)
            HR_rel = Variable(HR_rel).float().to(device=device)
            HR_rel = torch.unsqueeze(HR_rel, 1)
            bvp = bvp.unsqueeze(dim=1)
            STMap = data[:, :, :, 0:frames_num]
            Wave = bvp[:, :, 0:frames_num]
            b, _, _ = Wave.size()

            #######################################################
            with torch.no_grad():
                Wave_pr, HR_pr = tf_bvp_model(video=image, x_Unet=STMap)
            
            loss0 = loss_func_L1(HR_pr, HR_rel)
            loss1 = torch.zeros(1).to(device)
            loss2 = torch.zeros(1).to(device)
            whole_max_idx = []
            for width in range(64):
                loss1 = loss1 + loss_func_rPPG(Wave_pr[:, :, width, :], Wave)
                loss2_temp, whole_max_idx_temp = loss_func_SP(Wave_pr[:, :, width, :], HR_rel)
                loss2 = loss2_temp + loss2
                whole_max_idx.append(whole_max_idx_temp.data.cpu().numpy())
            HR_Droped = utils.Drop_HR(np.array(whole_max_idx))
            loss1 = loss1 / 64
            loss2 = loss2 / 64
            loss = loss0 + loss1 + loss2
            HR_pr_temp.extend(HR_pr.data.cpu().numpy())
            HR_pr2_temp.extend(HR_Droped)
            HR_rel_temp.extend(HR_rel.data.cpu().numpy())
            #######################################################
           
            if step % 50 == 0:
                print('Test Epoch: ', epoch,
                      '| loss: %.4f' % loss.data.cpu().numpy(),
                      '| loss0: %.4f' % loss0.data.cpu().numpy(),
                      '| loss1: %.4f' % loss1.data.cpu().numpy(),
                      '| loss2: %.4f' % loss2.data.cpu().numpy()
                      )

            # logger.info("TEST    %d  %d    %.4f    %.4f    %.4f    %.4f    %.4f",epoch,step,loss.data.cpu().numpy()
            # ,loss0.data.cpu().numpy(),loss1.data.cpu().numpy(),loss2.data.cpu().numpy(),tf_loss0.data.cpu().numpy())

            del HR_pr

        print('HR:')
        #Printing Performance metrics 

        #bvpnet
        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
        logger.info("TF_BVP***       %d     me:    %.4f        std:    %.4f           mae:     %.4f          rmse:     %.4f         mer:      %.4f       p:       %.4f",epoch,ME, STD, MAE, RMSE, MER, P)
        ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr2_temp, HR_rel_temp)
        logger.info("TF_BVP_DR       %d     me:    %.4f        std:    %.4f           mae:     %.4f          rmse:     %.4f         mer:      %.4f       p:       %.4f",epoch,ME, STD, MAE, RMSE, MER, P)
       
        torch.save(tf_bvp_model, tf_bvp_name)
        print('saveModel As ' + tf_bvp_name)
        
        if best_mae > MAE:
            best_mae = MAE
            io.savemat(tf_bvp_name + 'HR_pr.mat', {'HR_pr': HR_pr_temp})
            io.savemat(tf_bvp_name + 'HR_rel.mat', {'HR_rel': HR_rel_temp})
        
        
