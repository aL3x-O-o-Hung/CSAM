from dataloader import *
from CAT_Net import CrossSliceAttentionUNet,CrossSliceUNetPlusPlus
import argparse
from datetime import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle



def print_log(print_string,log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()




def dice_coeff(seg,target,smooth=0.001):
    intersection=np.sum(seg*target)
    dice=(2*intersection+smooth)/(np.sum(seg)+np.sum(target)+smooth)
    return dice


def validate(model,dataloader,args):
    device=args.device
    model.eval()
    total_dice=np.zeros(args.num_classes-1)
    c=0
    criterion=torch.nn.CrossEntropyLoss()
    loss=0
    for batch_num,data in enumerate(dataloader):
        img,mask,mask_onehot,length=data['im'],data['mask'],data['m'],data['length']
        img=img.to(device).squeeze(0)[:length[0],:,:,:]
        mask=mask.to(device).squeeze(0)[:length[0],:,:]
        mask_onehot=mask_onehot.to(device).squeeze(0)[:length[0],:,:,:]
        pred_raw=model(img)
        pred=F.softmax(pred_raw,dim=1)

        tmp_loss=criterion(pred_raw,mask)
        loss+=tmp_loss.item()

        pred_np=pred.detach().cpu().numpy()
        mask_onehot_np=mask_onehot.detach().cpu().numpy()

        pred_np=np.moveaxis(pred_np,1,-1)
        mask_onehot_np=np.moveaxis(mask_onehot_np,1,-1)
        pred_onehot_np=np.zeros_like(pred_np)

        pred_np=np.argmax(pred_np,axis=-1)
        for i in range(args.input_size):
            for j in range(args.input_size):
                for k in range(pred_np.shape[0]):
                    pred_onehot_np[k,i,j,pred_np[k,i,j]]=1
        for i in range(args.num_classes-1):
            total_dice[i]+=dice_coeff(pred_onehot_np[:,:,:,i:i+1],mask_onehot_np[:,:,:,i:i+1])
        c+=1

    return total_dice/c,loss/c

def unet_init(input_channels=1,num_classes=3,num_layers=6,heads=1,num_attention_blocks=1,base_num=64,pool_kernel_size=4,input_size=128,batch_size=20,pool_method="avgpool",is_pe_learnable=True):
    network=CrossSliceAttentionUNet(input_channels,num_classes,num_layers,heads,num_attention_blocks,base_num,(pool_kernel_size,pool_kernel_size),(input_size,input_size),batch_size,pool_method,is_pe_learnable)
    return network

def unetplusplus_init(input_channels=1,num_classes=3,num_layers=5,heads=1,num_attention_blocks=1,base_num=64,pool_kernel_size=4,input_size=128,batch_size=20,pool_method="avgpool",is_pe_learnable=True):
    network=CrossSliceUNetPlusPlus(input_channels,num_classes,num_layers,heads,num_attention_blocks,base_num,(pool_kernel_size,pool_kernel_size),(input_size,input_size),batch_size,pool_method,is_pe_learnable)
    return network

def train(args):
    device=args.device
    # epochs,current_epoch,mode,dataset,train_batch_size,heads,num_attention_blocks, data_path, save_path, learning_rate, try_id
    if args.mode=='unet':
        network=unet_init(heads=args.num_heads,num_attention_blocks=args.num_attention_blocks,pool_method=args.pool_method,is_pe_learnable=args.is_pe_learnable,batch_size=args.sequence_length,pool_kernel_size=args.pool_kernel_size,input_size=args.input_size)
    elif args.mode=='unetplusplus':
        network=unetplusplus_init(heads=args.num_heads,num_attention_blocks=args.num_attention_blocks,pool_method=args.pool_method,is_pe_learnable=args.is_pe_learnable,batch_size=args.sequence_length,pool_kernel_size=args.pool_kernel_size,input_size=args.input_size)
    #elif args.mode=='yk3d_softmax':
    #    network=yk3d_init(mode='softmax',heads=args.num_heads, num_attention_blocks=args.num_attention_blocks)

    else:
        print('not implemented yet!')
        return

    now=datetime.now()
    current_time=now.strftime("%m-%d-%Y_%H:%M:%S")

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    log=open(args.save_path+"model_training_log_id-{}_t-{}.txt".format(args.try_id,current_time),'w')
    state={k:v for k,v in args._get_kwargs()}
    print_log(state,log)  # generate logs e.g. {'alpha': 1.0, 'batch_size': 4, 'belta': 1.0, ...

    print("Load Network Successfully!")
    model_parameters=filter(lambda p:p.requires_grad,network.parameters())
    params=sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    network=network.to(device)

    if args.current_epoch>=0:
        network.load_state_dict(torch.load(args.save_path+str(args.current_epoch)+'.pt'))
    network=network.to(device)
    network.train()

    criterion=torch.nn.CrossEntropyLoss()

    if args.is_regularization:
        optimizer=torch.optim.Adam(network.parameters(),lr=args.lr,weight_decay=args.reg_val)
    else:
        optimizer=torch.optim.Adam(network.parameters(),lr=args.lr)

    train_dataloader=Internal3D(root=args.data_path,split='train')
    train_loader=torch.utils.data.DataLoader(train_dataloader,batch_size=args.batch_size,shuffle=True,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    val_dataloader=Internal3D(root=args.data_path,split='val')
    val_loader=torch.utils.data.DataLoader(val_dataloader,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    test_dataloader=Internal3D(root=args.data_path,split='test')
    test_loader=torch.utils.data.DataLoader(test_dataloader,batch_size=1,shuffle=False,num_workers=1,pin_memory=True,sampler=None,drop_last=True)
    loss_history_train=[]
    loss_history_val=[]
    dice_score_history_val=[]
    loss_history_test=[]
    dice_score_history_test=[]

    pz_val=0
    tz_val=0
    sum_val=0
    pz_test=0
    tz_test=0
    sum_test=0

    for epoch in range(args.current_epoch+1,args.epochs):

        for batch,data in enumerate(train_loader):
            img,mask,mask_onehot,length=data['im'],data['mask'],data['m'],data['length']
            #img should be batch_size x sequence_length x channels (1) x height x width

            img=img.to(device)
            mask=mask.to(device)
            mask_onehot=mask_onehot.to(device)
            length=length.to(device)

            network.train()
            optimizer.zero_grad()
            loss=0
            for i in range(img.size(0)):
                pred=network(img[i,:length[i],:,:,:])
                if 'unetplusplus' in args.mode:
                    for p in pred:
                        loss+=criterion(p,mask[i,:length[i],:,:])
                else:
                    loss+=criterion(pred,mask[i,:length[i],:,:])

            # average of all cases
            loss/=img.size(0)

            if batch==0:
                loss_history_train.append(loss.item())

            loss.backward()
            optimizer.step()

            # validation and test
        with torch.no_grad():
            network.eval()
            # Validation loss and dice calculation
            dice_scores_val,loss_val=validate(network,val_loader,args)
            dice_score_history_val.append(dice_scores_val)
            loss_history_val.append(loss_val)

            if dice_scores_val[0]>tz_val:
                tz_val=dice_scores_val[0]
                torch.save(network.state_dict(),args.save_path+'tz.pt')
                print_log("----------Save model for TZ at: {:.4f}, {:.4f} ---------".format(dice_scores_val[0],dice_scores_val[1]),log)

            if dice_scores_val[1]>pz_val:
                pz_val=dice_scores_val[1]
                torch.save(network.state_dict(),args.save_path+'pz.pt')
                print_log("----------Save model for PZ at: {:.4f}, {:.4f} ---------".format(dice_scores_val[0],dice_scores_val[1]),log)

            if dice_scores_val[1]+dice_scores_val[0]>sum_val:
                sum_val=dice_scores_val[1]+dice_scores_val[0]
                torch.save(network.state_dict(),args.save_path+'sum.pt')
                print_log("----------Save model for Both at: {:.4f}, {:.4f} ---------".format(dice_scores_val[0],dice_scores_val[1]),log)

            #testing loss and dice calculation
            dice_scores_test,loss_test=validate(network,test_loader,args)
            dice_score_history_test.append(dice_scores_test)
            loss_history_test.append(loss_test)

            if dice_scores_test[0]>tz_test:
                tz_test=dice_scores_test[0]
                print_log("----------Test TZ max at: {:.4f}, {:.4f} ---------".format(dice_scores_test[0],dice_scores_test[1]),log)

            if dice_scores_test[1]>pz_test:
                pz_test=dice_scores_test[1]
                print_log("----------Test PZ max at: {:.4f}, {:.4f} ---------".format(dice_scores_test[0],dice_scores_test[1]),log)

            if dice_scores_test[1]+dice_scores_test[0]>sum_test:
                sum_test=dice_scores_test[1]+dice_scores_test[0]
                print_log("----------Test Both max at: {:.4f}, {:.4f}---------".format(dice_scores_test[0],dice_scores_test[1]),log)

            msg="Epoch:{}, LR:{:.6f}, Train-Loss:{:.4f}, Val-Dice:[{:.4f}, {:.4f}], Val-Loss:{:.4f}, Test-Dice:[{:.4f}, {:.4f}], Test-Loss:{:.4f}".format\
                (epoch,optimizer.param_groups[0]['lr'],loss_history_train[-1],dice_scores_val[0],dice_scores_val[1],loss_history_val[-1],dice_scores_test[0],dice_scores_test[1],loss_history_test[-1])
            print_log(msg,log)

        pickle.dump(loss_history_train,open(args.save_path+'loss_history_train.p','wb'))
        pickle.dump(loss_history_val,open(args.save_path+'loss_history_val.p','wb'))
        pickle.dump(dice_score_history_val,open(args.save_path+'val_dice_score_history.p','wb'))
        pickle.dump(loss_history_test,open(args.save_path+'loss_history_test.p','wb'))
        pickle.dump(dice_score_history_test,open(args.save_path+'test_dice_score_history.p','wb'))


def main():
    parser=argparse.ArgumentParser(description='After fixing the bug of the first block')
    parser.add_argument('--comments',default="Modified the code, add log and history of loss/dice & change BN to default momentum 0.1",type=str,help='Comment to which hyperparameter this group experiments aim to test')
    parser.add_argument('--epochs',default=150,type=int,help='number of total epochs to run')
    parser.add_argument('--current_epoch',default=-1,type=int,help='current starting epoch')
    parser.add_argument('--mode',default='unet',type=str,help='mode name to be used')
    parser.add_argument('--num_classes',default=3,type=int,help='TZ, PZ, background')
    parser.add_argument('--batch_size',default=2,type=int,help='current starting epoch')
    parser.add_argument('--num_heads',default=3,type=int,help='num of heads')
    parser.add_argument('--num_attention_blocks',default=2,type=int,help='num of attention blocks')
    parser.add_argument('--pool_kernel_size',default=4,type=int,help='pool kernel size')
    parser.add_argument('--input_size',default=128,type=int,help='input size')
    parser.add_argument('--lr',default=0.0001,type=float,help='learning rate')
    parser.add_argument('--data_path',default='../data/',type=str,help='dataset using')
    parser.add_argument('--save_path',default='',type=str,help='dataset using')
    parser.add_argument('--try_id',default='0',type=str,help='id of try')
    parser.add_argument('--network_dim',default="3D",type=str,help='2D or 3D in network using')
    parser.add_argument('--is_gamma',default=True,help='Whether add gamma transformation or not',action='store_false')
    parser.add_argument('--is_regularization',default=False,help='Whether we add regularization in optimizer or not',action='store_false')
    parser.add_argument('--reg_val',default=1e-5,type=float,help='How much regularization we want to add')
    parser.add_argument('--pool_method',default="avgpool",type=str,help='maxpool or avgpool for extracting features for self attention')
    parser.add_argument('--is_pe_learnable',default=True,help='Is the positional embedding learnable?',action='store_false')
    parser.add_argument('--sequence_length',default=20,type=int,help='length of the sequence')
    parser.add_argument('--device',default='cuda:0',type=str,help='device to use')

    args=parser.parse_args()

    #########################################################
    now=datetime.now()
    current_time=now.strftime("%m-%d-%Y_%H:%M:%S")

    #args.save_path='{}_id:{}_{}_model/'.format(args.network_dim,args.try_id,current_time)
    args.save_path=args.mode+'_'+args.dataset+'_{}_model/'.format(current_time)
    train(args)


if __name__=='__main__':
    main()