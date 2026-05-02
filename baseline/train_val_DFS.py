import sys
from sklearn import metrics

from torch._C import device 
sys.path.append(".") 
import argparse
from nni.utils import merge_parameter
import nni
from utils.utils import *
from glob import glob
import pandas as pd
from DatasetLoader import MILDatasetLoader, MILFeatureLoader
from model import AttentionNetMoreLayer
import torch.optim as optim
from utils.logger import create_logger
import csv

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--name", type=str, default="DFS_SEED2")
    parser.add_argument("--savedir", type=str, default="./checkpoints/save_dir/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--jizhi", type=int, default=0)
    parser.add_argument("--lr_patience", type=int, default=100)
    parser.add_argument("--instance_attention_layers", type=int, default=5)
    parser.add_argument("--instance_attention_dim", type=int, default=256)
    parser.add_argument("--feature_attention_layers", type=int, default=5)
    parser.add_argument("--feature_attention_dim", type=int, default=128)
    parser.add_argument("--instance_lr", type=float, default=1e-3)
    parser.add_argument("--feat_lr", type=float, default=1e-3)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--split_seed", type=int, default=2)
    parser.add_argument("--aug", type=int, default=1)



def get_filepaths(opt):

     


    with open("/data3/louwei/MedComm/codes/baseline/utils/splits/seed_"+str(opt.split_seed)+"/train.csv", "r") as f:
        reader = csv.reader(f)
        train_ID = list(reader)[0]
    with open("/data3/louwei/MedComm/codes/baseline/utils/splits/seed_"+str(opt.split_seed)+"/val.csv", "r") as f:
        reader = csv.reader(f)
        val_ID = list(reader)[0]

    test = glob('/data3/louwei/MedComm/data/test/*')
    test_ID = []
    for ID in test:
        test_ID.append(ID.split('/')[-1][0:7])
    test_ID = list(set(test_ID))
    
    train_tab = pd.read_excel("/data3/louwei/MedComm/data/table/hx-lz-1.1.xlsx", index_col=0)
    train_DFS = train_tab.loc[train_ID, 'DFS'].values
    train_OS = train_tab.loc[train_ID, 'OS'].values
    train_DEvent = train_tab.loc[train_ID, 'Distant metastasis（no=0；yes=1）'].values
    train_OSEvent = train_tab.loc[train_ID, 'Death（no=0；yes=1）'].values

    val_tab = pd.read_excel("/data3/louwei/MedComm/data/table/hx-lz-1.1.xlsx", index_col=0)
    val_DFS = val_tab.loc[val_ID, 'DFS'].values
    val_OS = val_tab.loc[val_ID, 'OS'].values
    val_DEvent = val_tab.loc[val_ID, 'Distant metastasis（no=0；yes=1）'].values
    val_OSEvent = val_tab.loc[val_ID, 'Death（no=0；yes=1）'].values    
    
    # 获取验证集样本的DFS和OS
    test_tab = pd.read_excel("/data3/louwei/MedComm/data/table/sx - 5.0-更新后.xlsx", index_col=0)
    # 验证集部分样本在表格中无法找到对应的信息，需要去除这部分病人编号
    test_ID_exist = []
    test_DFS = []
    for ID in test_ID:
        try:
            test_DFS.append(test_tab.loc[int(ID), 'DFS'])
            test_ID_exist.append(ID)
        except:
            continue
    test_ID = test_ID_exist
    test_OS = []
    for ID in test_ID:
        test_OS.append(test_tab.loc[int(ID), 'OS'])
        
    test_DEvent = []
    for ID in test_ID:
        test_DEvent.append(test_tab.loc[int(ID), 'Distant metastasis（no=0；yes=1）'])

    test_OSEvent = []
    for ID in test_ID:
        test_OSEvent.append(test_tab.loc[int(ID), 'Death（no=0；yes=1）'])
    test_DFS, test_OS = np.array(test_DFS), np.array(test_OS)
    test_DEvent, test_OSEvent = np.array(test_DEvent), np.array(test_OSEvent) 


    return {'ID': train_ID, 'DFS': train_DFS, 'OS': train_OS, 'DEvent': train_DEvent, 'OSEvent': train_OSEvent}, \
        {'ID': val_ID, 'DFS': val_DFS, 'OS': val_OS, 'DEvent': val_DEvent, 'OSEvent': val_OSEvent}, \
        {'ID': test_ID, 'DFS': test_DFS, 'OS': test_OS, 'DEvent': test_DEvent, 'OSEvent': test_OSEvent}


def get_optimizer(model, args):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer

def create_optimizers(nets: torch.nn.Module, instance_lr, feat_lr, classifier_lr):
    weight_decay = 1e-4

    instance_att_params = nets.module.get_instance_attention_parameters()
    feature_att_params = nets.module.get_feature_attention_parameters()
    classifier_att_params = nets.module.get_classifier_parameters()

    ins_optimizer = torch.optim.AdamW(instance_att_params, lr=instance_lr, weight_decay=weight_decay)
    feat_optimizer = torch.optim.AdamW(feature_att_params, lr=feat_lr, weight_decay=weight_decay)
    classifier_optimizer = torch.optim.AdamW(classifier_att_params, lr=classifier_lr, weight_decay=weight_decay)
    return [ins_optimizer, feat_optimizer, classifier_optimizer]

def get_scheduler(optimizers, args):

    ins_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[0], "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    feat_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[1], "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[2], "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )
    return [ins_scheduler, feat_scheduler, classifier_scheduler]
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_factor)
    # return scheduler


def model_eval(test_loader, model, dataname):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data = batch['data'].cuda().float()
            data = data.squeeze(0)
            pred = model(data)
            if i!=0:
                pred_all = torch.cat((pred_all, pred), 0)
                OS_all = torch.cat((OS_all, batch['DFS'].cuda().float()), 0)
                OSEvent_all = torch.cat((OSEvent_all, batch['DEvent'].cuda().float()), 0)
            else:
                pred_all = pred
                OS_all, OSEvent_all = batch['DFS'].cuda().float(), batch['DEvent'].cuda().float()
        loss = CoxLoss(OS_all, OSEvent_all, pred_all)

    OS_all =  OS_all.detach().cpu().numpy()
    OSEvent_all = OSEvent_all.detach().cpu().numpy()
    pred_all = pred_all.detach().cpu().numpy()

    label = {"time": OS_all, "event": OSEvent_all}
    metrics = eval(pred_all, label, dataname)
    metrics.update({dataname+"loss": loss})
    return metrics



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train Models")
    opt = get_args(parser)
    opt, remaining_args = parser.parse_known_args()
    nniparams = nni.get_next_parameter()
    opt = merge_parameter(opt, nniparams)
    set_seed(opt.seed)
    print(opt)

    try:
        savedir = os.environ['NNI_OUTPUT_DIR']
    except:
        savedir = os.path.join(opt.savedir, opt.name)
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, 'config.yaml'), 'w') as f:
        f.write("{}".format(savedir))

    
    train_fp, val_fp, test_fp = get_filepaths(opt)

    model = AttentionNetMoreLayer(attentionL=2560, attentionD=512, attentionK=3, \
        dropout_p=opt.dropout, instance_attention_layers=opt.instance_attention_layers, instance_attention_dim=opt.instance_attention_dim, \
            feature_attention_layers=opt.feature_attention_layers, feature_attention_dim=opt.feature_attention_dim)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda(device=range(torch.cuda.device_count())[0])
    optimizers = create_optimizers(model, opt.instance_lr, opt.feat_lr, opt.classifier_lr)
    schedulers = get_scheduler(optimizers, opt)

    logger = create_logger("%s/logfile.txt" % savedir)
    logger.info(f"Training/evaluation parameters: {opt}")
    if opt.aug ==1:
        train_dataset = MILFeatureLoader(opt, train_fp, path = 'train',aug=True, shuffle_bag=True)
    else:
        train_dataset = MILFeatureLoader(opt, train_fp, path = 'train', aug=False, shuffle_bag=True)
    #for i in range(len(train_dataset)):
        #sample = train_dataset[i]
        #print(f"Sample {i}: {sample}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_sz,
        shuffle=True,
        num_workers = opt.n_workers,
        drop_last = True,
        pin_memory = True
    )
    #for i, batch in enumerate(train_loader):
        #p#rint(i,batch)
    val_dataset = MILFeatureLoader(opt, val_fp,path = 'train', aug=False, shuffle_bag=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_sz,
        shuffle=False,
        num_workers = opt.n_workers,
        drop_last = False,
        pin_memory = True
    )
    #print(len(test_fp['ID']))
    test_dataset = MILFeatureLoader(opt, test_fp,path = 'test', aug=False, shuffle_bag=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=opt.batch_sz,
        shuffle=False,
        num_workers = opt.n_workers,
        drop_last = False,
        pin_memory = True        
    )

    print('train num: ',len(train_dataset),'val num: ',len(val_dataset),'test num: ',len(test_dataset))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
    
    for i_epoch in range(start_epoch, opt.max_epochs):
        model.train()        
        train_loss = []
        model.zero_grad()
        for i, batch in enumerate(train_loader):
            data = batch['data'].cuda(range(torch.cuda.device_count())[0]).float()
            with torch.no_grad():
                if data.dim() == 3 and data.size(0) == 1:
                    data = data.squeeze(0)
                else:
                    data = data
            #print(batch['PatientID'])
            #print(batch['data'].shape)
            #pred = model(data)
            
            
            pred = model(data.unsqueeze(0))
            #print(i,pred)
            if i != 0:
                OS_all = torch.cat((OS_all, batch['DFS'].cuda(range(torch.cuda.device_count())[0]).float()), 0)
                OSEvent_all = torch.cat((OSEvent_all, batch['DEvent'].cuda(range(torch.cuda.device_count())[0]).float()), 0)                
                pred_all = torch.cat((pred_all, pred), 0)
            else:
                pred_all = pred
                OS_all, OSEvent_all = batch['DFS'].cuda(range(torch.cuda.device_count())[0]).float(), batch['DEvent'].cuda(range(torch.cuda.device_count())[0]).float()
        #print('loss:',OS_all.shape,OSEvent_all.shape,pred_all.shape)
        loss = CoxLoss(OS_all, OSEvent_all, pred_all)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            
        # optimizer.step()
        # optimizer.zero_grad()

        logger.info("Train Loss: {:.4f}".format(np.mean(loss.item())))
        train_metrics = model_eval(train_loader, model, 'tr')
        print(train_metrics)

        logger.info("Val Loss: {:.4f}".format(np.mean(loss.item())))
        val_metrics = model_eval(val_loader, model, 'val')
        print(val_metrics)

        test_metrics = model_eval(test_loader, model, 'sx')
        print(test_metrics)    

        #test_lz_metrics = model_eval(test_lz_loader, model, 'lz')
        #print(test_lz_metrics)    

        tuning_metric = val_metrics["val_cindex"]
        metric = {'default': val_metrics["val_cindex"]}
        metric.update(val_metrics)
        metric.update(test_metrics)
        #metric.update(test_lz_metrics)

        nni.report_intermediate_result(metric)
        for scheduler in schedulers:
            scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
            nni_best_metric = metric
            save_checkpoint(
                {
                    "epoch": i_epoch + 1,
                    "state_dict": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    # "scheduler": scheduler.state_dict(),
                    "n_no_improve": n_no_improve,
                    "best_metric": best_metric,
                },
                savedir+'/checkpoint',
            )            
        else:
            n_no_improve += 1

    nni.report_final_result(nni_best_metric)
    print(nni_best_metric)