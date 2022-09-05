from __future__ import print_function
import argparse
import sys
import time,yaml,easydict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from utils.data_loader import SYSUData, RegDBData, TestData
from utils.data_manager import *
from utils.eval_metrics import eval_sysu, eval_regdb
from models.network import embed_net
from utils.tools import *
from models.loss import OriTripletLoss, TripletLoss_WRT, KLDivLoss, TripletLoss_ADP
from tensorboardX import SummaryWriter
from utils.channel_aug import ChannelAdapGray, ChannelRandomErasing

parser = argparse.ArgumentParser(description="DSCNet VI-ReID for training")
parser.add_argument("--config", default="configs/sysu.yaml", help="config files, more details in sysu.yaml")           
args = parser.parse_args()
args = yaml.load(open(args.config), Loader=yaml.FullLoader)
args = easydict.EasyDict(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)
t = time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime())
dataset = args.dataset
if dataset == 'sysu':
    data_path = args.dataset_path
    log_path = args.log_path + 'sysu_log/' + args.mode + '/' + t + '/'
    test_mode = [1, 2]  # thermal to visible
    checkpoint_path = args.model_path + args.dataset +'/' + args.mode + '/' + t + '/'
elif dataset == 'regdb':
    data_path = args.dataset_path
    log_path = args.log_path + 'regdb_log/'+ t+'/'
    test_mode = [2, 1]  # visible to thermal
    checkpoint_path = args.model_path + args.dataset + '/' +t+'/'

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'dsc':
    suffix = suffix + '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
else:
    suffix = suffix + '_{}'.format(args.method)
if args.augc==1:
    suffix = suffix + '_aug_G'  
if args.rande>0:
    suffix = suffix + '_erase_{}'.format( args.rande)
    
suffix = suffix + '_p{}_n{}_lr_{}_seed_{}'.format( args.num_pos, args.batch_size, args.lr, args.seed)  

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + args.dataset +'/' + args.mode + '/' + t + '/' + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]
    
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.rande>0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability = args.rande)]

if args.augc ==1:
    transform_train_list = transform_train_list + [ChannelAdapGray(probability =0.5)]
    
transform_train = transforms.Compose( transform_train_list )

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset  = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)
cudnn.benchmark = True

# define loss function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
    # loader_batch = args.batch_size * args.num_pos
    # criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
elif args.method == 'adp':
    criterion_tri = TripletLoss_ADP(alpha = args.alpha, gamma = args.gamma, square = args.square)
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_kl = KLDivLoss()
criterion_id.to(device)
criterion_tri.to(device)
criterion_kl.to(device)

# ---------------------------------------------------------------- load optimizer
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    base_optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

# ----------------------------------------------------------- load checkpoints
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optim'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 40:
        lr = args.lr * 0.1
    elif epoch >= 40 and epoch < 50:
        lr = args.lr * 0.03    
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    
    base_optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(base_optimizer.param_groups) - 1):
        base_optimizer.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    kl_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0 
    modal_loss = AverageMeter()
    kl_intra_loss = AverageMeter()
    base_loss = AverageMeter()
    
    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label1, label2), 0)

        input2 = Variable(input2.cuda())
        
        
        input10 = Variable(input10.cuda())
        input11 = Variable(input11.cuda())

        labels = Variable(labels.cuda())
        
        input1 = torch.cat((input10, input11,),0)
        input2 = Variable(input2.cuda())


        data_time.update(time.time() - end)


        feat, out0, loss_modal, loss_kl_intra = net(input1, input2)
        
        loss_base = loss_modal + loss_kl_intra
        base_optimizer.zero_grad()
        loss_base.backward(retain_graph=True)
        base_optimizer.step()
        
        base_loss.update(loss_base.item(), 2 * input1.size(0))
        modal_loss.update(loss_modal.item(), 2 * input1.size(0))
        kl_intra_loss.update(loss_kl_intra.item(), 2 * input1.size(0))
        
        
        loss_id = criterion_id(out0, labels)
        
        
        # loss kl
        n = out0.shape[0]//3
        out1 = out0.narrow(0,0,n)
        out2 = out0.narrow(0,2*n,n)
        loss_kl = criterion_kl(out1, Variable(out2))
        # kl_loss += criterion_kl(F.log_softmax(out2, dim = 1), F.softmax(Variable(out1), dim=1))                                           
                                                    
        loss_tri, batch_acc = criterion_tri(feat, labels)
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)
        
        # pdb.set_trace()
        loss = loss_id + loss_tri + args.kl * loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        kl_loss.update(loss_kl.item(), 2 * input1.size(0))
        base_loss.update(loss_base.item(), 2 * input1.size(0))
        modal_loss.update(loss_modal.item(), 2 * input1.size(0))
        kl_intra_loss.update(loss_kl_intra.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'KLoss: {kl_loss.val:.4f} ({kl_loss.avg:.4f}) '
                  'BASELoss: {base_loss.val:.4f} ({base_loss.avg:.4f}) '
                  'MLoss: {modal_loss.val:.4f} ({modal_loss.avg:.4f}) '
                  'KILoss: {kl_intra_loss.val:.4f} ({kl_intra_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,kl_loss=kl_loss,base_loss=base_loss,modal_loss=modal_loss,kl_intra_loss=kl_intra_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('modal_loss', modal_loss.avg, epoch)
    writer.add_scalar('kl_loss', kl_loss.avg, epoch)
    writer.add_scalar('base_loss', base_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 200 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch >= 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'optim':optimizer.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]   Best Rank-1 {:.2%}'.format(best_epoch,best_acc))