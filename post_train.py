import params

from common import *
from dataset.carvana_cars import *
from train_seg_net import evaluate, criterion, show_batch_results

from model.tool import *
from model.rate import *
from model.segmentation.loss import *
from model.segmentation.blocks import *

Net = params.post_model

CSV_BLOCK_SIZE = 10000

# ------------------------------------------------------------------------------------
def run_post_train():

    if params.my_computer:
        out_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
    else:
        out_dir = '/kaggle_data_results/results/lhc/single/' + params.save_path

    initial_checkpoint = None
        #'/root/share/project/kaggle-carvana-cars/results/single/UNet128-00-xxx/checkpoint/006.pth'


    #logging, etc --------------------
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir+'/post_train/train/results', exist_ok=True)
    os.makedirs(out_dir+'/post_train/valid/results', exist_ok=True)
    os.makedirs(out_dir+'/post_train/backup', exist_ok=True)
    os.makedirs(out_dir+'/post_train/checkpoint', exist_ok=True)
    os.makedirs(out_dir+'/post_train/snap', exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/post_train/backup/train.code.zip')

    log = Logger()
    log.open(out_dir+'/post_train/log.post_train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** experiment for average labels channel as prior**\n\n')
    
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')



    ## dataset ----------------------------------------
    def train_augment(image,label):
        #image, label = random_horizontal_flipN([image, label])
        image, label = random_shift_scale_rotateN([image, label], shift_limit=(-0.0625,0.0625),
                  scale_limit=(0.91,1.21), rotate_limit=(-0,0))

        #image, label = random_mask_hue(image, label, hue_limit=(-1,1), u=0.5)
        #image = random_hue(image, hue_limit=(-1,1), u=0.5)

        # image = random_brightness(image, limit=(-0.5,0.5), u=0.5)
        # image = random_contrast  (image, limit=(-0.5,0.5), u=0.5)
        # image = random_saturation(image, limit=(-0.3,0.3), u=0.5)
        # image = random_gray(image, u=0.25)

        return  image, label

    ## ----------------------------------------------------



    log.write('** dataset setting **\n')
    #batch_size   =  2
    batch_size = params.step_batch_size
    num_grad_acc =  params.real_batch_size//batch_size

    train_dataset = post_prosses_Dataset(  'train_v0_4320',
                                   #'train_5088',
                                   'train',
                                   #'train128x128', ## 1024x1024 ##
                                   #'test_100064', 'test1024x1024',
                                   #transform = [ lambda x,y:train_augment(x,y), ],
                                   mode='train')
    train_loader  = DataLoader(
                        train_dataset,
                        #sampler = RandomSampler(train_dataset),
                        sampler = RandomSamplerWithLength(train_dataset,4320),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True)
    ##check_dataset(train_dataset, train_loader), exit(0)

    valid_dataset = post_prosses_Dataset('valid_v0_768',
                                 #'train128x128', 
                                 'train',
                                 mode='train')
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)


    log.write('\ttrain_dataset.split = %s\n'%train_dataset.split)
    log.write('\tvalid_dataset.split = %s\n'%valid_dataset.split)
    log.write('\tlen(train_dataset)  = %d\n'%len(train_dataset))
    log.write('\tlen(valid_dataset)  = %d\n'%len(valid_dataset))
    log.write('\n%s\n\n'%(inspect.getsource(train_augment)))


    ## net ----------------------------------------
    log.write('** net setting **\n')

    #net = Net(in_shape=(3, 128, 128))
    net = Net(in_shape=(4, params.input_size, params.input_size))
    net.cuda()

    log.write('%s\n\n'%(type(net)))
    log.write('%s\n\n'%(str(net)))
    log.write('%s\n\n'%(inspect.getsource(net.__init__)))
    log.write('%s\n\n'%(inspect.getsource(net.forward )))

    ## optimiser ----------------------------------
    if params.optimer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0005)
    if params.optimer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0)

    num_epoches = 150  #100
    it_print    = 1    #20
    it_smooth   = 20
    epoch_valid = list(range(0,num_epoches+1))
    epoch_save  = list(range(0,num_epoches+1))
    #LR = StepLR([ (0, 0.01),  (35, 0.005),  (40,0.001),  (42, -1),(44, -1)])
    if params.optimer == 'SGD':
        LR = StepLR([ (0, 0.01),  (35, 0.005),  (40,0.001),  (45, 0.0002),(55, -1)])
    if params.optimer == 'Adam':
        LR = StepLR([ (0, 0.001),  (35, 0.0005),  (55, -1)])

    #https://github.com/EKami/carvana-challenge/blob/7d20494f40b39686c25159403e2a27a82f4096a9/src/nn/classifier.py
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, min_lr=1e-7)
    #LR = StepLR([ (0, 0.01),])
    #LR = StepLR([ (0, 0.005),])


    ## resume from previous ------------------------
    log.write('\ninitial_checkpoint=%s\n\n'%initial_checkpoint)

    start_epoch=0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    #merge_bn_in_net(net)
    #training ####################################################################3
    log.write('** start training here! **\n')
    log.write(' num_grad_acc x batch_size = %d x %d=%d\n'%(num_grad_acc,batch_size,num_grad_acc*batch_size))
    log.write(' input_size = %d x %d\n'%(params.input_size,params.input_size) )
    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' is_ReduceLRonPlateau: %s\n'%str(params.using_ReduceLROnPlateau))
    log.write(' LR=%s\n\n'%str(LR) )
    log.write('\n')


    log.write('epoch    iter      rate   | valid_loss/acc | train_loss/acc | batch_loss/acc ... \n')
    log.write('--------------------------------------------------------------------------------------------------\n')

    num_its = len(train_loader)
    smooth_loss = 0.0
    smooth_acc  = 0.0
    train_loss  = 0.0
    train_acc   = 0.0
    #valid_loss  = 0.0
    valid_loss = 100
    valid_acc   = 0.0
    batch_loss  = 0.0
    batch_acc   = 0.0
    time = 0

    start_lr = get_learning_rate(optimizer)[0]

    start0 = timer()
    for epoch in range(start_epoch, num_epoches+1):  # loop over the dataset multiple times

        if epoch > params.max_epochs: break
        #---learning rate schduler ------------------------------
        if params.using_ReduceLROnPlateau == True:
            adjust_learning_rate(optimizer, start_lr/num_grad_acc)
            lr_scheduler.step(valid_loss)
            rate = get_learning_rate(optimizer)[0]*num_grad_acc #check
            start_lr = rate
        else:
            lr = LR.get_rate(epoch, num_epoches)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr/num_grad_acc)
            rate = get_learning_rate(optimizer)[0]*num_grad_acc #check
        #--------------------------------------------------------


        # validate at current epoch
        if epoch in epoch_valid:
            net.eval()
            valid_loss, valid_acc = evaluate(net, valid_loader)

            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   | %0.5f  %0.5f | %0.5f  %0.5f | %0.5f  %0.5f  |  %3.1f min \n' % \
                    (epoch, num_its, rate, valid_loss, valid_acc, train_loss, train_acc, batch_loss, batch_acc, time))


        #if 1:
        if epoch in epoch_save:
            torch.save(net.state_dict(),out_dir +'/post_train/snap/%03d.pth'%epoch)
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch,
            }, out_dir +'/post_train/checkpoint/%03d.pth'%epoch)
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py

        if epoch==num_epoches: break ##########################################-


        start = timer()
        sum_train_loss = 0.0
        sum_train_acc  = 0.0
        sum = 0

        net.train()
        for it, (images, labels, indices) in enumerate(train_loader, 0):
            images  = Variable(images).cuda()
            labels  = Variable(labels).cuda()

            #forward
            logits = net(images)
            probs  = F.sigmoid(logits)
            masks  = (probs>0.5).float()

            loss = criterion(logits, labels, is_weight=True)
            acc  = dice_loss(masks, labels)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # accumulate gradients
            if it==0:
                optimizer.zero_grad()
            loss.backward()
            if it%num_grad_acc==0:
                optimizer.step()
                optimizer.zero_grad()  # assume no effects on bn for accumulating grad


            # print statistics
            batch_acc  = acc.data [0]
            batch_loss = loss.data[0]
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if it%it_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = 0.0
                sum_train_acc  = 0.0
                sum = 0

            if it%it_print == 0 or it==num_its-1:
                print('\r%5.1f   %5d    %0.4f   | .......  ....... | %0.5f  %0.5f | %0.5f  %0.5f ' % \
                        (epoch + (it+1)/num_its, it+1, rate, train_loss, train_acc, batch_loss, batch_acc),\
                        end='',flush=True)

            #debug show prediction results ---
            if 0:
            #if it%100==0:
                show_batch_results(indices, images, probs, labels,
                                   wait=1, out_dir=out_dir+'/post_train/train/results', names=train_dataset.names, epoch=epoch, it=it)

        end  = timer()
        time = (end - start)/60
        #end of epoch --------------------------------------------------------------



    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60
    log.write('\nalltime = %f min\n'%time0)
    ## save final
    torch.save(net.state_dict(),out_dir +'/post_train/snap/final.pth')

# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    opts, args = getopt.getopt(sys.argv[1:], 't', ['s1', 's2'])

    for opt, val in opts:
        print(opt)

    if opt == '-t':
        run_post_train()
        '''
    elif opt == '--s1':
        run_submit1()
    elif opt == '--s2':
        run_submit2()
    else:
        print('nothing,stop')
        '''

    print('\nsucess!')