from common import *
from submit import *
from dataset.carvana_cars import *
from net.segmentation.my_unet import SoftDiceLoss, BCELoss2d, UNet_double_1024_5 as Net
from net.tool import *


## experiment setting here ----------------------------------------------------
def criterion(logits, labels):
    #l = BCELoss2d()(logits, labels)
    l = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)
    return l



## experiment setting here ----------------------------------------------------




#https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
#https://www.kaggle.com/c/carvana-image-masking-challenge#evaluation
def one_dice_loss_py(m1, m2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum()+1) / (m1.sum() + m2.sum()+1)
    return score

#https://github.com/pytorch/pytorch/issues/1249
def dice_loss(m1, m2 ):
    num = m1.size(0)
    m1  = m1.view(num,-1)
    m2  = m2.view(num,-1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = score.sum()/num
    return score




# def predict(net, test_loader):
#
#     test_dataset = test_loader.dataset
#
#     num = len(test_dataset)
#     H, W = CARVANA_H, CARVANA_W
#     predictions  = np.zeros((num, H, W),np.float32)
#
#     test_num  = 0
#     for it, (images, indices) in enumerate(test_loader, 0):
#         images = Variable(images.cuda(),volatile=True)
#
#         # forward
#         logits = net(images)
#         probs  = F.sigmoid(logits)
#
#         batch_size = len(indices)
#         test_num  += batch_size
#         start = test_num-batch_size
#         end   = test_num
#         predictions[start:end] = probs.data.cpu().numpy().reshape(-1, H, W)
#
#     assert(test_num == len(test_loader.sampler))
#     return predictions


# def predict(net, test_loader, dtype = np.uint8):
#
#     test_dataset = test_loader.dataset
#
#     num = len(test_dataset)
#     H, W = CARVANA_H, CARVANA_W
#     predictions  = np.zeros((num, 2*H, 2*W),dtype)
#
#     test_num  = 0
#     for it, (images, indices) in enumerate(test_loader, 0):
#         batch_size = len(indices)
#         test_num  += batch_size
#
#
#         # forward
#         images = Variable(images.cuda(),volatile=True)
#         logits = net(images)
#         probs  = F.sigmoid(logits)
#
#         probs = probs.data.cpu().numpy().reshape(-1, 2*H, 2*W)
#         if dtype==np.uint8:
#             probs = probs*255
#         predictions[test_num-batch_size : test_num] = probs
#
#     assert(test_num == len(test_loader.sampler))
#     return predictions

#  100064/32000 =
def predict_in_blocks(net, test_loader, block_size=32000):

    test_dataset = test_loader.dataset
    test_iter    = iter(test_loader)
    test_num     = len(test_dataset)
    batch_size   = test_loader.batch_size
    H, W = CARVANA_H, CARVANA_W


    #debug
    #checks = []

    num  = 0
    predictions = []
    for n in range(0, test_num, block_size):
        M = block_size if n+block_size < test_num else test_num-n
        print('n=%d, M=%d'%(n,M) )
        p = np.zeros((M, 2*H, 2*W),np.uint8)
        c = np.zeros((M),np.int64)

        for m in range(0, M, batch_size):
            images, indices  = test_iter.next()
            if images is None: break

            batch_size = len(indices)
            num  += batch_size

            #print('\tm=%d, m+batch_size=%d'%(m,m+batch_size) )
            #c[m : m+batch_size] = indices.cpu().numpy()

            # forward
            images = Variable(images.cuda(),volatile=True)
            logits = net(images)
            probs  = F.sigmoid(logits)

            probs = probs.data.cpu().numpy().reshape(-1, 2*H, 2*W)
            probs = probs*255
            p[m : m+batch_size] = probs

        predictions.append(p)
        #checks.append(c)

    assert(test_num == num)
    #return checks
    return predictions




def predict_and_evaluate(net, test_loader ):

    test_dataset = test_loader.dataset

    num = len(test_dataset)
    H, W = CARVANA_H, CARVANA_W
    predictions  = np.zeros((num, 2*H, 2*W),np.float32)

    test_acc  = 0
    test_loss = 0
    test_num  = 0
    for it, (images, labels, indices) in enumerate(test_loader, 0):
        images = Variable(images.cuda(),volatile=True)
        labels = Variable(labels.cuda(),volatile=True)

        # forward
        logits = net(images)
        probs  = F.sigmoid(logits)
        masks  = (probs>0.5).float()

        loss = criterion(logits, labels)
        acc  = dice_loss(masks, labels)


        batch_size = len(indices)
        test_num  += batch_size
        test_loss += batch_size*loss.data[0]
        test_acc  += batch_size*acc.data[0]
        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1, 2*H, 2*W)

    assert(test_num == len(test_loader.sampler))

    test_loss = test_loss/test_num
    test_acc  = test_acc/test_num

    return predictions, test_loss, test_acc










def show_train_batch_results(probs, labels, images, indices, wait=1, save_dir=None, names=None, epoch=0, it=0):

    probs  = (probs.data.cpu().numpy().squeeze()*255).astype(np.uint8)
    labels = (labels.data.cpu().numpy()*255).astype(np.uint8)
    images = (images.data.cpu().numpy()*255).astype(np.uint8)
    images = np.transpose(images, (0, 2, 3, 1))

    batch_size,H,W,C = images.shape
    results = np.zeros((H, 3*W, 3),np.uint8)
    prob    = np.zeros((H, W, 3),np.uint8)
    for b in range(batch_size):
        m = probs [b]>128
        l = labels[b]>128
        score = one_dice_loss_py(m , l)

        image = images[b]
        prob[:,:,1] = cv2.resize(probs [b],(H,W))
        prob[:,:,2] = cv2.resize(labels[b],(H,W))

        results[:,  0:W  ] = image
        results[:,  W:2*W] = prob
        results[:,2*W:3*W] = cv2.addWeighted(image, 1, prob, 1., 0.) # image * α + mask * β + λ
        draw_shadow_text  (results, '%0.3f'%score, (5,15),  0.5, (255,255,255), 1)

        if save_dir is not None:
            shortname = names[indices[b]].split('/')[-1].replace('.jpg','')
            #cv2.imwrite(save_dir + '/%s.jpg'%shortname, results)
            cv2.imwrite(save_dir + '/%05d-%03d.jpg'%(it,b), results)

        im_show('train',  results,  resize=1)
        cv2.waitKey(wait)



# ------------------------------------------------------------------------------------
def run_train():


    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet_double_1024_5'
    initial_checkpoint = None #'/root/share/project/kaggle-carvana-cars/results/xx5-UNet128_2_two-loss/checkpoint/020.pth'
    #



    #logging, etc --------------------
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir+'/train/results', exist_ok=True)
    os.makedirs(out_dir+'/valid/results', exist_ok=True)
    os.makedirs(out_dir+'/test/results',  exist_ok=True)
    os.makedirs(out_dir+'/backup', exist_ok=True)
    os.makedirs(out_dir+'/checkpoint', exist_ok=True)
    os.makedirs(out_dir+'/snap', exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/train.code.zip')

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')




    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 8
    train_dataset = KgCarDataset( 'train%dx%d_v0_4320'%(CARVANA_H,CARVANA_W),
                                  #'train%dx%d_5088'%(CARVANA_H,CARVANA_W),   #'train128x128_5088',  #'train_5088'
                                transform=[
                                    lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.1,0.1), rotate_limit=(-0,0)),
                                    #lambda x,y:  randomShiftScaleRotate2(x,y,shift_limit=(-0.0625,0.0625), scale_limit=(-0.0,0.0),  aspect_limit = (1-1/1.2   ,1.2-1), rotate_limit=(0,0)),
                                    lambda x,y:  randomHorizontalFlip2(x,y),
                                ],
                                is_label=True,
                                is_preload=True,)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),  #ProbSampler(train_dataset),  #ProbSampler(train_dataset,SAMPLING_PROB),  # #FixedSampler(train_dataset,list(range(batch_size))),  ##
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 3,
                        pin_memory  = True)



    valid_dataset = KgCarDataset( 'valid%dx%d_v0_768'%(CARVANA_H,CARVANA_W),
                                is_label=True,
                                is_preload=True,)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 3,
                        pin_memory  = True)


    test_dataset = KgCarDataset( 'test%dx%d_3197'%(CARVANA_H,CARVANA_W),
                                  is_label=False,
                                  is_preload=False,)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True)

    H,W = CARVANA_H, CARVANA_W
    log.write('\tbatch_size          = %d\n'%batch_size)
    log.write('\ttrain_dataset.split = %s\n'%train_dataset.split)
    log.write('\tvalid_dataset.split = %s\n'%valid_dataset.split)
    log.write('\ttest_dataset.split  = %s\n'%test_dataset.split)
    log.write('\tlen(train_dataset)  = %d\n'%len(train_dataset))
    log.write('\tlen(valid_dataset)  = %d\n'%len(valid_dataset))
    log.write('\tlen(test_dataset)   = %d\n'%len(test_dataset))

    ## net ----------------------------------------
    log.write('** net setting **\n')

    net = Net(in_shape=(3, H, W), num_classes=1)
    net.cuda()

    log.write('%s\n\n'%(type(net)))

    ## optimiser ----------------------------------
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  ###0.0005

    num_epoches = 35  #100
    it_print    = 1   #20
    it_smooth   = 20
    epoch_test  = 5
    epoch_valid = 1
    epoch_save  = [0,3,5,10,15,20,25,35,40,45,50, num_epoches-1]

    ## resume from previous ----------------------------------
    start_epoch=0
    if initial_checkpoint is not None:
        checkpoint  = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        net.load_state_dict(checkpoint['state_dict'])





    #training ####################################################################3
    log.write('** start training here! **\n')
    log.write('\n')


    log.write('epoch    iter      rate   | smooth_loss/acc | train_loss/acc | valid_loss/acc ... \n')
    log.write('--------------------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    smooth_acc  = 0.0
    train_loss = np.nan
    train_acc  = np.nan
    valid_loss = np.nan
    valid_acc  = np.nan
    time = 0
    start0 = timer()
    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        #print ('epoch=%d'%epoch)
        start = timer()

        #---learning rate schduler ------------------------------
        # lr = LR.get_rate(epoch, num_epoches)
        # if lr<0 : break
        #if epoch>=25:
        #    adjust_learning_rate(optimizer, lr=0.005)
        if epoch>=num_epoches-2:
            adjust_learning_rate(optimizer, lr=0.001)

        rate =  get_learning_rate(optimizer)[0] #check
        #--------------------------------------------------------


        sum_smooth_loss = 0.0
        sum_smooth_acc  = 0.0
        sum = 0
        net.train()
        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):
            images  = Variable(images.cuda())
            labels  = Variable(labels.cuda())

            #forward
            logits = net(images)
            probs = F.sigmoid(logits)
            masks = (probs>0.5).float()


            #backward
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            acc  = dice_loss(masks, labels)

            sum_smooth_loss += loss.data[0]
            sum_smooth_acc  += acc .data[0]
            sum += 1

            if it%it_smooth == 0:
                smooth_loss = sum_smooth_loss/sum
                smooth_acc  = sum_smooth_acc /sum
                sum_smooth_loss = 0.0
                sum_smooth_acc  = 0.0
                sum = 0


            if it%it_print == 0 or it==num_its-1:
                train_acc  = acc.data [0]
                train_loss = loss.data[0]
                print('\r%5.1f   %5d    %0.4f   |  %0.4f  %0.4f | %0.4f  %6.4f | ... ' % \
                        (epoch + (it+1)/num_its, it+1, rate, smooth_loss, smooth_acc, train_loss, train_acc),\
                        end='',flush=True)


            #debug show prediction results ---
            if 0:
                show_train_batch_results(probs, labels, images, indices,
                                         wait=1, save_dir=out_dir+'/train/results', names=train_dataset.names, epoch=epoch, it=it)

        end  = timer()
        time = (end - start)/60
        #end of epoch --------------------------------------------------------------



        if epoch % epoch_valid == 0 or epoch == 0 or epoch == num_epoches-1:
            net.eval()
            valid_predictions, valid_loss, valid_acc = predict_and_evaluate(net, valid_loader)


            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   |  %0.4f  %0.4f | %0.4f  %6.4f | %0.4f  %6.4f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, smooth_loss, smooth_acc, train_loss, train_acc, valid_loss, valid_acc, time))

        if 0:
        #if epoch % epoch_test == 0 or epoch == 0 or epoch == num_epoches-1:
            net.eval()
            probs = predict(net, test_loader)

            results = np.zeros((H, 3*W, 3),np.uint8)
            prob    = np.zeros((H, W, 3),np.uint8)
            num_test = len(probs)
            for b in range(100):
                n = random.randint(0,num_test-1)
                shortname    = test_dataset.names[b].split('/')[-1].replace('.jpg','')
                image, index = test_dataset[n]
                image        = tensor_to_image(image, std=255)
                prob[:,:,1]  = probs[n]*255

                results[:,  0:W  ] = image
                results[:,  W:2*W] = prob
                results[:,2*W:3*W] = cv2.addWeighted(image, 1, prob, 1., 0.) # image * α + mask * β + λ

                cv2.imwrite(out_dir+'/test/results/%s.jpg'%shortname, results)
                im_show('test',  results,  resize=1)
                cv2.waitKey(1)

        if epoch in epoch_save:
            torch.save(net.state_dict(),out_dir +'/snap/%03d.pth'%epoch)
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch,
            }, out_dir +'/checkpoint/%03d.pth'%epoch)
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py





    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60

    ## check : load model and re-test
    torch.save(net.state_dict(),out_dir +'/snap/final.pth')


# ------------------------------------------------------------------------------------
# https://www.kaggle.com/tunguz/baseline-2-optimal-mask/code
def run_submit():

    #out_dir  = '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2_two-loss-fulla'
    #out_dir  = '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2'
    out_dir  = '/root/share/project/kaggle-carvana-cars/results/single/UNet_double_1024_5'
    model_file = out_dir +'/snap/final.pth'  #final

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 8

    test_dataset = KgCarDataset( 'test%dx%d_100064'%(CARVANA_H,CARVANA_W),
                                  is_label=False,
                                  is_preload=False,  #True,
                               )
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True)

    H,W = CARVANA_H, CARVANA_W
    log.write('\tbatch_size         = %d\n'%batch_size)
    log.write('\ttest_dataset.split = %s\n'%test_dataset.split)
    log.write('\tlen(test_dataset)  = %d\n'%len(test_dataset))
    log.write('\n')


    ## net ----------------------------------------
    net = Net(in_shape=(3, H, W), num_classes=1)
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    ## start testing now #####
    log.write('start prediction ...\n')
    if 0:
        #start = timer()
        #log.write('\tnp.load time = %f min\n'%((timer() - start) / 60))

        start = timer()
        net.eval()
        probs = predict_in_blocks( net, test_loader, block_size=32000 )           # 20 min
        log.write('\tpredict_in_blocks = %f min\n'%((timer() - start) / 60))

        start = timer()
        for i,p in enumerate(probs):
            np.save(out_dir+'/submit/probs-part%d.8.npy'%i, p) #  1min
        log.write('\tnp.save time = %f min\n'%((timer() - start) / 60))


    else:
        probs = [None, None, None, None]



    # do encoding  -------------------------------------------------------------
    names = test_dataset.names

    num_test = len(test_dataset)
    threshold = 0.5*255
    rles=[]
    for i,p in enumerate(probs):
        if p is None:
            start = timer()
            p = np.load(out_dir+'/submit/probs-part%d.8.npy'%i)
            log.write('\tnp.load time = %f min\n'%((timer() - start) / 60))

        M = len(p)
        for m in range(M):
            if (m%1000==0):
                n = len(rles)
                end  = timer()
                time = (end - start) / 60
                time_remain = (num_test-n-1)*time/(n+1)
                print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))
            #-----------------------------

            prob = p[m]
            prob = cv2.resize(prob,(CARVANA_WIDTH,CARVANA_HEIGHT))
            mask = prob>threshold
            rle = run_length_encode(mask)
            rles.append(rle)
        p=None

    #fixe corrupted image
    # https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37247
    if 1:
        n = names.index('29bb3ece3180_11.jpg')
        mask = cv2.imgread('/root/share/project/kaggle-carvana-cars/data/others/ave/11.png',cv2.IMREAD_GRAYSCALE)>128
        rle  = run_length_encode(mask)
        rles[n] = rle


    start = timer()
    dir_name = out_dir.split('/')[-1]
    gz_file  = out_dir + '/submit/results-%s.csv.gz'%dir_name
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')
    log.write('\tdf.to_csv time = %f min\n'%((timer() - start) / 60)) #3 min
    log.write('\n')




#decode and check
def run_check_submit_csv():

    gz_file = '/root/share/project/kaggle-carvana-cars/results/single/UNet_double_1024_5/submit/results.csv.gz'
    df = pd.read_csv(gz_file, compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)

    indices =[0,1,2,32000-1,32000,32000+1,100064-1]
    for n in indices:
        name = df.values[n][0]
        img_file = CARVANA_DIR +'/images/test/%s'%name
        img = cv2.imread(img_file)
        im_show('img', img, resize=0.25)

        rle   = df.values[n][1]
        mask  = run_length_decode(rle,H=CARVANA_HEIGHT, W=CARVANA_WIDTH)
        im_show('mask', mask, resize=0.25)
        cv2.waitKey(0)

    pass




    # if 0: #------------------------------------------------------------------------------
    #     num = 500
    #     results = np.zeros((H, 3*W, 3),np.uint8)
    #     prob    = np.zeros((H, W, 3),np.uint8)
    #     num_test = len(probs)
    #     for n in range(num):
    #         shortname    = test_dataset.names[n].split('/')[-1].replace('.jpg','')
    #         image, index = test_dataset[n]
    #         image        = tensor_to_image(image, std=255)
    #         prob[:,:,1]  = probs[n]
    #
    #         results[:,  0:W  ] = image
    #         results[:,  W:2*W] = prob
    #         results[:,2*W:3*W] = cv2.addWeighted(image, 0.75, prob, 1., 0.) # image * α + mask * β + λ
    #
    #         cv2.imwrite(out_dir+'/submit/results/%s.jpg'%shortname, results)
    #         im_show('test',  results,  resize=1)
    #         cv2.waitKey(1)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_train()
    run_submit()

    print('\nsucess!')