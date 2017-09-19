from common import *
from submit import *
from dataset.carvana_cars import *
from model.tool import *

def run_vote():

    prediction_files=[
        '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2/submit/probs.8.npy',
        '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2_two-loss/submit/probs.8.npy',
        '/root/share/project/kaggle-carvana-cars/results/xx5-UNet512_2_two-loss-full_1/submit/probs.8.npy',
    ]
    out_dir ='/root/share/project/kaggle-carvana-cars/results/ensemble/xxx'

    log = Logger()
    log.open(out_dir+'/log.vote.txt',mode='a')
    os.makedirs(out_dir,  exist_ok=True)

    write_list_to_file(prediction_files, out_dir+'/prediction_files.txt')

    #----------------------------------------------------------

    #read names
    split_file = CARVANA_DIR +'/split/'+ 'test%dx%d_100064'%(CARVANA_H,CARVANA_W)
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    names = [name.split('/')[-1]+'.jpg' for name in names]

    #read probs
    num_test   = len(names)
    votes = np.zeros((num_test, CARVANA_H, CARVANA_W), np.uint8)

    num_files = len(prediction_files)
    for n in range(num_files):
        prediction_file = prediction_files[n]
        print(prediction_files[n])

        probs = np.load(prediction_file)
        votes += probs >=128
        probs = None



    #prepare csv file -------------------------------------------------------
    threshold = 1  #/num_files
    probs = votes

    gz_file = out_dir+'/results-ensemble-th%05f.csv.gz'%threshold
    prob_to_csv(gz_file, names, votes, log, threshold)

def save_origin_mask(): #保存预测的原始mask图片

    is_merge_bn = 1
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-00d'
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'
    if params.my_computer:
        out_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
    else:
        out_dir = '/kaggle_data_results/results/lhc/single/' + params.save_path
    # model_file = out_dir +'/snap/047.pth'  #final
    model_file = out_dir + '/snap/' + params.model_snap   ################################################

    #logging, etc --------------------
    os.makedirs(out_dir+'/out_mask/results',  exist_ok=True)
    os.makedirs(out_dir+'/out_mask/test_mask',  exist_ok=True)
    os.makedirs(out_dir+'/out_mask/train_mask',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.make_mask.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('* model_file=%s\n' % model_file)#############################################



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 4

    if params.save_test:
        test_dataset = KgCarDataset( 'test_100064',  'test',#100064  #3197
                                    #'valid_v0_768',  'train1024x1024',#100064  #3197
                                        transform= [
                                        ],mode='test')
        test_loader  = DataLoader(
                            test_dataset,
                            sampler     = SequentialSampler(test_dataset),
                            batch_size  = batch_size,
                            drop_last   = False,
                            num_workers = 12,
                            pin_memory  = True)

    else:
        test_dataset = KgCarDataset( 'train_5088',  'train',#100064  #3197
                                    #'valid_v0_768',  'train1024x1024',#100064  #3197
                                        transform= [
                                        ],mode='test')
        test_loader  = DataLoader(
                            test_dataset,
                            sampler     = SequentialSampler(test_dataset),
                            batch_size  = batch_size,
                            drop_last   = False,
                            num_workers = 12,
                            pin_memory  = True)

    log.write('\tbatch_size         = %d\n'%batch_size)
    log.write('\ttest_dataset.split = %s\n'%test_dataset.split)
    log.write('\tlen(test_dataset)  = %d\n'%len(test_dataset))
    log.write('\n')


    ## net ----------------------------------------
    net = Net(in_shape=(3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))######################################
    net.cuda()

    num_valid = len(test_dataset)
    names = test_dataset.names

    if params.save_test:
        np.savetxt(out_dir+'/out_mask/test_mask/names.txt', names, fmt='%s')
    else:
        np.savetxt(out_dir+'/out_mask/train_mask/names.txt', names, fmt='%s')

    df = test_dataset.df.set_index('id')


    if is_merge_bn: merge_bn_in_net(net)
    ## start testing now #####
    log.write('start saving masks ...\n')
    start = timer()

    net.eval()
    # probs = predict8_in_blocks( net, test_loader, block_size=CSV_BLOCK_SIZE, save_dir=out_dir+'/submit',log=log)           # 20 min

    time_taken = 0
    end = 0
    num = 0

    test_num = len(test_loader)

    for it, (images, indices) in enumerate(test_loader, 0):
        images  = Variable(images,volatile=True).cuda()
        #labels  = Variable(labels).cuda().half()
        batch_size = len(indices)

        num = num + batch_size

        #forward
        t0 =  timer()
        logits = net(images)
        probs  = F.sigmoid(logits)

        #warm start
        #if it>10:
        #    time_taken = time_taken + timer() - t0
        #    print(time_taken)

        #a = dice_loss((probs.float()>0.5).float(), labels.float(), is_average=False)
        #accs[start:start + batch_size]=a.data.cpu().numpy()

        ## full results ----------------
        probs  = (probs.data.float().cpu().numpy()*255).astype(np.uint8)
        for b in range(batch_size):
            name = names[indices[b]]

            prob = probs[b]
            if params.save_full_resolution_mask == True:
                prob = cv2.resize(prob,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)  #INTER_CUBIC  ##

            if params.save_test:
                cv2.imwrite(out_dir+'/out_mask/test_mask/%s.png'%(name), prob)
            else:
                cv2.imwrite(out_dir+'/out_mask/train_mask/%s.png'%(name), prob)

        pass ######################
        #start = start + batch_size
        print('it: %d, num: %d'%(it,num), end=' ', flush=True)
        if num%1000 == 0:
            log.write(' [it: %d, num: %d] \n'%(it,num))
            log.write('\t time = %0.2f min \n'%((timer() - start)/60))
    
    log.write(' save_masks = %f min\n'%((timer() - start) / 60))
    log.write('\n')
    assert(num == test_num)


def ensamble_th():

    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-peduo-label-00c'
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'
    if params.my_computer:
        out_dir1 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir2 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir3 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir4 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir5 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir6 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir7 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir8 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir9 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
        out_dir10 = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
    else:
        out_dir1 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir2 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir3 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir4 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir5 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir6 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir7 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir8 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir9 = '/kaggle_data_results/results/lhc/single/' + params.save_path
        out_dir10 = '/kaggle_data_results/results/lhc/single/' + params.save_path

    #logging, etc --------------------
    os.makedirs(out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')


    # read names
    # split_file = CARVANA_DIR +'/split/'+ 'valid_v0_768'
    # CARVANA_NUM_BLOCKS =1


    split_file = CARVANA_DIR +'/split/'+ 'test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    num_test = len(names)


    rles=[]
    #num_blocks = int(math.ceil(num_test/CSV_BLOCK_SIZE))
    #print('num_blocks=%d'%num_blocks)
    total_start = timer()
    start = timer()
    for i in range(len(names)): 
        if params.save_test:
            p1 = cv2.imread(out_dir1+'/out_mask/test_mask/%s.png'%(names[i]))
            p2 = cv2.imread(out_dir2+'/out_mask/test_mask/%s.png'%(names[i]))
            p3 = cv2.imread(out_dir3+'/out_mask/test_mask/%s.png'%(names[i]))
            p4 = cv2.imread(out_dir4+'/out_mask/test_mask/%s.png'%(names[i]))
            p5 = cv2.imread(out_dir5+'/out_mask/test_mask/%s.png'%(names[i]))
            p6 = cv2.imread(out_dir6+'/out_mask/test_mask/%s.png'%(names[i]))
            p7 = cv2.imread(out_dir7+'/out_mask/test_mask/%s.png'%(names[i]))
            p8 = cv2.imread(out_dir8+'/out_mask/test_mask/%s.png'%(names[i]))
            p9 = cv2.imread(out_dir9+'/out_mask/test_mask/%s.png'%(names[i]))
            p10 = cv2.imread(out_dir10+'/out_mask/test_mask/%s.png'%(names[i]))
        else:
            p1 = cv2.imread(out_dir1+'/out_mask/train_mask/%s.png'%(names[i]))
            p2 = cv2.imread(out_dir2+'/out_mask/train_mask/%s.png'%(names[i]))
            p3 = cv2.imread(out_dir3+'/out_mask/train_mask/%s.png'%(names[i]))
            p4 = cv2.imread(out_dir4+'/out_mask/train_mask/%s.png'%(names[i]))
            p5 = cv2.imread(out_dir5+'/out_mask/train_mask/%s.png'%(names[i]))
            p6 = cv2.imread(out_dir6+'/out_mask/train_mask/%s.png'%(names[i]))
            p7 = cv2.imread(out_dir7+'/out_mask/train_mask/%s.png'%(names[i]))
            p8 = cv2.imread(out_dir8+'/out_mask/train_mask/%s.png'%(names[i]))
            p9 = cv2.imread(out_dir9+'/out_mask/train_mask/%s.png'%(names[i]))
            p10 = cv2.imread(out_dir10+'/out_mask/train_mask/%s.png'%(names[i]))


        p1 = image_to_tensor(p1)
        p2 = image_to_tensor(p2)
        p3 = image_to_tensor(p3)
        p4 = image_to_tensor(p4)
        p5 = image_to_tensor(p5)
        p6 = image_to_tensor(p6)
        p7 = image_to_tensor(p7)
        p8 = image_to_tensor(p8)
        p9 = image_to_tensor(p9)
        p10 = image_to_tensor(p10)

        p = (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10)/10
        if (i%1000==0):
            end  = timer()
            n = len(rles)
            time = (end - start) / 60
            time_remain = (num_test-n-1)*time/(n+1)
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))
            start = timer()

        prob = cv2.resize(p,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)
        mask = prob>127
        rle  = run_length_encode(mask)
        rles.append(rle)

    names = [name+'.jpg' for name in names]

    dir_name = out_dir.split('/')[-1]
    gz_file  = out_dir + '/submit/results-%s.csv.gz'%dir_name
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})    
    df.to_csv(gz_file, index=False, compression='gzip')

    log.write('\tdf.to_csv time = %f min\n'%((timer() - total_start) / 60)) #3 min
    log.write('\n')

def run_submit2():

    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-peduo-label-00c'
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'
    if params.my_computer:
        out_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
    else:
        out_dir = '/kaggle_data_results/results/lhc/single/' + params.save_path
    #logging, etc --------------------
    os.makedirs(out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')


    # read names
    # split_file = CARVANA_DIR +'/split/'+ 'valid_v0_768'
    # CARVANA_NUM_BLOCKS =1


    split_file = CARVANA_DIR +'/split/'+ 'test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    num_test = len(names)


    rles=[]
    #num_blocks = int(math.ceil(num_test/CSV_BLOCK_SIZE))
    #print('num_blocks=%d'%num_blocks)
    total_start = timer()
    start = timer()
    for i in range(len(names)): 
        if params.save_test:
            p = cv2.imread(out_dir+'/out_mask/test_mask/%s.png'%(names[i]))
        else:
            p = cv2.imread(out_dir+'/out_mask/train_mask/%s.png'%(names[i]))

        if (i%1000==0):
            end  = timer()
            n = len(rles)
            time = (end - start) / 60
            time_remain = (num_test-n-1)*time/(n+1)
            print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))
            start = timer()

        prob = cv2.resize(p,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)
        mask = prob>127
        rle  = run_length_encode(mask)
        rles.append(rle)

    names = [name+'.jpg' for name in names]

    dir_name = out_dir.split('/')[-1]
    gz_file  = out_dir + '/submit/results-%s.csv.gz'%dir_name
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})    
    df.to_csv(gz_file, index=False, compression='gzip')

    log.write('\tdf.to_csv time = %f min\n'%((timer() - total_start) / 60)) #3 min
    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_vote()

    print('\nsucess!')
