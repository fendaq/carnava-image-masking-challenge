import params
import torch

def MSC_infer(): #msc检测坏测试数据，参考run_valid

    is_merge_bn = 1
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-00d'
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'
    out_dir = params.out_dir + params.save_path

    # model_file = out_dir +'/snap/060.pth'  #final
    model_file = out_dir + '/snap/' + params.model_snap

    #logging, etc --------------------
    os.makedirs(out_dir+'/MSC_infer',  exist_ok=True)
    os.makedirs(out_dir+'/MSC_infer/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.MSC_infer.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('* model_file=%s\n' % model_file)



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 1

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

    log.write('\tbatch_size         = %d\n'%batch_size)
    log.write('\ttest_dataset.split = %s\n'%test_dataset.split)
    log.write('\tlen(test_dataset)  = %d\n'%len(test_dataset))
    log.write('\n')


    ## net ----------------------------------------
    net = Net(in_shape=(3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))
    net.cuda().half()

    #num_valid = len(test_dataset)
    names = test_dataset.names
    df = test_dataset.df.set_index('id')

    if is_merge_bn: merge_bn_in_net(net)
    ## start testing now #####
    log.write('start prediction ...\n')
    start = timer()

    net.eval()

    time_taken = 0
    end = 0
    num = 0

    test_num = len(test_loader)

    for it, (images, indices) in enumerate(test_loader, 0):

        images1 = torch.squeeze(images,dim=1)
        images1 = (images1.data.float().cpu().numpy()*255).astype(np.uint8)
        images1 = cv2.cvtColor(images1)
        images1 = cv2.resize(images1,(512,512))
        images1 = images1.astype(np.float32)/255
        images1 = torch.from_numpy(images1)
        images1.unsqueeze(0)
        
        images2 = torch.squeeze(images,dim=1)
        images2 = (images2.data.float().cpu().numpy()*255).astype(np.uint8)
        images2 = cv2.cvtColor(images2)
        images2 = cv2.resize(images2,(1600,1600))
        images2 = images2.astype(np.float32)/255
        images2 = torch.from_numpy(images2)
        images2.unsqueeze(0)

        images  = Variable(images,volatile=True).cuda()
        images1  = Variable(images1,volatile=True).cuda()
        images2  = Variable(images2,volatile=True).cuda()
        
        #labels  = Variable(labels).cuda().half()
        batch_size = len(indices)

        num = num + batch_size

        #forward
        t0 =  timer()

        logits0 = net(images)
        probs0  = F.sigmoid(logits0)

        logits1 = net(images1)
        probs1  = F.sigmoid(logits1)

        logits2 = net(images2)
        probs2  = F.sigmoid(logits2)

        probs1 = torch.squeeze(probs1,dim=1)
        probs1 = (probs1.data.float().cpu().numpy()*255).astype(np.uint8)
        probs1 = cv2.cvtColor(probs1)
        probs1 = cv2.resize(probs1,(params.input_w,params.input_h))
        probs1 = probs1.astype(np.float32)/255
        probs1 = torch.from_numpy(probs1)
        probs1.unsqueeze(0).cuda()
        
        probs2 = torch.squeeze(probs2,dim=1)
        probs2 = (probs2.data.float().cpu().numpy()*255).astype(np.uint8)
        probs2 = cv2.cvtColor(probs2)
        probs2 = cv2.resize(probs2,(params.input_w,params.input_h))
        probs2 = probs2.astype(np.float32)/255
        probs2 = torch.from_numpy(probs2)
        probs2.unsqueeze(0).cuda()

        #warm start
        #if it>10:
        #    time_taken = time_taken + timer() - t0
        #    print(time_taken)

        #a = dice_loss((probs.float()>0.5).float(), labels.float(), is_average=False)
        #accs[start:start + batch_size]=a.data.cpu().numpy()

        ## full results ----------------
        a = dice_loss((probs0.float()>0.5).float(), (probs1.float()>0.5).float(), is_average=True)
        b = dice_loss((probs1.float()>0.5).float(), (probs2.float()>0.5).float(), is_average=True)

        if (a<0.9969 or b<0.9969):
            for b in range(batch_size): #batch = 1
                name = names[indices[b]]
                prob = probs0[b]
                
                cv2.imwrite(out_dir+'/MSC_infer/%s.png'%(name), prob)

        print('\r it: %d, num: %d'%(it,num), end=' ', flush=True)
        if num%8000 == 0:
            log.write(' [it: %d, num: %d] \n'%(it,num))
            log.write('\t time = %0.2f min \n'%((timer() - start)/60))
    
    log.write(' save_masks = %f min\n'%((timer() - start) / 60))
    log.write('\n')
    assert(num == test_num)