import params
import torch
from common import *

from dataset.carvana_cars import *

from model.tool import *
from model.rate import *
from model.segmentation.loss import *
from model.segmentation.blocks import *


Net = params.model_factory

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
    #backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir+'/log.MSC_infer.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('* model_file=%s\n' % model_file)



    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 1

    test_dataset = KgCarDataset_MSC_infer( 'test_100064',  'test',#100064  #3197
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

    for it, (images0, images1, images2, indices) in enumerate(test_loader, 0):

        images0  = Variable(images0,volatile=True).cuda().half()
        images1  = Variable(images1,volatile=True).cuda().half()
        images2  = Variable(images2,volatile=True).cuda().half()
       
        batch_size = len(indices)

        num = num + batch_size

        #forward
        t0 =  timer()

        logits0 = net(images0)
        probs0  = F.sigmoid(logits0)

        logits1 = net(images1)
        probs1  = F.sigmoid(logits1)

        logits2 = net(images2)
        probs2  = F.sigmoid(logits2)
      
        probs1 = (probs1[0].data.float().cpu().numpy()*255)
        probs1 = cv2.resize(probs1, dsize=(params.input_w, params.input_h), interpolation=cv2.INTER_LINEAR)
        probs1 = probs1.astype(np.float32)/255
        probs1 = torch.from_numpy(probs1)
        probs1 = torch.unsqueeze(probs1, dim = 0)
        probs1 = Variable(probs1,volatile=True).cuda().half()
        
        
        probs2 = (probs2[0].data.float().cpu().numpy()*255)
        probs2 = cv2.resize(probs2, dsize=(params.input_w, params.input_h), interpolation=cv2.INTER_LINEAR)
        probs2 = probs2.astype(np.float32)/255
        probs2 = torch.from_numpy(probs2)
        probs2 = torch.unsqueeze(probs2, dim = 0)
        probs2 = Variable(probs2, volatile=True).cuda().half()


        #warm start
        #if it>10:
        #    time_taken = time_taken + timer() - t0
        #    print(time_taken)

        #a = dice_loss((probs.float()>0.5).float(), labels.float(), is_average=False)
        #accs[start:start + batch_size]=a.data.cpu().numpy()

        ## full results ----------------
        a = dice_loss((probs1.float()>0.5).float(), (probs0.float()>0.5).float(), is_average=True).data.cpu().float().numpy()
        b = dice_loss((probs2.float()>0.5).float(), (probs0.float()>0.5).float(), is_average=True).data.cpu().float().numpy()

        if (a<0.9969 or b<0.9969):
            for b in range(batch_size): #batch = 1
                name = names[indices[b]]
                prob = (probs0.data.float().cpu().numpy()*255).astype(np.uint8)[b]
                #prob1 = (probs1.data.float().cpu().numpy() * 255).astype(np.uint8)[b]
                cv2.imwrite(out_dir+'/MSC_infer/%s.png'%(name), prob)
                #cv2.imwrite(out_dir + '/MSC_infer/%s512.png' % (name), prob1)
        print('\r it: %d, num: %d'%(it,num), end=' ', flush=True)
        if num%8000 == 0:
            log.write(' [it: %d, num: %d] \n'%(it,num))
            log.write('\t time = %0.2f min \n'%((timer() - start)/60))
    
    log.write(' save_masks = %f min\n'%((timer() - start) / 60))
    log.write('\n')
    assert(num == test_num)


def MSC_infer_ensemble():  # msc检测坏测试数据，参考run_valid

    is_merge_bn = 1
    # out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-00d'
    # out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'
    out_dir = params.out_dir + params.save_path

    # model_file = out_dir +'/snap/060.pth'  #final
    model_file = out_dir + '/snap/' + params.model_snap

    # logging, etc --------------------
    os.makedirs(out_dir + '/MSC_infer', exist_ok=True)
    os.makedirs(out_dir + '/MSC_infer/results', exist_ok=True)
    os.makedirs(out_dir + '/MSC_infer/bad_results', exist_ok=True)
    # backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(out_dir + '/MSC_infer/log.MSC_infer.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('* model_file=%s\n' % model_file)

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size = 1

    test_dataset = KgCarDataset_ensemble('test_100064', 'test',  # 100064  #3197
                                          # 'valid_v0_768',  'train1024x1024',#100064  #3197
                                          transform=[
                                          ], mode='train')
    test_loader = DataLoader(
        test_dataset,
        sampler=RandomSamplerWithLength(test_dataset,100064),
        batch_size=batch_size,
        drop_last=False,
        num_workers=12,
        pin_memory=True)

    log.write('\tbatch_size         = %d\n' % batch_size)
    log.write('\ttest_dataset.split = %s\n' % test_dataset.split)
    log.write('\tlen(test_dataset)  = %d\n' % len(test_dataset))
    log.write('\n')

    ## net ----------------------------------------
    net = Net(in_shape=(3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))
    net.cuda().half()

    # num_valid = len(test_dataset)
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
    min = 1.0
    max = 0.0

    bad_img = 0
    good_img = 0

    good_list = []
    for it, (images0, label, indices) in enumerate(test_loader, 0):

        images0 = Variable(images0, volatile=True).cuda().half()

        batch_size = len(indices)

        num = num + batch_size

        # forward
        t0 = timer()

        logits0 = net(images0)
        probs0 = F.sigmoid(logits0)

        label = Variable(label).cuda().half()
        # warm start
        # if it>10:
        #    time_taken = time_taken + timer() - t0
        #    print(time_taken)

        # a = dice_loss((probs.float()>0.5).float(), labels.float(), is_average=False)
        # accs[start:start + batch_size]=a.data.cpu().numpy()

        ## full results ----------------
        a = dice_loss((probs0.float() > 0.5).float(), (label.float() > 0.5).float(),
                      is_average=True).data.cpu().float().numpy()
        if min>a:
            min = a
        if max<a:
            max = a
        if (a < 0.9965):
            for b in range(batch_size):  # batch = 1
                name = names[indices[b]]
                prob = (probs0.data.float().cpu().numpy() * 255).astype(np.uint8)[b]
                label_w = (label.data.float().cpu().numpy() * 255).astype(np.uint8)[b]
                #image_w = (images0.data.float().cpu().numpy() * 255).astype(np.uint8)[b]
                image_w = tensor_to_image(images0.data.float().cpu()[b]*255)
                cv2.imwrite(out_dir + '/MSC_infer/bad_results/%s.png' % (name), prob)
                cv2.imwrite(out_dir + '/MSC_infer/bad_results/%s_origin.jpg' % (name), image_w)
                cv2.imwrite(out_dir + '/MSC_infer/bad_results/%s_ensemble.jpg' % (name), label_w)
                bad_img += 1

        if (a > 0.99962):
            good_img +=1
            name = names[indices[0]]
            good_list.append(name)

        print('\r it: %d, num: %d, min = %0.6f, max = %0.6f, bad_img_num = %d, good_img_num = %d' \
              % (it, num, min, max, bad_img, good_img), end=' ', flush=True)

        if num % 8000 == 0:
            log.write(' [it: %d, num: %d] \n' % (it, num))
            log.write('\t time = %0.2f min \n' % ((timer() - start) / 60))

        #if num > 1000: break

    file = out_dir + '/MSC_infer/good_result.txt'

    with open(file, 'w') as f:
        for name_ in good_list:
            f.write('%s\n' % (name_))

    log.write(' save_masks = %f min\n' % ((timer() - start) / 60))
    log.write(' min = %d, max = %d, bad_img_num = %d, good_img_num = %d' % (min,max,bad_img,good_img))
    log.write('\n')
    #assert (num == test_num)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    MSC_infer_ensemble()
    print('\nsucess!')
