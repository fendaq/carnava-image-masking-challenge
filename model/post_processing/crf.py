#https://github.com/milesial/Pytorch-UNet
#https://github.com/milesial/Pytorch-UNet/blob/74ee006a7a4dbde4c9aef16680b1424632c7cfe4/predict.py

import numpy as np
import pydensecrf.densecrf as dcrf

import params

from common import *
from dataset.carvana_cars import *
from train_seg_net import show_batch_results

from model.tool import *
from model.rate import *
from model.segmentation.loss import *
from model.segmentation.blocks import *

Net = params.model_factory

CSV_BLOCK_SIZE = 10000


def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)


    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

def test_valid():

    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'
    #out_dir    = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-peduo-label-00'
    #out_dir    = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-peduo-label-00c'
    #out_dir    = '/root/share/project/kaggle-carvana-cars/results/__old_4__/UNet1024-shallow-01b'
    if params.my_computer:
        out_dir = '/home/lhc/Projects/Kaggle-seg/My-Kaggle-Results/single/' + params.save_path
    else:
        out_dir = '/kaggle_data_results/results/lhc/single/' + params.save_path
    model_file = out_dir +'/snap/final.pth'  #final

    is_results      = True
    is_full_results = True  #True

    #logging, etc --------------------
    if is_results:      shutil.rmtree(out_dir+'/valid/results_by_score',ignore_errors=True)
    if is_full_results: shutil.rmtree(out_dir+'/valid/full_results_by_score',ignore_errors=True)

    os.makedirs(out_dir+'/valid/full_results_by_score',  exist_ok=True)
    os.makedirs(out_dir+'/valid/full_results_by_name',   exist_ok=True)
    os.makedirs(out_dir+'/valid/results_by_score',  exist_ok=True)
    os.makedirs(out_dir+'/valid/results_by_name',   exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), out_dir +'/backup/valid.code.zip')

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')


    batch_size   =  8
    valid_dataset = KgCarDataset(
                                #'train_160', 'train512x512',
                                #'train_v0_4320', 'train512x512',
                                'valid_v0_768',   'train', #'train1024x1024',
                                transform=[
                                ])
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 6,
                        pin_memory  = True)
    ##check_dataset(valid_dataset, valid_loader), exit(0)

    ## net ----------------------------------------
    net = Net(in_shape=(3, CARVANA_HEIGHT, CARVANA_WIDTH))
    net.load_state_dict(torch.load(model_file))
    net.cuda()

    num_valid = len(valid_dataset)
    names = valid_dataset.names
    df = valid_dataset.df.set_index('id')

    full_indices = np.zeros(num_valid,np.float32)
    full_accs    = np.zeros(num_valid,np.float32)
    accs         = np.zeros(num_valid,np.float32)
    net.eval().half()


    time_taken =0
    start=0
    end  =0
    for it, (images, labels, indices) in enumerate(valid_loader, 0):
        images  = Variable(images,volatile=True).cuda().half()
        labels  = Variable(labels).cuda().half()
        batch_size = len(indices)

        #forward
        t0 =  timer()
        logits = net(images)
        probs  = F.sigmoid(logits)

        #warm start
        if it>10:
            time_taken = time_taken + timer() - t0
            #print(time_taken)



        a = dice_loss((probs.float()>0.5).float(), labels.float(), is_average=False)
        accs[start:start + batch_size]=a.data.cpu().numpy()

        if is_results:
            show_batch_results(indices, images, probs, labels=labels, wait=1,
                    out_dir=out_dir+'/valid', mode='both', names=names, df=df, epoch=0, it=0)


        ## full results ----------------
        #probs  = (probs.data.float().cpu().numpy()*255).astype(np.uint8)
        for b in range(batch_size):
            name = names[indices[b]]
            mask_file = CARVANA_DIR + '/annotations/%s/%s_mask.png'%('train',name)
            label = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)

            image_file = CARVANA_DIR + '/images/%s/%s.jpg'%('train',name)
            image = cv2.imread(image_file)

            prob = probs[b]
            prob = cv2.resize(prob,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)  #INTER_CUBIC  ##

            prob_crf = dense_crf(np.array(image).astype(np.unit8), prob)

            prob = (prob.data.float().cpu().numpy()*255).astype(np.uint8)

            score = one_dice_loss_py(prob>127, label>127)
            full_accs   [start+b] = score
            full_indices[start+b] = indices[b]

            if is_full_results:
                meta = df.loc[name[:-3]]
                description = '%d %s %s %s'%( int(meta['year']), meta['make'],  meta['model'], meta['trim2'])
                results = draw_dice_on_image(label, prob)
                draw_shadow_text  (results, '%s.jpg'%(name), (5,30),  1, (255,255,255), 2)
                draw_shadow_text  (results, description, (5,60),  1, (255,255,255), 2)

                #print('full : %0.6f'%score)
                #im_show('results',results,0.33)
                #cv2.waitKey(1)

                cv2.imwrite(out_dir+'/valid/full_results_by_score/%0.5f-%s.png'%(score,name), results)
                cv2.imwrite(out_dir+'/valid/full_results_by_name/%s.png'%(name), results)


        pass ######################
        start = start + batch_size


    #save ----------------
    time_taken = time_taken/60
    accuracy = accs.mean()
    full_accuracy = full_accs.mean()
    print('accuracy (full) = %f (%f)'%(accuracy,full_accuracy))
    print('time_taken min = %f'%(time_taken))
    with open(out_dir+'/valid/full_results-summary.INTER_LINEAR.txt', 'w') as f:
        for n in range(num_valid):
            f.write('%s\t%f\t%f\t%d\n'%(names[n],accs[n],full_accs[n],full_indices[n]))
        f.write('\naccuracy (full) = %f (%f)\n'%(accuracy,full_accuracy))
        f.write('\ntime_taken min = %f\n'%(time_taken))
