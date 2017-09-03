from train_seg_net import *
#from dataset.carvana_cars import *
#import os
#import cv2
#from model.tool import *

def run_guided_filter():

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
    num_blocks = int(math.ceil(num_test/CSV_BLOCK_SIZE))
    print('num_blocks=%d'%num_blocks)
    for i in range(num_blocks):
        start = timer()
        ps   = np.load(out_dir+'/submit/probs-part%02d.8.npy'%i)
        inds = np.loadtxt(out_dir+'/submit/indices-part%02d.8.txt'%i,dtype=np.int32)
        log.write('\tnp.load time = %f min\n'%((timer() - start) / 60))

        M = len(ps)
        for m in range(M):
            if (m%1000==0):
                n = len(rles)
                end  = timer()
                time = (end - start) / 60
                time_remain = (num_test-n-1)*time/(n+1)
                print('rle : b/num_test = %06d/%06d,  time elased (remain) = %0.1f (%0.1f) min'%(n,num_test,time,time_remain))
            #--------------------------------------------------------
            p=ps[m]
            ind=inds[m]

            prob = cv2.resize(p,dsize=(CARVANA_WIDTH,CARVANA_HEIGHT),interpolation=cv2.INTER_LINEAR)
            mask = prob>127
            #----------------insert guided filter--------------------
            guided = cv2.ximgproc.guidedFilter
            cv2.ximgpro
            cv2.imshow("image",img)    
            cv2.imshow("guided filtering",guided)    
            cv2.waitKey()

            #--------------------------------------------------------
            rle  = run_length_encode(mask)
            rles.append(rle)


            #debug
            #if 0:
            if m<10 and i==0:
                name = names[ind]
                img_file = CARVANA_DIR + '/images/test/%s.jpg'%(name)
                #img_file = CARVANA_DIR + '/images/train/%s'%(name)
                image = cv2.imread(img_file)
                results = make_results_image(image, label=None, prob=prob)
                im_show('results',results,0.33)
                im_show('prob',prob,0.33)
                im_show('p',p,0.33)
                cv2.waitKey(1)

    #-----------------------------------------------------
    start = timer()
    names = [name+'.jpg' for name in names]

    dir_name = out_dir.split('/')[-1]
    gz_file  = out_dir + '/submit/results-%s.csv.gz'%dir_name
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')

    log.write('\tdf.to_csv time = %f min\n'%((timer() - start) / 60)) #3 min
    log.write('\n')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
    #run_submit1()
    #run_submit2()

    #run_valid()
    print('\nsucess!')