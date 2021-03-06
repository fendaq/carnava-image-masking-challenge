from common import *
from dataset.carvana_cars import *
from model.tool import *

import multiprocessing
import lmdb
#import six
#from PIL import Image

def ensamble_thread(threadID, start_, end_, out_dir_=[], names=[], final_out_dir='', use_lmdb=True):

    if use_lmdb:
        env = []
        for i in range(len(out_dir_)):
            env.append(lmdb.open(out_dir_[i]+'/submit/test_lmdb'))
    
    for i in range(start_, end_):
        p = []
        average = np.zeros((CARVANA_H,CARVANA_W),np.uint16)
        for j in range(len(out_dir_)):
            if use_lmdb:
                with env[j].begin(write=False) as txn:
                    imgbuf = txn.get(names[i].encode())
                    #buf = six.BytesIO();buf.write(imgbuf);buf.seek(0)
                    #p.append(np.array(Image.open(buf).convert('L')))
                    buf2 = np.fromstring(imgbuf, np.uint8)
                    p.append(cv2.imdecode(buf2, cv2.IMREAD_GRAYSCALE))
            else:
                p.append(cv2.imread(out_dir_[j]+'/submit/test_mask/%s.png'%(names[i]),cv2.IMREAD_GRAYSCALE))
        
            p[j] = p[j].astype(np.uint8)
            average += p[j]

        #average = average/5
        average = average/len(out_dir_)
        cv2.imwrite(final_out_dir+'/submit/test_mask/%s.png'%(names[i]), average.astype(np.uint8))

        #debug
        if 0:
            print('\r threadID = %d, start = %d, end = %d, curr = %d, next_img_id = %s' %(threadID,start_,end_,i,names[i+1]),end='\n',flush=True)
        
        if i%5000 == 0:
            print('* threadID = %d, start = %d, end = %d, curr = %d' %(threadID,start_,end_,i))

def ensemble_png_multi_process():
    out_dir_ = []
    '''
    for i in range(5):
        out_dir_.append(params.out_dir + 'UNet1024_ASPP_08_k%d'%(i+1))
    '''
    # out_dir_.append(params.out_dir + 'UNet1024_ASPP_08_single')

    '''
    for i in range(5):
        out_dir_.append(params.out_dir + 'UNet1024_ASPP_08_k%d/post_train'%(i+1))
    '''
    for i in range(7):
        out_dir_.append(params.out_dir + 'UNet1024_GCN_06_k%d'%(i+1))
    '''
    out_dir_.append(params.out_dir + 'UNet1024_GCN_06_single')
    
    out_dir_.append(params.out_dir + 'UNet1024_ASPP_08_post_train_no_src')
    out_dir_.append(params.out_dir + 'UNet1024_ASPP_08_ens0')
    out_dir_.append(params.out_dir + 'UNet1024_GCN_06_ens0')
    '''
    #final_out_dir = params.out_dir + 'params.ensemble_dir + '_ens2'
    final_out_dir = params.out_dir + 'test'


    #logging, etc --------------------
    os.makedirs(final_out_dir+'/submit/results',  exist_ok=True)
    os.makedirs(final_out_dir + '/submit/test_mask', exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), final_out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(final_out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some project setting **\n')
    for i in range(len(out_dir_)):
        log.write('*' + out_dir_[i] + '\n')

    split_file = CARVANA_DIR +'/split/'+ 'test_100064'
    with open(split_file) as f:
        names = f.readlines()
    names = [name.strip()for name in names]
    num_test = len(names)

    total_start = timer()
    start = timer()

    print('\n** thread start **')

    thread1 = multiprocessing.Process(target = ensamble_thread, args=(1,     0, 25000, out_dir_, names, final_out_dir))
    thread2 = multiprocessing.Process(target = ensamble_thread, args=(2, 25000, 50000, out_dir_, names, final_out_dir))
    thread3 = multiprocessing.Process(target = ensamble_thread, args=(3, 50000, 75000, out_dir_, names, final_out_dir))
    thread4 = multiprocessing.Process(target = ensamble_thread, args=(4, 75000,100064, out_dir_, names, final_out_dir))
    
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()


    log.write(' save_masks = %f min\n'%((timer() - total_start) / 60))
    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #ensamble_png_custom()    
    ensemble_png_multi_process()

    print('\nsucess!')
