from common import *
from dataset.carvana_cars import *
from model.tool import *

import multiprocessing

def vote_process(threadID, start_, end_, out_dir_=[], names=[], final_out_dir=''):
    for i in range(start_, end_): 
        p = []
        average = np.zeros((CARVANA_H,CARVANA_W),np.uint8)
        for j in range(len(out_dir_)):
            p.append(cv2.imread(out_dir_[j]+'/submit/test_mask/%s.png'%(names[i]),cv2.IMREAD_GRAYSCALE))
            #p.append(cv2.imread(out_dir_[j]+'/post_train/submit/test_mask/%s.png'%(names[i]),cv2.IMREAD_GRAYSCALE))
            p[j] = p[j].astype(np.uint8)
                
            #p[j] = p[j] >= 128
            average += (p[j] > 127)
            
        #average = average/5
        #average = average/len(self.out_dir_)
        average = (average > (len(out_dir_)//2))*255  #change
        cv2.imwrite(final_out_dir+'/submit/test_mask/%s.png'%(names[i]), average.astype(np.uint8))

        if i%5000 == 0:
            print('* threadID = %d, start = %d, end = %d, curr = %d\n' %(threadID,start_,end_,i))

def run_vote_multi_process():
    out_dir_ = []
    for i in range(5):
    #for i in range(0,7):
        out_dir_.append(params.out_dir + params.ensemble_dir + '_k%d'%(i+1))

    #out_dir_.append(params.out_dir + params.ensemble_dir + '_single')
    
    final_out_dir = params.out_dir + params.ensemble_dir + '_'

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

    #**************************************************************
    print('\n** thread start **')

    p1 = multiprocessing.Process(target = vote_process, args = (1,     0, 25000, out_dir_, names, final_out_dir,))
    p2 = multiprocessing.Process(target = vote_process, args = (2, 25000, 50000, out_dir_, names, final_out_dir,))
    p3 = multiprocessing.Process(target = vote_process, args = (3, 50000, 75000, out_dir_, names, final_out_dir,))
    p4 = multiprocessing.Process(target = vote_process, args = (4, 75000,100064, out_dir_, names, final_out_dir,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()

    #**************************************************************

    log.write(' save_masks = %f min\n'%((timer() - total_start) / 60))
    log.write('\n')

if __name__ == "__main__":
    run_vote_multi_process()

    print('\nsucess!')