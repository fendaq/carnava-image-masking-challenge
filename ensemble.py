from common import *
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

def ensamble_png():

    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-peduo-label-00c'
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'

    out_dir_ = []
    for i in range(0,5):
        out_dir_.append(params.out_dir + params.ensemble_dir + '_k%d'%(i+1))

    final_out_dir = params.out_dir + params.ensemble_dir

    #logging, etc --------------------
    os.makedirs(final_out_dir+'/submit/results',  exist_ok=True)
    os.makedirs(final_out_dir + '/submit/test_mask', exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), final_out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(final_out_dir+'/log.submit.txt',mode='a')
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
        p = []
        average = np.zeros((CARVANA_H,CARVANA_W),np.uint16)
        for j in range(0,5):
            p.append(cv2.imread(out_dir_[j]+'/out_mask/test_mask/%s.png'%(names[i]),cv2.IMREAD_GRAYSCALE))
            p[j] = p[j].astype(np.uint8)
            
            average += p[j]
        
        average = average/5
        
        cv2.imwrite(final_out_dir+'/submit/test_mask/%s.png'%(names[i]), average.astype(np.uint8))

        if i%1000 == 0:
            log.write(' [num: %d] \n'%(i))
            log.write('\t time = %0.2f min \n'%((timer() - start)/60))
            start = timer()
    
    log.write(' save_masks = %f min\n'%((timer() - total_start) / 60))
    log.write('\n')

def run_submit_ensemble():

    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet512-peduo-label-00c'
    #out_dir = '/root/share/project/kaggle-carvana-cars/results/single/UNet1024-peduo-label-01c'

    final_out_dir = params.out_dir + params.ensemble_dir

    #logging, etc --------------------
    os.makedirs(final_out_dir+'/submit/results',  exist_ok=True)
    backup_project_as_zip( os.path.dirname(os.path.realpath(__file__)), final_out_dir +'/backup/submit.code.zip')

    log = Logger()
    log.open(final_out_dir+'/log.submit.txt',mode='a')
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
    total_start = timer()
    start = timer()
    for i in range(len(names)):
        #test
        p = cv2.imread(final_out_dir+'/submit/test_mask/%s.png'%(names[i]),cv2.IMREAD_GRAYSCALE)
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
    #-----------------------------------------------------
    names = [name+'.jpg' for name in names]

    dir_name = final_out_dir.split('/')[-1]
    gz_file  = final_out_dir + '/submit/results-%s.csv.gz'%dir_name
    df = pd.DataFrame({ 'img' : names, 'rle_mask' : rles})
    df.to_csv(gz_file, index=False, compression='gzip')

    log.write('\tdf.to_csv time = %f min\n'%((timer() - total_start) / 60)) #3 min
    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #ensamble_png()
    run_submit_ensemble()

    print('\nsucess!')
