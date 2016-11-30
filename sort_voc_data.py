import os, time, glob

def sort_data():
    numFiles = len([name for name in os.listdir('../VOC/')])
    done = 0
    
    with open('../data/cat.txt','r') as f:
        for line in f:
            name_label = line.split()
            imagefile = name_label[0] + '.jpg'
            label = name_label[1]
            
            if os.path.isfile('../VOC/' + imagefile): 
                if label == '1':
                    os.rename('../VOC/' + imagefile, "../VOC_POSITIVES/" + imagefile)
                elif label == '-1':
                    os.rename('../VOC/' + imagefile, "../VOC_NEGATIVES/" + imagefile)
                
                if get_time() % 100 == 0:
                    print('processed...' + str(done) + ' of ' + str(numFiles))
                
                done += 1
                
def count_negatives():
    print(len([name for name in os.listdir('../VOC_NEGATIVES/')]))
    
def count_cat_data():
    print(len([name for name in glob.glob('../CAT_DATASET/*.jpg')]))
            
def get_time():
    return int(round(time.time() * 1000))

count_cat_data()
