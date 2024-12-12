import numpy  as np
import os


detection_dir = '/home/kemove/yyz/MMAUD/challenge_data2/detect/p4s2'
os.makedirs(detection_dir,exist_ok=True)
start_end = [0,2204+1] #+1!!!!!

seq1 = np.arange(397,800+1)
seq2 = np.arange(1168,1383+1)
seq3 = np.arange(1833,2154+1)
seq4 = np.arange(2020,2139+1)

seq_all = np.concatenate([seq1,seq2,seq3])

for i in range(start_end[0],start_end[1]):
    save_path = detection_dir+'/'+str(i)+'.npy'

    if i in seq_all:
        detect = np.array([1])
        np.save(save_path,detect)
    else:
        detect = np.array([0])
        np.save(save_path,detect)
