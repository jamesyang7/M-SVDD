import sys
sys.path.append("/home/iot/collision_detect/svdd/")
from dataloader.svdd_dataloader import CollisionLoader_new
import numpy as np


def load_data():
    # train_imu_path = '/home/iot/collision_detect/data/imu/normal_train'
    # train_audio_path = '/home/iot/collision_detect/data/audio/normal_train'
    #
    # test_imu_path = '/home/iot/collision_detect/data/imu/abnormal/'
    # test_audio_path = '/home/iot/collision_detect/data/audio/abnormal/'
    #
    # test_imu_normal_path = '/home/iot/collision_detect/data/imu/normal_test'
    # test_audio_normal_path = '/home/iot/collision_detect/data/audio/normal_test'
    train_audio_path = '/home/iot/collision_detect/new_data/audio_np/Normal_train'
    train_imu_path = '/home/iot/collision_detect/new_data/imu_np/Normal_train'

    test_audio_path = '/home/iot/collision_detect/new_data/audio_np/Abnormal'
    test_imu_path = '/home/iot/collision_detect/new_data/imu_np/Abnormal'

    test_imu_normal_path = '/home/iot/collision_detect/new_data/imu_np/Normal_test'
    test_audio_normal_path = '/home/iot/collision_detect/new_data/audio_np/Normal_test'

    train_dataset        = CollisionLoader_new(train_imu_path,train_audio_path)
    val_dataset          = CollisionLoader_new(test_imu_path,test_audio_path)
    val_normal_dataset   = CollisionLoader_new(test_imu_normal_path,test_audio_normal_path)


    train_imu,train_audio,train_spec = [],[],[]
    val_imu,val_audio,val_spec = [],[],[]
    val_imu_normal,val_audio_normal,val_spec_normal = [],[],[]
    abnormal_list = val_dataset.audio_list
    normal_list   = val_normal_dataset.audio_list
    total_list = normal_list+abnormal_list

    for i in range(len(train_dataset)):
        spec,imu,audio = train_dataset.__getitem__(i)
        spec,imu,audio = np.transpose(spec.numpy(),(1,2,0)),imu.numpy(),np.transpose(audio.numpy(),(1,0))
        train_spec.append(spec)
        train_imu.append(imu)
        train_audio.append(audio)

    for i in range(len(val_dataset)):
        spec,imu,audio = val_dataset.__getitem__(i)
        spec,imu,audio = np.transpose(spec.numpy(),(1,2,0)),imu.numpy(),np.transpose(audio.numpy(),(1,0))
        val_spec.append(spec)
        val_imu.append(imu)
        val_audio.append(audio)

    for i in range(len(val_normal_dataset)):
        spec,imu,audio = val_normal_dataset.__getitem__(i)
        spec,imu,audio = np.transpose(spec.numpy(),(1,2,0)),imu.numpy(),np.transpose(audio.numpy(),(1,0))
        val_spec_normal.append(spec)
        val_imu_normal.append(imu)
        val_audio_normal.append(audio)

    train_imu,train_audio,train_spec = np.array(train_imu),np.array(train_audio),np.array(train_spec)
    val_imu,val_audio,val_spec = np.array(val_imu),np.array(val_audio),np.array(val_spec)
    val_imu_normal,val_audio_normal,val_spec_normal = np.array(val_imu_normal),np.array(val_audio_normal),np.array(val_spec_normal)

    return train_imu,train_audio,train_spec,val_imu,val_audio,val_spec,val_imu_normal,val_audio_normal,val_spec_normal,total_list
