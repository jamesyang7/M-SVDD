import sys
import numpy as np
sys.path.append('../')
from deepod.models.time_series import DCdetector as model
from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment


train_imu,train_audio,train_spec,val_imu,val_audio,val_spec,val_imu_normal,val_audio_normal,val_spec_normal = load_data()

train_combine = np.concatenate([np.mean(train_audio,axis=-1),train_imu],axis=1)

test_audio_all = np.concatenate([val_audio_normal,val_audio],axis=0)
test_imu_all   = np.concatenate([val_imu_normal,val_imu],axis=0)
test_combine = np.concatenate([np.mean(test_audio_all,axis=-1),test_imu_all],axis=1)

labels_all = np.concatenate([np.zeros(val_audio_normal.shape[0]),np.ones(val_audio.shape[0])])


clf_imu = model(epochs=20,batch_size=4)
clf_imu.fit(train_imu)
clf_audio = model(epochs=20,batch_size=4)
clf_audio.fit(np.mean(train_audio,axis=-1))
clf_all = model(epochs=20,batch_size=4)
clf_all.fit(train_combine)

scores_audio      = clf_audio.decision_function(np.mean(test_audio_all,axis=-1))
scores_imu        = clf_imu.decision_function(test_imu_all)
scores_all        = clf_all.decision_function(test_combine)

#auc, mean precision, F1, percision, recall
eval_metrics_audio = ts_metrics(labels_all, scores_audio)
eval_metrics_imu = ts_metrics(labels_all, scores_imu)
result_audio = eval_metrics_audio
result_imu = eval_metrics_imu
eval_metrics_all = ts_metrics(labels_all, scores_all)
result_all = eval_metrics_all

print(result_audio)
print(result_imu)
print(result_all)