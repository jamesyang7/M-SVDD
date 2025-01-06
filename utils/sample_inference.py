from sklearn import metrics
import numpy as np

def calculate_normal(train_data,svdd_losses_train,best_threshold):
    normal_list = train_data.audio_list
    indoor_group = []
    outdoor_group = []

    for idx, filepath in enumerate(normal_list):
        parts = filepath.split('/')
        sequence = parts[-2]  # e.g., 'iseq2', 'iseq5', 'seq16'
        file_number = int(parts[-1].split('.')[0])  # Extract file number from filename (e.g., '0' from '0.npy')
        # Check if the file belongs to the indoor group (iseq2, iseq5, or seq16 with 0-104.npy)
        if sequence in ['iseq2', 'iseq5'] or (sequence == 'seq16' and  file_number >= 104):
            indoor_group.append(idx)  # Add the index of the file
        else:
            outdoor_group.append(idx)  # Add the index of the file

    indoor_losses = [svdd_losses_train[idx] for idx in indoor_group]
    outdoor_losses = [svdd_losses_train[idx] for idx in outdoor_group]

    # Given threshold
    threshold = best_threshold
    # Example threshold, replace with actual value

    # True labels for indoor and outdoor groups
    true_labels_indoor = [0] * len(indoor_losses)  # All indoor samples are normal
    true_labels_outdoor = [0] * len(outdoor_losses)  # All outdoor samples are anomalies

    # Combine losses and true labels
    all_losses = indoor_losses + outdoor_losses
    all_true_labels = true_labels_indoor + true_labels_outdoor

    # Calculate predicted labels based on the threshold
    predicted_labels = [1 if score > threshold else 0 for score in all_losses]

    # Calculate accuracy
    correct_predictions = sum(pred == true for pred, true in zip(predicted_labels, all_true_labels))
    total_samples = len(all_true_labels)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # Output accuracy for both groups
    print("Overall Accuracy:")
    print(f"  Accuracy: {accuracy:.4f}")

    # Optional: Calculate accuracy separately for indoor and outdoor groups
    accuracy_indoor = sum(predicted_labels[i] == true_labels_indoor[i] for i in range(len(indoor_losses))) / len(indoor_losses) if len(indoor_losses) > 0 else 0
    accuracy_outdoor = sum(predicted_labels[len(indoor_losses) + i] == true_labels_outdoor[i] for i in range(len(outdoor_losses))) / len(outdoor_losses) if len(outdoor_losses) > 0 else 0

    print("\nIndoor Group Accuracy:")
    print(f"  Accuracy: {accuracy_indoor:.4f}")

    print("\nOutdoor Group Accuracy:")
    print(f"  Accuracy: {accuracy_outdoor:.4f}")
    print(f"-----------------------------------")

def calculate_anomaly(val_data,svdd_losses_val,best_threshold):
    abnormal_list = val_data.audio_list
    mach_group = []
    coli_group = []

    for idx, filepath in enumerate(abnormal_list):
        parts = filepath.split('/')
        sequence = parts[-2]  # e.g., 'iseq2', 'iseq5', 'seq16'
        # file_number = int(parts[-1].split('.')[0])  # Extract file number from filename (e.g., '0' from '0.npy')

        # Check if the file belongs to the indoor group (iseq2, iseq5, or seq16 with 0-104.npy)
        if sequence in ['bl', 'br','fl','fr']:
            mach_group.append(idx)  # Add the index of the file
        else:
            coli_group.append(idx)  # Add the index of the file
            
    mach_losses = [svdd_losses_val[idx] for idx in mach_group]
    coli_losses = [svdd_losses_val[idx] for idx in coli_group]

    # Given threshold
    threshold = best_threshold
    # Example threshold, replace with actual value

    # True labels for indoor and outdoor groups
    true_labels_mach = [1] * len(mach_losses)  # All indoor samples are normal
    true_labels_coli = [1] * len(coli_losses)  # All outdoor samples are anomalies

    # Combine losses and true labels
    all_losses = mach_losses + coli_losses
    all_true_labels = true_labels_mach + true_labels_coli

    # Calculate predicted labels based on the threshold
    predicted_labels = [1 if score > threshold else 0 for score in all_losses]

    # Calculate accuracy
    correct_predictions = sum(pred == true for pred, true in zip(predicted_labels, all_true_labels))
    total_samples = len(all_true_labels)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    # Output accuracy for both groups
    print("Overall Accuracy:")
    print(f"  Accuracy: {accuracy:.4f}")

    # Optional: Calculate accuracy separately for indoor and outdoor groups
    accuracy_mach = sum(predicted_labels[i] == true_labels_mach[i] for i in range(len(mach_losses))) / len(mach_losses) if len(mach_losses) > 0 else 0
    accuracy_coli = sum(predicted_labels[len(mach_losses) + i] == true_labels_coli[i] for i in range(len(coli_losses))) / len(coli_losses) if len(coli_losses) > 0 else 0

    print("\nmach Group Accuracy:")
    print(f"  Accuracy: {accuracy_mach:.4f}")

    print("\ncoli Group Accuracy:")
    print(f"  Accuracy: {accuracy_coli:.4f}")
    print(f"-----------------------------------")


def calculate_acc(labels_all,loss_all,train_data,svdd_losses_train,val_data,svdd_losses_val):
    precision, recall, thresholds= metrics.precision_recall_curve(labels_all, (loss_all))
    epsilon = 1e-10
    f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)
    best_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_index]
    print(f"The thredhold is: {best_threshold}")
    print(f"-----------------------------------")
    calculate_normal(train_data,svdd_losses_train,best_threshold)
    calculate_anomaly(val_data,svdd_losses_val,best_threshold)