import glob
import numpy as np
import os
import shutil
import sys

def extract_train_imgs(train_dir, labels_path, output_dir):
    matching_label_pairs = extract_class_names(train_dir, labels_path)
    for img_dir, class_name in matching_label_pairs:
        print(img_dir, class_name)
        output_path = '%s/%s' % (output_dir, img_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img_paths = sorted(glob.glob(os.path.join(train_dir, img_dir) + '/images/*'))
        for i, img_path in enumerate(img_paths):
            shutil.copyfile(img_path, '%s/%s/%s_%d.jpg' % (output_dir, img_dir, img_dir, i))

def extract_val_imgs(input_dir, train_dir, labels_path, val_labels_path, output_dir):
    matching_label_pairs = extract_class_names(train_dir, labels_path)
    val_labels = \
        np.loadtxt(val_labels_path, delimiter='	', dtype=np.object)
    for img_dir, class_name in matching_label_pairs:
        print(img_dir, class_name)
        output_path = '%s/%s' % (output_dir, img_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for i, img_path in enumerate(val_labels[val_labels[:, 1] == img_dir][:, 0]):
            full_img_path = '%s/%s' % (input_dir, img_path)
            target_path = '%s/%s/%s_%d.jpg' % (output_dir, img_dir, img_dir, i)
            shutil.copyfile(full_img_path, target_path)

def extract_class_names(train_dir, labels_path):
    train_class_dirs = sorted(glob.glob(train_dir + '/*'))
    dirs_set = set([s.split('/')[-1] for s in train_class_dirs])
    all_labels = np.loadtxt(labels_path, delimiter='	', dtype=np.object)
    matching_label_pairs = filter(lambda x: x[0] in dirs_set, all_labels)
    return matching_label_pairs

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'list_class_names':
        train_dir = sys.argv[2]
        labels_path = sys.argv[3]
        matching_label_pairs = extract_class_names(train_dir, labels_path)
        print([x[1] for x in matching_label_pairs])
    elif mode == 'extract_train_imgs':
        train_dir = sys.argv[2]
        labels_path = sys.argv[3]
        output_dir = sys.argv[4]
        extract_train_imgs(train_dir, labels_path, output_dir)
    elif mode == 'extract_val_imgs':
        input_dir = sys.argv[2]
        train_dir = sys.argv[3]
        labels_path = sys.argv[4]
        val_labels_path = sys.argv[5]
        output_dir = sys.argv[6]
        extract_val_imgs(input_dir, train_dir, labels_path,
                         val_labels_path, output_dir)
