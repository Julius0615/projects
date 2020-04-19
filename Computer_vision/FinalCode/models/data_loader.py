#!/usr/bin/env python
# coding: utf-8

# In[14]:


#  Zeruo
# load data and use face recognition to resize the location


import os#,cv2
import random


from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import face_recognition
import dlib
from skimage.draw import polygon
import skimage
import torch
import numpy as np
import torchvision.transforms.functional as tf
import glob

# In[15]:

# TODO transform single
# list random choose
# get item 
# 不传 transform 只穿path
# list里面选一个 getitem
# 
randompick = ['flip','color']
secure_random = random.SystemRandom()

choice = secure_random.choice(randompick)

class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    # TODO: get dir with lable, transform list
    def __init__(self, filenames, labels,transforms=None):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms
        self.label2id = {'HAPPY':0, 'SURPRISE':1, 'DISGUST':2, 'ANGRY':3, 'SAD':4, 'NEUTRAL':5, 'FEAR':6}

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
                Fetch index idx image and labels from dataset. Perform transforms on image.
                Args:
                    idx: (int) index in [0, 1, ..., size_of_dataset-1]
                Returns:
                    image: (Tensor) transformed image
                    label: (int) corresponding label of image
                """
        image = face_recognition.load_image_file(self.filenames[idx])  # PIL image
        W,H = image.shape[:2]
        landmark_image = np.zeros_like(image)
        face_landmarks = face_recognition.face_landmarks(image)[0]
        image = Image. fromarray(image)#cv2.imread(self.filenames[idx]))
        # # TODO: using face_landmark
        face_landmark = [] + face_landmarks["chin"] + \
                        face_landmarks["left_eyebrow"] + \
                        face_landmarks["right_eyebrow"] + \
                        face_landmarks["nose_bridge"] + \
                        face_landmarks["nose_tip"] + \
                        face_landmarks["left_eye"] + \
                        face_landmarks["right_eye"] + \
                        face_landmarks["top_lip"] + \
                        face_landmarks["bottom_lip"]
        for (x,y) in face_landmark:
            landmark_image[min(x,W-1),min(y,H-1),:] = 1.0
        face_landmark = torch.tensor(face_landmark,dtype=torch.float32)/256
        # # TODO: random chosed transformer
        transform = secure_random.choice(self.transforms)
        pil_image = transform(image)
        landmark_image = Image.fromarray(landmark_image)
        landmark_image = transform(landmark_image)
        return {"image": pil_image, "labels": self.label2id[self.labels[idx]],"landmark":face_landmark,"landmarkimage":landmark_image}
        # return {"image":image, "labels":self.labels[idx]}

class AUGSIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    # TODO: get dir with lable, transform list
    def __init__(self, filenames, labels, transforms=None,if_training=True):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        self.filenames = filenames
        self.labels = labels
        self.transforms = [transforms[0]]
        self.mask_resize = transforms[1]
        self.label2id = {'HAPPY':0, 'SURPRISE':1, 'DISGUST':2, 'ANGRY':3, 'SAD':4, 'NEUTRAL':5, 'FEAR':6}
        self.bg_list = list(glob.glob("./coco/*.jpg"))
        print("BG number ",len(self.bg_list))
        self.if_training = if_training
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./aug_data/shape_predictor_68_face_landmarks.dat')

    def __len__(self):
        # return size of dataset
        return len(self.filenames)*40

    def __getitem__(self, idx):
        """
                Fetch index idx image and labels from dataset. Perform transforms on image.
                Args:
                    idx: (int) index in [0, 1, ..., size_of_dataset-1]
                Returns:
                    image: (Tensor) transformed image
                    label: (int) corresponding label of image
                """
        img = dlib.load_rgb_image(self.filenames[idx % len(self.filenames)])  # PIL image

        rect = self.detector(img)[0]
        sp = self.predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = polygon(outline[:, 1], outline[:, 0])
        np.clip(X, 0, 255, out=X)
        np.clip(Y, 0, 255, out=Y)
        #     clip for out range
        cropped_img = np.zeros(img.shape, dtype=np.uint8)
        cropped_img[Y, X] = img[Y, X]

        img_mask = np.zeros(img.shape, dtype=np.uint8)
        img_mask[Y, X] = 1
        if self.if_training:
            #     call trans
            # SAVE TRANS IMG

        # TRANS_IMG
            trans_img_pil,transed_img_mask = self.transFace(cropped_img,img_mask)
            #     Image.fromarray(trans_img).save(outimg_path)

            pil_comb_np = self.appendFace(trans_img_pil)
            pil_comb_pil = Image.fromarray(pil_comb_np)

        # BRIGHTNESS
            brightness_factor = random.uniform(0, 2)
            result_img = tf.adjust_brightness(pil_comb_pil, brightness_factor)
            #         Image.fromarray(pil_comb).save(outimg_path)
            # outimg_path = os.path.join("outimg", emotion_tag, filename + "_f" + str(counter) + "_br" + str(br_ct) + ".jpg")
            # result_img.save(outimg_path)

            result_img = self.transforms[0](result_img)
            transed_img_mask = self.mask_resize(transed_img_mask)
        else:
            result_img =self.transforms[0](Image.fromarray(img))
            transed_img_mask = self.mask_resize(Image.fromarray(img_mask))
        return {"image": result_img, "labels": self.label2id[self.labels[idx%len(self.filenames)]],"landmark_mask":transed_img_mask}
        # return {"image":image, "labels":self.labels[idx]}

    def transFace(self,face_image,img_mask):
        #     eval_transformer = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomAffine((-180, 180), translate=(0.5,0.5), scale=(0.5,1.5))
        #     torchvision.transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406),
        #                          (0.229, 0.224, 0.225)),
        #     ])  # transform it into a torch tensor

        pil_img = Image.fromarray(face_image)
        img_mask = Image.fromarray(img_mask)
        angle = random.randint(-180, 180)
        scale = random.uniform(0.3, 1.5)
        translateH = random.randint(-20, 20)
        translateV = random.randint(-20, 20)

        trans_img = tf.affine(img=pil_img, angle=angle, translate=(translateH, translateV), scale=scale, shear=0)
        img_mask =tf.affine(img=img_mask, angle=angle, translate=(translateH, translateV), scale=scale, shear=0)
        #     torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
        return trans_img,img_mask

    def rand_load_bgimg(self):
        bg_idx = random.randint(0, len(self.bg_list)-1)
        #     bg_list.length()
        bg_path = self.bg_list[bg_idx]
        #     bg_img = dlib.load_rgb_image(bg_path)
        bg_img = Image.open(bg_path)
        #     plt.imshow(bg_img)
        area = (0, 0, 256, 256)
        bg_img = bg_img.crop(area)
        bg_img = tf.to_grayscale(bg_img, num_output_channels=3)
        return bg_img

    def appendFace(self,trans_pil):
        bg_img_pil = self.rand_load_bgimg()
        trans_np = np.array(trans_pil)
        bg_img_np = np.array(bg_img_pil)
        bg_img_np = np.reshape(bg_img_np, (256, 256, 3))
        face_mask = trans_np[:, :, 2]
        face_mask_3_dim = np.reshape(face_mask, (256, 256, 1))
        bg_img_np_mask = bg_img_np * (1 - face_mask_3_dim)
        appended_img = bg_img_np_mask + trans_np
        #     bg_img_pil.paste(trans_pil)
        #     bg_img_pil.show()
        return appended_img

# In[18]:


def fetch_dataloader(types, totalFiles, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        files: dictionary of image path {"train":[xxx,label],"val":[xxx,label],"test":[xxxx,label]}
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    resize = params.resize if hasattr(params,"resize") else 64
    eval_transformer = transforms.Compose([
        transforms.Resize(resize),  # resize the image to 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])  # transform it into a torch tensor
    train_transformers = [transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),])
    ]
    for split in ['train', 'val', 'test']:
        if split in types:
            files,labels = totalFiles[split]
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SIGNSDataset(files, labels,train_transformers), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(SIGNSDataset(files, labels,[eval_transformer]), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders



def fetch_aug_dataloader(types, totalFiles, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        files: dictionary of image path {"train":[xxx,label],"val":[xxx,label],"test":[xxxx,label]}
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    resize = params.resize if hasattr(params,"resize") else 64
    eval_transformer = transforms.Compose([
        transforms.Resize(resize),  # resize the image to 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])  # transform it into a torch tensor
    train_transformers = [transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),])
    ]
    for split in ['train', 'val', 'test']:
        if split in types:
            files,labels = totalFiles[split]
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(AUGSIGNSDataset(files, labels,[eval_transformer,transforms.Compose([
        transforms.Resize(resize),  # resize the image to 64x64
        transforms.ToTensor(),])]), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(AUGSIGNSDataset(files, labels,[eval_transformer,transforms.Compose([
        transforms.Resize(resize),  # resize the image to 64x64
        transforms.ToTensor(),])],if_training=False), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

# In[ ]:




