import os

import numpy
import  numpy as np
import  torch
import  torch.nn as nn
import  torchvision.models as models
from holoviews import output
from torchvision.models import  ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
from Load_Visualize_DATASET import load_players_data
model_weights_path = '/home/hossam/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'

device='cuda' if torch.cuda.is_available() else 'cpu'
dataset_root='/media/hossam/New Volume/dl_proj_b1'
categories_dct = {
    'l-pass': 0,
    'r-pass': 1,
    'l-spike': 2,
    'r_spike': 3,
    'l_set': 4,
    'r_set': 5,
    'l_winpoint': 6,
    'r_winpoint': 7
}

train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
             "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]

val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]


def prepare_model(image_level=True):
    if image_level:
        preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
    else:
        preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])

    model = models.resnet50(weights=ResNet50_Weights)
    model = nn.Sequential(*list(model.children())[:-1])

    in_features = model.fc.in_features

    print(in_features)


    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.to(device)

    return model, preprocess
def extract_features(clip_dir_path,annotation_file,output_file,model,preprocess,image_level=False):
    frame_boxes=load_players_data(annotation_file)
    with torch.no_grad():
        for frame_id,boxes_info in frame_boxes.items():
            img_path=os.path.join(clip_dir_path,f'{frame_id}.jpg')
            image=Image.open(img_path).convert('RGB')
            if image_level:
                preprocessed_img=preprocess(image).unsqueeze(0)
                preprocessed_img=preprocessed_img.to(device)
                nn_repr=model(preprocessed_img)
                nn_repr=nn_repr.view(1,-1)
            else:
                preprocessed_images = []
                for box_info in boxes_info:
                    x1, y1, x2, y2 = box_info.box
                    cropped_image = image.crop((x1, y1, x2, y2))

                    # if True:  # visualize a crop
                    #     cv2.imshow('Cropped Image', np.array(cropped_image))
                    #     cv2.waitKey(0)

                    preprocessed_images.append(preprocess(cropped_image).unsqueeze(0))

                preprocessed_images = torch.cat(preprocessed_images)
                preprocessed_images=preprocessed_images.to(device)
                dnn_repr = model(preprocessed_images)  # Batch Processing
                dnn_repr = dnn_repr.view(len(preprocessed_images), -1)  # 12 x 2048 for resnet 50

            # np.save(output_file,nn_repr.cpu().numpy())


if __name__ == '__main__':
    image_level=False
    model,preprocess=prepare_model(image_level)
    videos_root=f'{dataset_root}/videos_sample'
    annotation_root=f'{dataset_root}/Annotation'
    output_root=f'{dataset_root}/output'
    videos_dir=os.listdir(videos_root)
    videos_dir.sort()
    for idx ,video_dir in enumerate(videos_dir):
        video_dir_path=os.path.join(videos_root,video_dir)
        if not os.path.isdir(video_dir_path):
            continue
        print(f'{idx}/{len(videos_dir)-3} - Preprocessing dir {video_dir_path}')
        clips_dir=os.listdir(video_dir_path)
        clips_dir.sort()
        for clip_dir in clips_dir:
            clip_dir_path=os.path.join(video_dir_path,clip_dir)
            if not os.path.isdir(clip_dir_path):
                continue
            print(f'-------{clip_dir_path}---------')
            annotation_file=os.path.join(annotation_root,video_dir,clip_dir,f'{clip_dir}.txt')
            output_file=os.path.join(output_root,video_dir)
            if not os.path.exists(output_file):
                os.makedirs(output_file)
            output_file=os.path.join(output_file,f'{clip_dir}.npy')
            extract_features(clip_dir_path,annotation_file,output_file, model, preprocess, image_level)


