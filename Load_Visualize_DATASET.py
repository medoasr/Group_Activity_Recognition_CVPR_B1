import  os
import  cv2
import  pandas as pd
from tqdm import tqdm

import  joblib

from BoxInfo import BoxInfo
dataset_root='/media/hossam/New Volume/dl_proj_b1'

def load_players_data(annotation_path,clip_dir):
    with open(annotation_path,'r') as f:
        player_boxes={idx:[] for idx in range(12)}
        frame_boxes_dct={}
        for idx,line in enumerate(f):
            box_info=BoxInfo(line)
            if box_info.player_id>11:
                continue
            player_boxes[box_info.player_id].append(box_info)
        for player_id,boxes_info in player_boxes.items():

            boxes_info=boxes_info[5:]
            boxes_info=boxes_info[:-6]
            # boxes_info=boxes_info[box_info.frame_id==clip_dir]

            for box_info in boxes_info:

                if box_info.frame_id not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_id]=[]
                frame_boxes_dct[box_info.frame_id].append(box_info)
        return  frame_boxes_dct
def visualize_clip(annotation_path,clip_path):
    frame_boxes_dct=load_players_data(annotation_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_id,boxes_info in frame_boxes_dct.items():
        img_path=os.path.join(clip_path,f'{frame_id}.jpg')
        image=cv2.imread(img_path)
        for box_info in boxes_info:
            x1,y1,x2,y2=box_info.box
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(image,box_info.category,(x1,y1-10),fontFace=font,fontScale=.5,color=(0,255,0),thickness=2)

        cv2.imshow('image',image)
        cv2.waitKey(180)
    cv2.destroyAllWindows()

def load_video_annotation(path):
    with open(path,'r')as f:
        clip_cat={}
        for line in f:
            items=line.strip().split(' ')[:2]
            clip_dir=items[0].replace('.jpg','')
            clip_cat[clip_dir]=items[1]
        return  clip_cat

def load_volley_ball_dataset(videos_root,annotation_root):
    videos_dir=os.listdir(videos_root)
    videos_dir.sort()
    videos_annotation={}

    for idx,video_dir in enumerate(videos_dir):
        video_dir_path=os.path.join(videos_root,video_dir)
        if not os.path.isdir(video_dir_path):
            continue
        video_annotation=os.path.join(video_dir_path,'annotation.txt')
        clip_category_dict=load_video_annotation(video_annotation)
        clips_dir=os.listdir(video_dir_path)
        clips_dir.sort()
        clip_annotation={}
        for clip_dir in clips_dir:
            clip_dir_path=os.path.join(video_dir_path,clip_dir)
            if not os.path.isdir(clip_dir_path):
                continue
            annot_file=os.path.join(annotation_root,video_dir,clip_dir,f'{clip_dir}.txt')
            frame_boxes_dct=load_players_data(annot_file,clip_dir)
            clip_annotation[clip_dir]={
                'category':clip_category_dict[clip_dir],
                'frame_boxes_dct':frame_boxes_dct
            }
        videos_annotation[video_dir]=clip_annotation
    return videos_annotation


if __name__=='__main__':

    dic= load_volley_ball_dataset(f'{dataset_root}/videos_sample',f'{dataset_root}/Annotation')

    df=pd.DataFrame(dic)
    pd.set_option('display.max_columns', None)

    print(dic['10'])
    df_10=pd.DataFrame(dic['10'])
    for key in dic.keys():
        for ke1 in dic[key].keys():
            for frame_id in dic[key][ke1]['frame_boxes_dct'].keys():
                pass

                # print(key,ke1,val1['category'])
