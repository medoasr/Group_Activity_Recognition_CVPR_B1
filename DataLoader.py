from PIL import  Image

from torch.utils.data import Dataset
import  os

import  pandas as pd
class Volly_b1(Dataset):
    def __init__(self,dataset,videos_path,indices,transform=None):
        self.dataset=dataset
        self.videos_path=videos_path
        self.indices=indices
        self.dataset={key:value for key ,value in dataset.items() if key in self.indices}
        self.transform=transform

        self.indices=[(video_index, clip_id, frame_id,clip['category']) for video_index in self.dataset.keys()
                        for clip_id,clip in self.dataset[video_index].items()
                        for frame_id in self.dataset[video_index][clip_id]['frame_boxes_dct'].keys()]

        self.classes = set(clip['category'] for video_index in self.dataset.keys() for clip_id, clip in
                           self.dataset[video_index].items())
        self.classes=sorted(self.classes)
        self.class_to_index={clip_class:idx for idx,clip_class in enumerate (self.classes)}



    def __to_df__(self):
        indices=self.indices
        self.df=pd.DataFrame(indices,columns=['Video','Frame_id','Clip_id','Category'])
        # self.df['Category']=self.df['Category'].map(self.class_to_index)
        # print(self.df)
        return self.df




    def __len__ (self):
            return len(self.indices)

    def __getitem__ (self, idx):
            # Get the (video_index, clip_id, frame_id,category) tuple for the given index
            video_index, clip_id, frame_id,category = self.indices[idx]

            # Get the class of the clip (label) and map to an integer index
            clip_class = category
            clip_class_idx = self.class_to_index[clip_class]

            # Convert clip_class to a tensor
            # clip_class_idx = torch.tensor(clip_class_idx, dtype=torch.long)

            # Construct the path to the frame image
            frame_path = os.path.join(self.videos_path, video_index, clip_id, f"{frame_id}.jpg")

            # Load the frame image
            frame = Image.open(frame_path)

            if self.transform:
                frame = self.transform(frame)

            # Return the frame image and its integer class label
            return frame, clip_class_idx