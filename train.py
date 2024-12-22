import torch
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import  torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import  numpy as np
from Extract_Featuers import prepare_model
from Load_Visualize_DATASET import load_volley_ball_dataset
from DataLoader import Volly_b1
import  matplotlib.pyplot as plt
import  seaborn as sns
import  pandas as pd
epochs_num=20
seed=33
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model,dataloader,optimizer,criterion,device,epochs):
    model.train()
    for epoch in range(epochs):
        total_loss=0
        avg_loss=[]

        correct=0
        total=0
        for input,label in dataloader:
            input,label=input.to(device),label.to(device)
            optimizer.zero_grad()

            outputs=model(input)
            batch_loss=criterion(outputs,label)

            batch_loss.backward()
            optimizer.step()
            total_loss+=batch_loss.item()

            _,predicted=torch.max(outputs,1)
            correct+=(predicted == label).sum().item()
            total+=label.size(0)
        epoch_loss=total_loss/len(dataloader)
        epoch_acc=100*correct/total
        print(f'Epoch {epoch+1}/{epochs},Loss: {epoch_loss:.4f}, Accuracy_per_epoch: {epoch_acc:.2f}%')
        avg_loss.append(epoch_loss)
def eval_model(model,dataloader,device):
    all_pred=[]
    all_labels=[]
    total=0
    correct=0

    model.eval()

    with torch.no_grad():
        for input,label in dataloader:
            input, label = input.to(device), label.to(device)
            preds = model(input)

            preds=torch.argmax(preds,dim=1).cpu().numpy()
            label=label.cpu().numpy()
            all_pred.extend(preds)
            all_labels.extend(label)
            total += len(label)
            correct += (preds == label).sum().item()
            print(f'Accuracy: {100 * correct/total}%')
        all_pred = np.array(all_pred)
        all_labels = np.array(all_labels)

        accuracy=accuracy_score(all_labels,all_pred)
        cm=confusion_matrix(all_labels,all_pred)
        report_val=classification_report(all_labels,all_pred,zero_division=0)
    return  cm,report_val,accuracy

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_root = '/media/hossam/New Volume/dl_proj_b1'
    train_ids = ["1", "3", "6", "7", "10", "15", "16", "18", "22", "23", "31",
             "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]
    val_ids = ["0", "2", "8","13", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
    model,preprocess=prepare_model(image_level=True)
    videos_root=f'{dataset_root}/videos_sample'

    annotation_root=f'{dataset_root}/Annotation'
    dataset=load_volley_ball_dataset(videos_root,annotation_root)
    train_dataset=Volly_b1(dataset,videos_root,train_ids,transform=preprocess)
    val_dataset=Volly_b1(dataset,videos_root,val_ids,transform=preprocess)
    val_df=Volly_b1(dataset,videos_root,val_ids,transform=preprocess).__to_df__()
    print(val_df.head())

    train_loader=DataLoader(train_dataset, batch_size=9, shuffle=True, num_workers=4)
    val_loader=DataLoader(val_dataset,batch_size=9,shuffle=False)
    crieterion=nn.CrossEntropyLoss()
    print(len(train_loader))
    optimizer=optim.AdamW(model.parameters(),lr=0.001)

    train(model,train_loader,optimizer,crieterion,device,epochs=1)

    cm,report_val,accuracy=eval_model(model,val_loader,device)

    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True,cmap='tab20')
    plt.show()
    plt.close()
    print(report_val)

    print(f"acuracy:{accuracy}")

