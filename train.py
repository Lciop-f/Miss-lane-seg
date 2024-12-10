import paddle
from mpmath.identification import transforms
from paddle import nn
from Dataload import MyData_label
from paddle.io import DataLoader
from Model.SCT_Sea_decoder import sctsea
import numpy as np


def train(epochs,mode):
    # if use train_m mode,change your path to mask images
    train_data = MyData_label(["./lane/train/images/"], "./lane/train/annotations/")
    #train_data = MyData_label(["./lane/train/masks/","./lane/train/masks2/"],"./lane/train/annotations/")
    data = DataLoader(train_data,batch_size=5,shuffle=True)
    model = sctsea(mode)
    beta1 = paddle.to_tensor([0.9], dtype="float32")
    beta2 = paddle.to_tensor([0.999], dtype="float32")
    #if train_m mode, use the semantic parameters
    params = model.params()
    #params = list(model.decoder.sem_seg.parameters())
    opt = paddle.optimizer.AdamW(learning_rate=0.0025,beta1=beta1,beta2=beta2, weight_decay=0.01,parameters=params)
    loss_fn = nn.CrossEntropyLoss(axis=1)
    for i in range(epochs):
        model.train()
        total_loss = 0
        for batch_id,train_data in enumerate(data):
            image,label = train_data
            image_mask = np.array(image,dtype="float32")
            image_mask = paddle.to_tensor(image_mask,dtype="float32")
            image_mask = image_mask.transpose([0,3,1,2])
            label = label.unsqueeze(1)
            label = paddle.to_tensor(label,dtype="int64")
            pred = model(image_mask)
            loss = loss_fn(pred,label)
            loss.backward()
            opt.step()
            opt.clear_grad()
            if batch_id % 20 == 0:
                print(f"batchID:{batch_id},loss : {loss}")
            total_loss = loss + total_loss
        print("-------------------------------------")
        print(f"epoch{i}: total loss:{total_loss/batch_id}")
        print("-------------------------------------")
        if i % 5 == 0:
            paddle.save(model.state_dict(),f"./sctsea/epoch{i}.pdparams")


if __name__ == '__main__':
    train(epochs=100,mode="train_s")