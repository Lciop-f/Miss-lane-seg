# Miss Lane Seg
Use light semantic branch to complete the miss lane in lane segment mission.
In 2-classes segment mission, spatial branch is good enough to segment the lane, but is not robust. So we use the extra semantic branch to link up the spatial block to complete the miss part(excessive exposure reasons). 

![test1_gt](https://github.com/user-attachments/assets/f16480db-98c6-4ce7-9e68-68723971d9a4)
![test1_img](https://github.com/user-attachments/assets/927809b4-f2a0-41f1-9d86-7089fceceb10)
![test1_pred1](https://github.com/user-attachments/assets/399ac19d-8ce0-44dc-b0c3-5f08cb0de457)
![test1_pred2](https://github.com/user-attachments/assets/88ed748f-00fe-41d3-9a11-cef32b43635b)
![test3_gt](https://github.com/user-attachments/assets/2832d360-2d3d-4083-8062-9d11a74962ed)
![test3_img](https://github.com/user-attachments/assets/759399ef-bea1-40c0-92e8-19d238f48c0b)
![test3_pred1](https://github.com/user-attachments/assets/f31a7d17-98ca-4bee-9887-e870623f391c)
![test3_pred2](https://github.com/user-attachments/assets/7316587b-72d4-42de-8178-a5a270a93ef9)

As shown in each line, the first picture is the original image, the second picture is the mask picture as well as the input.

The third picture is the spatial branch output without semantic block, the fourth picture is the output with semantic branch.(which performs well on both straight lane and curve lane)
