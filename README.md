# yolov7-pose
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806). 

## Dataset preparison

[[Keypoints Labels of MS COCO 2017]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip)

## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

``` shell
python train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 960 --kpt-label --sync-bn --device 0 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
```

## Deploy
TensorRT:[https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

## Testing

[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

``` shell
python test.py --data data/coco_kpts.yaml --img 960 --conf 0.001 --iou 0.65 --weights yolov7-w6-pose.pt --kpt-label
```

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>

# train param
```

                 from  n    params  module                                  arguments                     
  0                -1  1         0  models.common.ReOrg                     []                            
  1                -1  1      7040  models.common.Conv                      [12, 64, 3, 1]                
  2                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  3                -1  1      8320  models.common.Conv                      [128, 64, 1, 1]               
  4                -2  1      8320  models.common.Conv                      [128, 64, 1, 1]               
  5                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
  6                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
  7                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
  8                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
  9  [-1, -3, -5, -6]  1         0  models.common.Concat                    [1]                           
 10                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 11                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
 12                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 13                -2  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 14                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 15                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 16                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 17                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 18  [-1, -3, -5, -6]  1         0  models.common.Concat                    [1]                           
 19                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 20                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
 21                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 22                -2  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 23                -1  1    590336  models.common.Conv                      [256, 256, 3, 1]              
 24                -1  1    590336  models.common.Conv                      [256, 256, 3, 1]              
 25                -1  1    590336  models.common.Conv                      [256, 256, 3, 1]              
 26                -1  1    590336  models.common.Conv                      [256, 256, 3, 1]              
 27  [-1, -3, -5, -6]  1         0  models.common.Concat                    [1]                           
 28                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
 29                -1  1   3540480  models.common.Conv                      [512, 768, 3, 2]              
 30                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 31                -2  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 32                -1  1   1327872  models.common.Conv                      [384, 384, 3, 1]              
 33                -1  1   1327872  models.common.Conv                      [384, 384, 3, 1]              
 34                -1  1   1327872  models.common.Conv                      [384, 384, 3, 1]              
 35                -1  1   1327872  models.common.Conv                      [384, 384, 3, 1]              
 36  [-1, -3, -5, -6]  1         0  models.common.Concat                    [1]                           
 37                -1  1   1181184  models.common.Conv                      [1536, 768, 1, 1]             
 38                -1  1   7079936  models.common.Conv                      [768, 1024, 3, 2]             
 39                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
 40                -2  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
 41                -1  1   2360320  models.common.Conv                      [512, 512, 3, 1]              
 42                -1  1   2360320  models.common.Conv                      [512, 512, 3, 1]              
 43                -1  1   2360320  models.common.Conv                      [512, 512, 3, 1]              
 44                -1  1   2360320  models.common.Conv                      [512, 512, 3, 1]              
 45  [-1, -3, -5, -6]  1         0  models.common.Concat                    [1]                           
 46                -1  1   2099200  models.common.Conv                      [2048, 1024, 1, 1]            
 47                -1  1   7609344  models.common.SPPCSPC                   [1024, 512, 1]                
 48                -1  1    197376  models.common.Conv                      [512, 384, 1, 1]              
 49                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 50                37  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 51          [-1, -2]  1         0  models.common.Concat                    [1]                           
 52                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 53                -2  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 54                -1  1    663936  models.common.Conv                      [384, 192, 3, 1]              
 55                -1  1    332160  models.common.Conv                      [192, 192, 3, 1]              
 56                -1  1    332160  models.common.Conv                      [192, 192, 3, 1]              
 57                -1  1    332160  models.common.Conv                      [192, 192, 3, 1]              
 58[-1, -2, -3, -4, -5, -6]  1         0  models.common.Concat                    [1]                           
 59                -1  1    590592  models.common.Conv                      [1536, 384, 1, 1]             
 60                -1  1     98816  models.common.Conv                      [384, 256, 1, 1]              
 61                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 62                28  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 63          [-1, -2]  1         0  models.common.Concat                    [1]                           
 64                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 65                -2  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 66                -1  1    295168  models.common.Conv                      [256, 128, 3, 1]              
 67                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 68                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 69                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 70[-1, -2, -3, -4, -5, -6]  1         0  models.common.Concat                    [1]                           
 71                -1  1    262656  models.common.Conv                      [1024, 256, 1, 1]             
 72                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 73                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 74                19  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 75          [-1, -2]  1         0  models.common.Concat                    [1]                           
 76                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 77                -2  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 78                -1  1     73856  models.common.Conv                      [128, 64, 3, 1]               
 79                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
 80                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
 81                -1  1     36992  models.common.Conv                      [64, 64, 3, 1]                
 82[-1, -2, -3, -4, -5, -6]  1         0  models.common.Concat                    [1]                           
 83                -1  1     65792  models.common.Conv                      [512, 128, 1, 1]              
 84                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
 85          [-1, 71]  1         0  models.common.Concat                    [1]                           
 86                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 87                -2  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 88                -1  1    295168  models.common.Conv                      [256, 128, 3, 1]              
 89                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 90                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 91                -1  1    147712  models.common.Conv                      [128, 128, 3, 1]              
 92[-1, -2, -3, -4, -5, -6]  1         0  models.common.Concat                    [1]                           
 93                -1  1    262656  models.common.Conv                      [1024, 256, 1, 1]             
 94                -1  1    885504  models.common.Conv                      [256, 384, 3, 2]              
 95          [-1, 59]  1         0  models.common.Concat                    [1]                           
 96                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 97                -2  1    295680  models.common.Conv                      [768, 384, 1, 1]              
 98                -1  1    663936  models.common.Conv                      [384, 192, 3, 1]              
 99                -1  1    332160  models.common.Conv                      [192, 192, 3, 1]              
100                -1  1    332160  models.common.Conv                      [192, 192, 3, 1]              
101                -1  1    332160  models.common.Conv                      [192, 192, 3, 1]              
102[-1, -2, -3, -4, -5, -6]  1         0  models.common.Concat                    [1]                           
103                -1  1    590592  models.common.Conv                      [1536, 384, 1, 1]             
104                -1  1   1770496  models.common.Conv                      [384, 512, 3, 2]              
105          [-1, 47]  1         0  models.common.Concat                    [1]                           
106                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
107                -2  1    525312  models.common.Conv                      [1024, 512, 1, 1]             
108                -1  1   1180160  models.common.Conv                      [512, 256, 3, 1]              
109                -1  1    590336  models.common.Conv                      [256, 256, 3, 1]              
110                -1  1    590336  models.common.Conv                      [256, 256, 3, 1]              
111                -1  1    590336  models.common.Conv                      [256, 256, 3, 1]              
112[-1, -2, -3, -4, -5, -6]  1         0  models.common.Concat                    [1]                           
113                -1  1   1049600  models.common.Conv                      [2048, 512, 1, 1]             
114                83  1    295424  models.common.Conv                      [128, 256, 3, 1]              
115                93  1   1180672  models.common.Conv                      [256, 512, 3, 1]              
116               103  1   2655744  models.common.Conv                      [384, 768, 3, 1]              
117               113  1   4720640  models.common.Conv                      [512, 1024, 3, 1]             
118[114, 115, 116, 117]  1  10466036  models.yolo.IKeypoint                   [1, [[19, 27, 44, 40, 38, 94], [96, 68, 86, 152, 180, 137], [140, 301, 303, 264, 238, 542], [436, 615, 739, 380, 925, 792]], 17, [256, 512, 768, 1024]]
```