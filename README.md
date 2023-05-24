# retico-clip
A ReTiCo module for CLIP. See below for more information.

### Installation and requirements

### Example
```python
import sys
from retico import *

prefix = '/path/to/modules/'
sys.path.append(prefix+'retico-clip')
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-yolov8')

# from retico_yolov8 import YoloV8
from retico_clip.clip import ClipObjectFeatures
from retico_vision.vision import WebcamModule 
from retico_yolov8.yolov8 import Yolov8



webcam = WebcamModule()
yolo = Yolov8()
feats = ClipObjectFeatures(show=True)
debug = modules.DebugModule()

webcam.subscribe(yolo)
yolo.subscribe(feats)
feats.subscribe(debug)

run(webcam)

print("Network is running")
input()

stop(webcam)
```


Citation
```
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```