# retico_clip
A ReTico module for CLIP visual feature embeddings from objects detected within an image.

## Installation and Requirements

To use the CLIP module you first need to install the retico-core and retico-vision packages:
* Install retico_core:
  ```pip install git+https://github.com/retico-team/retico-core.git```
* Install retico_vision:
  ```pip install git+https://github.com/retico-team/retico-vision.git```

Right after that, install OpenAI's CLIP package:
* Install clip:
`````pip install git+https://github.com/openai/CLIP.git```

Then install the retico-clip package:
* Install retico-clip:
````pip install git+https://github.com/retico-team/retico-clip.git```

## Modules

### `ClipModule`
Subscribes to `ExtractedObjectsIU` (produced by `ExtractObjectsModule`) and encodes each cropped object image using CLIP, producing an `ObjectFeaturesIU` containing a dictionary of feature vectors, one per successfully encoded object.

#### Arguments:
* `model_name` (str): the CLIP model checkpoint to load; defaults to `'ViT-B/32'`
* `show` (bool): if `True`, opens a live `cv2` preview window showing each object crop as it's processed; useful for visually confirming detections and crop quality; defaults to `False`
* `top_objects` (int): the maximum number of valid (non-empty) objects to encode per incoming frame; objects with degenerate/empty crops are skipped and do not count against this limit; defaults to `1`

#### Arguments
* `model_name` (str): the CLIP model checkpoint to load; defaults to `'ViT-B/32'`
* `show` (bool): if `True`, opens a live `cv2` preview window showing each object crop as it's processed; useful for visually confirming detections and crop quality; defaults to `False`
* `top_objects` (int): the maximum number of valid (non-empty) objects to encode per incoming frame; objects with degenerate/empty crops are skipped and do not count against this limit; defaults to `1`

### Example
**Note:** The example runner uses `ExtractObjectsModule` from `retico_vision`. The Extract Objects module allows the user to determine how many images to plot to be displayed

```python
import sys, os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

prefix = "/path/to/prefix/"
sys.path.append(prefix + "retico-core")
sys.path.append(prefix + "retico-vision")
sys.path.append(prefix + "retico-sam")
sys.path.append(prefix + "retico-dino")
sys.path.append(prefix + "retico-clip")

from retico_core import *
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule
from retico_vision.vision import ExtractObjectsModule
from retico_sam.sam import SAMModule
from retico_clip.clip import ClipObjectFeatures

path_var = "sam_vit_h_4b8939.pth"

webcam = WebcamModule()
sam = SAMModule(model="h", path_to_chkpnt=path_var, use_bbox=True)
extractor = ExtractObjectsModule(num_obj_to_display=1)
feats = ClipObjectFeatures(show=True)
debug = DebugModule()

webcam.subscribe(sam)
sam.subscribe(extractor)
extractor.subscribe(feats)
feats.subscribe(debug)

webcam.run()
sam.run()
extractor.run()
feats.run()
debug.run()

print("Network is running")
input()

webcam.stop()
sam.stop()
extractor.stop()
debug.stop()
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
