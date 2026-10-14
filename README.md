# R3RSCLIP: A Bias Correction Method for Remote Sensing Semantic Segmentation
📢This repository is the official implementation of the paper "R3RSCLIP: A Bias Correction Method for Remote Sensing Semantic Segmentation".

⏳ The full source code is currently undergoing internal cleanup and will be posted here upon release or during the rebuttal phase.

💡 Abstract:
  The transfer of vision-language models (VLMs) such as CLIP to unsupervised semantic segmentation (USS) of high-resolution remote sensing images (HRRSIs) is severely hampered by a pronounced domain gap. This paper argues that the gap can be decomposed into three structured biases rooted in the physics of remote sensing (RS) imaging. Building on this, we propose R3RSCLIP, a unified framework that rectifies the three biases in a complementary manner. Extensive experiments on four benchmarks demonstrate that R3RSCLIP achieves state-of-the-art performance on the ISPRS orthophoto benchmarks, reaching 38.40% and 30.24% mIoU on Potsdam and Vaihingen, respectively, and remains competitive on the oblique-view UAVid (31.18%) and the complex urban UDD5 (37.30%), confirming its robustness across sensors, resolutions, and viewing geometries.

![Uploading image.png…]()

Illustration of the structured domain gap in CLIP-based HRRSI semantic segmentation and the motivation of R3RSCLIP.

📝 Dataset storage format:
<pre> ReCLIP
├── data
│   ├── Potsdam
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── Vaihingen
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── UAVid
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── UDD5
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
</pre>
