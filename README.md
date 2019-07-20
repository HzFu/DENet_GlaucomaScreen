DENet_GlaucomaScreen
====================
![Python version range](https://img.shields.io/badge/python-2.7%E2%80%933.6+-blue.svg)

Code for TMI 2018 "Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image"

Project homepage：http://hzfu.github.io/proj_glaucoma_fundus.html

1. The code is based on: *TensorFlow 1.14 (with Keras)*
2. The deep output is raw segmentation result without ellipse fitting.
3. The pre-train models are trained on **ORIGA full dataset**.
4. Download the trained models for DENet to 'pre_model' folder: [[OneDrive]](https://1drv.ms/f/s!ArBRrL8ao6jzmkFHvpKCrwVzRdVh) [[BaiduPan]](https://pan.baidu.com/s/1eDT0N4tQsWI4McyGB36vLw):
	1. Disc detection model: 'pre_model_DiscSeg.h5' 
	2. Global image Screening model: 'pre_model_img.h5' 
	3. Segmentation-guided Screening model: 'pre_model_disc.h5' 
	4. Local disc Screening model: 'pre_model_ROI.h5' 
	5. Polar disc Screening model: 'pre_model_flat.h5'  

----------------

If you use this code, please cite the following papers:

[1] Huazhu Fu, Jun Cheng, Yanwu Xu, Changqing Zhang, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao, "Disc-aware Ensemble Network for Glaucoma Screening from Fundus Image", IEEE Transactions on Medical Imaging (TMI), 2018. DOI: 10.1109/TMI.2018.2837012 ([ArXiv version](http://arxiv.org/abs/1805.07549))

[2] Huazhu Fu, Jun Cheng, Yanwu Xu, Damon Wing Kee Wong, Jiang Liu, and Xiaochun Cao, "Joint Optic Disc and Cup Segmentation Based on Multi-label Deep Network and Polar Transformation", IEEE Transactions on Medical Imaging (TMI), vol. 37, no. 7, pp. 1597–1605, 2018. ([ArXiv version](https://arxiv.org/abs/1801.00926)) 

----------------

## For ORIGA and SCES datasets

Unfortunately, the ORIGA, SCES, and SINDI datasets cannot be released due to the clinical policy.

But, here is an other glaucoma challenge, *Retinal Fundus Glaucoma Challenge (REFUGE)*, including disc/cup segmentation, glaucoma screening, and localization of Fovea. If you are interested, you can register it from:
[[HERE]](https://refuge.grand-challenge.org/home/)

We also provide the results of our DENet, the details could be found from: [[HERE]](https://github.com/HzFu/REFUGE_baseline) 

----------------

Update log:

- 18.07.06: Released the code.

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .
