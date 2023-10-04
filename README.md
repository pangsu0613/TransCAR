# TransCAR: Transformer-based Camera-And-Radar for 3D Object Detection

TransCAR is a Transformer-based Camera-And-Radar fusion solution for 3D object detection. The cross-attention layer within the transformer decoder can adaptively learn the soft-association between the radar features and vision-updated queries instead of hard-association based on sensor calibration only. Our model estimates a bounding box per query using set-to-set Hungarian loss, which enables the method to avoid non-maximum suppression. TransCAR improves the velocity estimation using the radar scans without temporal information.

Our implementations are built on top of MMdetection3D.  

### Install
Please follow [detr3d](https://github.com/WangYueFt/detr3d) to prepare the [Prerequisite](https://github.com/WangYueFt/detr3d#prerequisite) and [Data](https://github.com/WangYueFt/detr3d#data). This project is developed based on detr3d codebase, thanks for their excellent work!

We recommend to use conda to setup the environment, [this](https://docs.google.com/document/d/14O8JgboRjUl1ihMmHYy6GogOyxQhXAg82z9yxP-hcK0/edit?usp=sharing) is the installed packages list for our conda environment for reference.

### Train
After preparing the data following mmdet3d and installation of the environment. Please download the [pre-trained detr3d weights](https://drive.google.com/file/d/1RpGIwQSHobcUO56Q0d7VToWG0kQVg0Pr/view?usp=sharing) for the initialization of the camera network. Then update the `load_from` under `projects/configs/detr3d/detr3d_res101_gridmask.py` to point to your downloaded pre-trained detr3d weights. There are three different detr3d models, the one mentioned above is the smallest which is suitable for fast develop and debug, if you have a high-end GPU system with sufficient memory and compute power, you can use the other two bigger detr3d models ([model1 pre-trained weights](https://drive.google.com/file/d/1D4h4YO_M_gdp6xPll4Vo9Aa5K8yaWa1E/view?usp=drive_link) and [model2 pre-trained weights](https://drive.google.com/file/d/1dYWuq26NxMY4mobNLjo3vX9K45AnkSLe/view?usp=drive_link)). And then update the `load_from` in the corresponding config files (detr3d_res101_gridmask_cbgs.py or detr3d_res101_gridmask_det_final_trainval_cbgs.py depending on the model that you choose).
For standard train/eval, please use the following line at the top of `projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py`, note that you need to change the `dataroot` to point to your nuScenes data directory.
`nusc = NuScenes(version='v1.0-trainval', dataroot='/home/xxx/nuscene_data/NUSCENES_DATASET_ROOT', verbose=True)`
For fast testing and debugging using nuScenes mini dataset, please use the line below under `projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py`, note that you need to change the `dataroot` to point to your nuScenes data directory.
`nusc = NuScenes(version='v1.0-mini', dataroot='/home/xxx/nuscene_data/NUSCENES_DATASET_ROOT', verbose=True)``

Run the command below to launch the training:
`python tools/train.py /TransCAR/projects/configs/detr3d/detr3d_res101_gridmask.py`

### Evaluation on Validation set
Following the directions in section Train to setup the NuScenes data object.
To evaluate our trained model, please download the weights (model_weights.pth) from [here](https://drive.google.com/file/d/1B5Mi_4pSzHZU-JywRtX9YVD01SvB-CqB/view?usp=sharing).
then run the command below for evaluation:
`python tools/test.py /TransCAR/projects/configs/detr3d/detr3d_res101_gridmask.py /path/to/trained/weights --eval=bbox`
Example command:
`python tools/test.py /TransCAR/projects/configs/detr3d/detr3d_res101_gridmask.py /path/to/model_weights.pth --eval=bbox`

### Evaluation using pretrained models
Download the weights accordingly.  

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|DETR3D (baseline)|34.7|42.2|[model](https://drive.google.com/file/d/1RpGIwQSHobcUO56Q0d7VToWG0kQVg0Pr/view?usp=sharing)|
|TransCAR|35.5|47.1|[model](https://drive.google.com/file/d/1B5Mi_4pSzHZU-JywRtX9YVD01SvB-CqB/view?usp=sharing)|


### Run inference on nuScenes test set (prepare for submission)
For best performance, we recommend using detr3d_vovnet_trainval version detr3d as the camera network (download the pre-trained weights [here](https://drive.google.com/file/d/1dYWuq26NxMY4mobNLjo3vX9K45AnkSLe/view?usp=sharing)). Then use the line below under `projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py`.
`nusc = NuScenes(version='v1.0-test', dataroot='/home/xxx/nuscene_data/NUSCENES_DATASET_ROOT', verbose=True)`
Run the following command to generate the detection files (you can download the pre-trained TransCAR model weights [here](https://drive.google.com/file/d/1hAP6ddGoZAJdt7p3UXv8m_ihqxdRmr8b/view?usp=drive_link)):
`python tools/test.py /TransCAR/projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py /dir/to/trained/weights/weights_final_test.pth --format-only --eval-options 'jsonfile_prefix=/dir/to/save/the/results'`

Evaluation results on the nuScenes test set: mAP: 42.2; NDS: 52.2

If you find this work useful in your research, please consider citing:

```
@article{pang2023transcar,
  title={TransCAR: Transformer-based Camera-And-Radar Fusion for 3D Object Detection},
  author={Pang, Su and Morris, Daniel and Radha, Hayder},
  journal={arXiv preprint arXiv:2305.00397},
  year={2023}
}
```

### Acknowledgement
Again, this work is developed based on [detr3d](https://github.com/WangYueFt/detr3d), thanks for their good work!
