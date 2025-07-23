# Soybean nodule detection and traits extraction pipeline

**Project background**: To be updated with manuscript

## Introduction<br />
  The purpose of this one is to detect soybean nodule, extract traits of each plant.<br />

## Installation <br />
1. **Clone the repository to the local drive**<br />
  ```
  git clone https://github.com/Salk-Harnessing-Plants-Initiative/soybean-nodule-detection.git
  ```

2. **Navigate to the cloned directory**
  ```
  cd soybean-nodule-detection
  ```

## Organize the pipeline and your images
Models can be downloaded from [Box](https://salkinstitute.box.com/s/cqgv1dwm1hkf84eid72hdjqg47nwbpo5).

Please make sure to organize the downloaded pipeline, model, and your own images in the following architecture:

```
soybean-nodule-detection/
├── images_labels/
│   ├── image_1 (e.g., Root_FC1234-2.JPG)
│   ├── label_1 (e.g., Root_FC1234-2.json)
│   ├── image_2 (e.g., Root_PI_56789-1.JPG)
│   ├── label_2 (e.g., Root_PI_56789-1.json)
│   ├── image_... 
│   ├── label_...
│   │   ├── species (e.g., Arabidopsis)/
│   │   │   ├── plant name (e.g., ZHOKUWVOIZ)/
│   │   │   │   ├── frame image (e.g., 1.png)
│   │   │   ├── plant and experiment mapping (e.g., acc_barcodes_cylinders.csv)
├── src/
│   ├── pipeline_get_traits.py
│   ├── pipeline_predict.py
│   ├── pipeline.sh
├── model/
│   ├── best_70img_4batch_512_64.pt
├── Dockerfile
├── environment.yml
├── LICENSE
├── README.md
├── requirements.txt
```

## Running the pipeline with a shell file (RootXplorer_pipeline.sh)
1. **create the environment**:

  In terminal, navigate to your root folder and type:
  ```
  conda env create -f environment.yml
  ```

  or
  ```
  mamba env create -f environment.yml
  ```

2. **activate the environment**:

  ```
  conda activate soybean-nodule-detection
  ```

3. **run the shell file**:

  ```
  sed -i 's/\r$//' src/pipeline.sh
  bash src/pipeline.sh
  ```

## 2. Methods<br />
### 2.1. Nodule detection<br />
Detect the soybean nodule using YOLOv8.2 based on ultralytics.<br />
References: <br />
https://docs.ultralytics.com/models/yolov8/<br />
https://github.com/ultralytics/ultralytics<br />
### 2.2. Traits extraction (plant- or image-based traits)<br />
- `count`: number of soybean nodules;<br />
- `height`: nodule zone in vertical axis, measured of the center locations from the top nodule to the bottom nodule in vertical axis (y-axis), unit is pixel, same below;<br />
- `width`: nodule zone in horizontal axis, measured of the center locations from the left nodule to the right nodule in forizontal axis (x-axis);<br />
- `sdx`, `sdy`: the standard deviations of all root nodules in x-axis (horizontal axis) or y-axis (vertical axis);<br />
- `sdxy`: the ratio of sdx and sdy;<br />
- `box_area_xx`: detected soybean nodule area, there are one or more nodules in each image, `xx` means nine statistics of all nodules in each image, which are minimum, maximum, standard deviation, average, median, 5th, 25th, 75th, and 95th percentiles;<br />
- `y_center_xx`: nodule distribution in y axis, there are one or more nodules in each image, `xx` means nine statistics of all nodule y locations in each image, which are minimum, maximum, standard deviation, average, median, 5th, 25th, 75th, and 95th percentiles;<br />
