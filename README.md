# WildFusion: Multimodal Implicit 3D Reconstructions in the Wild
Yanbaihui Liu, [Boyuan Chen](http://boyuanchen.com/)
<br>
Duke University
<br>
## Overview
We propose WildFusion, a novel approach for 3D scene reconstruction in unstructured, in-the-wild environments using multimodal implicit neural representations. WildFusion integrates signals from LiDAR, RGB camera, contact microphones, tactile sensors, and IMU. This multimodal fusion generates comprehensive, continuous environmental representations, including pixel-level geometry, color, semantics, and traversability. Through real-world experiments on legged robot navigation in challenging forest environments, WildFusion demonstrates improved route selection by accurately predicting traversability. Our results highlight its potential to advance robotic navigation and 3D mapping in complex outdoor terrains.

## Prerequisites

1. Clone the repository:

    ```bash
    git clone https://github.com/generalroboticslab/WildFusion.git
    ```

2. Create and activate a new virtual environment:

    ```bash
    virtualenv new_env_name
    source new_env_name/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Training

Run the following command to train the model. The `--scratch` flag will force training from scratch, while `--skip_plot` will skip saving training loss plots.

```bash
python main.py --scratch --skip_plot
```


## Evaluation

To evaluate the trained models and visualize the results, run:

```bash
python evaluation/test.py --test_file /path/to/data
```

To visualize the ground truth in `.pcd` format, use:

```bash
python evaluation/gt_vis_pcd.py --data_path /path/to/data
```

## Dataset
Download our [dataset](https://duke.box.com/s/02algnthvx1fb3znt50cdpov7ehgseto) and unzip

## Hardwares
The list of our hardware set and CAD model are under [hardwares](https://github.com/yanbhliu/WF_final_hold/tree/main/hardwares) subdirectory.

## Acknowledgement
[go2_ros2_sdk](https://github.com/abizovnuralem/go2_ros2_sdk)
