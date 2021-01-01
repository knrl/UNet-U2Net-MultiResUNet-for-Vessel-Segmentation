<h1 align="center">  U-Net U^2-Net MultiRes U-Net for Retinal Vessel Segmentation  </h1>

<p align="center">
  <img height="400"  src="results/patch_pred.png">
</p>

## Codes

The model architecture codes:

* [U-Net Model](https://github.com/knrl/UNet-USquarredNet-MultiResUNet-for-Retinal-Vessel-Segmentation/blob/main/models/unet.py)
* [USquared-Net Model](https://github.com/knrl/UNet-USquarredNet-MultiResUNet-for-Vessel-Segmentation/blob/main/models/usquarednet.py)
* [MultiRes U-Net Model](https://github.com/knrl/UNet-USquarredNet-MultiResUNet-for-Vessel-Segmentation/blob/main/models/resunet.py)

Dataset, training and test codes:

* [Extract Patch and Data Augmentation](https://github.com/knrl/UNet-USquarredNet-MultiResUNet-for-Retinal-Vessel-Segmentation/blob/main/preparation_dataset.py)
* [Train, Test, Visualization (merge images and compare prediction with test images)](https://github.com/knrl/UNet-USquarredNet-MultiResUNet-for-Retinal-Vessel-Segmentation/blob/main/train_test_visualize.py)

## Prerequisities

The following dependencies are needed:
- numpy >= 1.19.4
- tensorflow >= 2.3.1
- tqdm >= 4.52.0
- keras >= 2.4.3
- albumentations >= 0.5.2
- scikit-learn >= 0.23.2


## Citation Request

If you have benefited from this project, please cite the following paper.

```
@article{

  Soon

}

```
