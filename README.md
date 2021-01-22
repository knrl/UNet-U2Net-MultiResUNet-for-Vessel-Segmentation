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

<h2 align="center">  Deep Learning Methods for Retinal Vessel Segmentation  </h2>

<p>
Retinal vessel segmentation is popular in medical image processing and is an important topic for treatment, diagnosis and clinical evaluations. Deep learning-based methods for retinal vessel segmentation are often calculated and trained on a pixel basis, evaluating all vessel pixels with equal importance. This approach is more successful for micro vessel segmentation. In this study, U-Net model, which is the most successful deep learning model for biomedical image segmentation, and U2-Net models inspired by U-Net architecture were used. Quantitative and qualitative comparisons were made with these two models in STARE, DRIVE, and HRF data sets for retinal vessel segmentation. In order to evaluate the factors affecting the performance in vessel segmentation, the same data sets were also compared with different sizes and image processing techniques.
</p>

<h4>  Method  </h4>

<p>
U-Net architecture consists of a symmetrical U-shaped encoder and decoder. The success of the U-Net architecture is one of the jump links [1]. The output of the convolutional layer of each level is transferred to the decoder of the same level before the encoder pooling. Then the up-sampling process is transmitted to successive layers. These jump links enable the spatial information lost due to network pooling processes to be retrieved. Spatial information is important in medical image analysis applications [2]. U2-Net is a two-level nested U-Net architecture.
</p>

<h4>Method of Application</h4>

<p>
In this study, scimage, sci-kit learn, OpenCV, albumentations libraries are used for data enhancement and preprocessing with Keras using U-Net, U2-Net models Tensorflow background. Models were evaluated in low (DRIVE and STARE) and high (HRF) resolution datasets [4, 5, 6]. DRIVE and STARE were examined in two different fragment sizes and the effect of fragment size on vessel segmentation was observed. In addition, the data sets were passed through two different pre-processing stages and the effects of these stages on vessel segmentation were evaluated within the scope of models.
</p>


<p align="center">
  <img height="250"  src="results/preprocess.png">
  <p>Figure 1. (a) Original image; (b) Green channel; (c) CLAHE; (d) Gamma enhancement</p>
</p>

<p>
a. Vessel Enhancement
In retinal images, the green channel exhibits the best vessel and background contrast, while the red and blue ones tend to be very noisy [7]. The first preprocessing approach used in the study is to take the green channel from each visual and normalize it. The normalized visuals obtained from the green channel were enhancement by applying gamma correction after CLAHE (Figure 1) [8].
</p>

