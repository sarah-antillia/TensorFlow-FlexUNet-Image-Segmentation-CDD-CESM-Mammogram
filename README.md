<h2>TensorFlow-FlexUNet-Image-Segmentation-CDD-CESM-Mammogram (2025/06/12)</h2>

This is the first experiment of Image Segmentation for CDD-CESM-Mammogram (Benign and Malignant)
 based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>) and an
 <a href="https://drive.google.com/file/d/1d8AXGvjWbDP2MIKVXNLWlq5__OhhsGuT/view?usp=sharing">
Augmented-CDD-CESM-Mammogram-ImageMask-Dataset.zip</a>, which was derived by us from the 
<a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8">
Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images (CDD-CESM)</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for 512x512 Mammogram Dataset</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>
In the following predicted mask images, green regions indicate benign areas, while red regions indicate malignant tumors.<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P161_L_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P161_L_CM_MLO.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P161_L_CM_MLO.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P84_L_DM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P84_L_DM_MLO.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P84_L_DM_MLO.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P99_R_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P99_R_DM_CC.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P99_R_DM_CC.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<b>1. Dataset Citation</b><br>
The image dataset used here has been taken from the following web site.<br>
<a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8">
Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images (CDD-CESM)</a>
<br>
<br>
<b>Citations & Data Usage Policy</b><br>
Users must abide by the TCIA Data Usage Policy and Restrictions. Attribution should include references<br> 
to the following citations:<br>
<br>
<b>Data Citation</b><br>
Khaled R., Helal M., Alfarghaly O., Mokhtar O., Elkorany A., El Kassas H., Fahmy A. Categorized Digital Database<br>
for Low energy and Subtracted Contrast Enhanced Spectral Mammography images [Dataset]. (2021) The Cancer Imaging<br>
Archive. DOI:  10.7937/29kw-ae92<br>
<br>

<b>Publication Citation</b>
Khaled, R., Helal, M., Alfarghaly, O., Mokhtar, O., Elkorany, A., El Kassas, H., & Fahmy, A. Categorized contrast<br>
enhanced mammography dataset for diagnostic and artificial intelligence research. (2022) Scientific Data,<br>
Volume 9, Issue 1. DOI: 10.1038/s41597-022-01238-0<br>
<br>

<b>TCIA Citation</b><br>
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L,<br>
Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository,<br>
Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7<br>
<br>
<h3>
<a id="2">
2 Mammogram ImageMask Dataset
</a>
</h3>
 If you would like to train this Mammogram Segmentation model by yourself,
 please download the dataset from the google drive  
 <a href="https://drive.google.com/file/d/1d8AXGvjWbDP2MIKVXNLWlq5__OhhsGuT/view?usp=sharing">
Augmented-CDD-CESM-Mammogram-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Mammogram
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Mammogram Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Mammogram/Mammogram_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large to use for the
 training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Mammogram TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Mammogram/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Mammogram and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3

base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
Specifed rgb color map dict for Mammogram 3 classes.<br>
<pre>
[mask]
mask_datatype= "categorized"
mask_file_format = ".png"
;Mammogram rgb color map dict for 1+2 classes.
;        background:black , Benign:green  Malignant: red
rgb_map = {(0,0,0):0,(0,255,0):1, (255,0,0):2 }
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> 
<br> 
As shown below, early in the model training, the predicted masks from our UNet segmentation model showed 
discouraging results.
 However, as training progressed through the epochs, the predictions gradually improved. 
 <br> 
<br>
<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 20,21,22)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/epoch_change_infer_at_middle.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 42,43,44)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 44 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/train_console_output_at_epoch44.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Mammogram/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Mammogram/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Mammogram</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Mammogram.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/evaluate_console_output_at_epoch44.png" width="920" height="auto">
<br><br>Image-Segmentation-Mammogram

<a href="./projects/TensorFlowFlexUNet/Mammogram/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Mammogram/test was not low, and dice_coef_multiclass 
was high as shown below.
<br>
<pre>
categorical_crossentropy,0.1221
dice_coef_multiclass,0.9467
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Mammogram</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for Mammogram.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<!--
They should ideally be smoother.
 -->
 
<br>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Mammogram/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
As shown below, the predicted masks for the green benign areas by this <b>simple</b> semgentation model
were unsatifactory results. However, there might be a slight issue with the annotation of the mask (label) images 
in this segmentation dataset. Specifically, some polygons representing the benign regions appear unnaturally angular.

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P61_L_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P61_L_CM_MLO.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P61_L_CM_MLO.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P71_L_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P71_L_CM_MLO.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P71_L_CM_MLO.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P181_L_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P181_L_DM_CC.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P181_L_DM_CC.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P84_L_DM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P84_L_DM_MLO.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P84_L_DM_MLO.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P116_R_CM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P116_R_CM_CC.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P116_R_CM_CC.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/images/flipped_P99_R_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test/masks/flipped_P99_R_DM_CC.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Mammogram/mini_test_output/flipped_P99_R_DM_CC.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images (CDD-CESM)
</b><br>
<a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8">
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8
</a>
<br>
<br>
<b>2. Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research
</b><br>
Rana Khaled, Maha Helal, Omar Alfarghaly, Omnia Mokhtar, Abeer Elkorany,<br>
Hebatalla El Kassas & Aly Fahmy<br>
<a href="https://www.nature.com/articles/s41597-022-01238-0">
https://www.nature.com/articles/s41597-022-01238-0
</a>
<br>
<br>
<b>3. CDD-CESM-Dataset</b><br>
<a href="https://github.com/omar-mohamed/CDD-CESM-Dataset">
https://github.com/omar-mohamed/CDD-CESM-Dataset
</a>
<br>
<br>
<b>4. Breast Cancer Segmentation Methods: Current Status and Future Potentials</b><br>
Epimack Michael, He Ma, Hong Li, Frank Kulwa, and Jing<br>
<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321730/">
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321730/
</a>
<br>
<br>
<b>5. Tensorflow-Image-Segmentation-CDD-CESM-Mammogram</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-CDD-CESM-Mammogram">

https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-CDD-CESM-Mammogram
</a>
<br>
<br>

<b>6. TensorflowEfficientUNet-Segmentation-CDD-CESM-Mammogram</b><br>
Toshiyuki Arai @antillia.com<br>

<a href="https://github.com/sarah-antillia/TensorflowEfficientUNet-Segmentation-CDD-CESM-Mammogram">
https://github.com/sarah-antillia/TensorflowEfficientUNet-Segmentation-CDD-CESM-Mammogram
</a>

