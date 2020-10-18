## Exporatory Data Analysis and Data Preperation (EDA)

Complete dataset analysis for original, duplicate, misclassified and noise count is captured in the notebook  
**Notebook:** DAM_AI_eda_data_summary_v1.ipynb[(Link)](DAM_AI_eda_data_summary_v1.ipynb)

> References:
- https://pysource.com/2018/07/19/check-if-two-images-are-equal-with-opencv-and-python/
- https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/

## Finding duplicate images for same driver in a class

**Notebook:** DAM_AI_eda_find_duplicates_v1.ipynb[(Link)](DAM_AI_eda_find_duplicates_v1.ipynb)
- input: imgs_left (original dataset)
- output: dam_duplicate_list.csv (duplicate and ssim score is marked against images)  

## Streamlining duplicates base don different threshold for each classes

- using input file(dam_duplicate_list.csv) it further identify the duplicate images for different SSIM threshold.
- final duplicate list is stored into dam_duplicate_list_ext.csv
- **Notebook:** DAM_AI_eda_find_duplicates_diff_threshold_v1.ipynb[(Link)](DAM_AI_eda_find_duplicates_diff_threshold_v1.ipynb) 
- input: dam_duplicate_list.csv
- output: dam_duplicate_list_ext.csv

## Creating dataset by removing duplicates

**Notebook:** DAM_AI_eda_create_dataset_removing_duplicates_v1.ipynb[(Link)](DAM_AI_eda_create_dataset_removing_duplicates_v1.ipynb) 
- using original dataset and file dam_duplicate_list_ext.csv.. new dataset is created
- input: imgs_left (original dataset)
- output: imgs_left_cure (after removing duplicates)

## Removing misclassified images

misclassified images are identified manually for each class.
- input: imgs_left_cure
- output1: imgs_left_cure_final (all misclassified images are removed)
- output2: imgs_left_cure_misclassified_samples (this folder contains all misclassified images)

> NOTE: 
- imgs_left_cure_final: this dataset contains all the samples after removing duplicates and misclassified images.

## Converting left dataset into right hand dataset

**Notebook:** AI_DAM_eda_create_righthand_dataset_v1.ipynb[(Link)](AI_DAM_eda_create_righthand_dataset_v1.ipynb) 
- input: imgs_left_cure_final
- output: imgs_right_cure_final

## Adding noise

**Notebook:** AI_DAM_eda_add_noise_to_dataset_v1_ssim.ipynb[(Link)](AI_DAM_eda_add_noise_to_dataset_v1_ssim.ipynb) 
- input: imgs_right_cure_final
- output: imgs_right_cure_final_noise

> NOTE: Model shall be build for imgs_right_cure_final_noise dataset