Data for the model

1. Preprocess:
- input: food image and distance. Use a standard dinner plate size
- output: resized image so the food takes up the entire image space
2. Labelling:
- input: output of the preprocess, use opencv to segment the image by colour and shape
- output: image with ingredients identified using ML strategies (CNN)

Datasets to use:
- from huggingface: https://huggingface.co/datasets/aryachakraborty/Food_Calorie_Dataset
- from huggingface: https://huggingface.co/datasets/breadlicker45/Calorie-dataset
- from kaggle (main dataset): https://www.kaggle.com/datasets/dansbecker/food-101
- from kaggle: https://www.kaggle.com/datasets/kkhandekar/calories-in-food-items-per-100-grams
- from foodd: https://ieee-dataport.org/open-access/foodd-food-detection-dataset-calorie-measurement-using-food-images
- from kaggle: https://www.kaggle.com/datasets/vaishnavivenkatesan/food-and-their-calories