# Instruction - scalable computing project 1  
**Group 15**

## Group Members
- Hugh Langan (21367862)  
- Vihit Khetle (25339598)  
- Nakshatra Shrivastava (25332635)

## Files:

**`captcha_gen_given.py`**  
creates a set of captchas for training data with an associated .csv file of correct answers for validation purposes, defaulted to 64000 files

**`image_processing.py`**  
performs basic image pre-processing to improve performance of machine learning. includes basic main script for testing but mainly called from other modules

**`bulk_process_images.py`**  
applies image processing to all files in output_captchas, storing the outputs in processed_captchas. can feasibly be executed on the pi itself.

**`train_tf_ctc.py`**  
trains the Keras model based on the dataset in processed_captchas. given parameters should match the performance of our group's submitty scores after finetuning. runtime at these parameters is 12+ hours.

**`finetune_tf_ctc.py`**  
applies more training to improve the accuracy. generally more effective when working with a smaller subset of the data rather than the full 64000 images, but still provides significant improvements. runtime is in the same range as training, maybe slightly slower

**`keras_to_csv.py`**  
converts the keras model directly to a submittable csv file, good for sanity check before moving on to running inference on the pi

**`convert_tflite.py`**  
converts the given keras model to a tflite model that will run directly on the pi

**`verify_tflite.py`**  
another sanity check before moving to pi

**`tflite_to_csv.py`**  
script for classifying the test data directly on the pi. applies image preprocessing and inferencing through tflite_runtime. csv output from this should be identical to the csv output from keras_to_csv.py, but executable without tensorflow

**`preview preprocessing`**  
shows a random set of 5 captchas, before and after preprocessing

**`individual_data` (directory)**  
includes the fonts and zipped PNG test set for each team member
