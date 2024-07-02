# Detection-of-Crackles-and-Wheezes-in-Lung-Audio-Samples

## Dataset Description

### Context
Respiratory sounds are important indicators of respiratory health and disorders. The sound emitted when a person breathes is directly related to air movement, changes within lung tissue, and the position of secretions within the lung. A wheezing sound, for example, is a common sign that a patient has an obstructive airway disease like asthma or chronic obstructive pulmonary disease (COPD).

### Content
It includes 920 annotated recordings of varying length (10s to 90s) from 126 patients, totaling 5.5 hours of recordings. The dataset contains 6898 respiratory cycles, with 1864 containing crackles, 886 containing wheezes, and 506 containing both crackles and wheezes. The recordings encompass both clean respiratory sounds and noisy recordings that simulate real-life conditions. Patients of all age groups - children, adults, and the elderly - are included in the dataset.

#### Dataset Files
- 920 .wav sound files
- 920 annotation .txt files
- Diagnosis file listing the medical condition of each patient
- Demographic information file with details like age, sex, BMI (for adults), weight, and height (for children)

#### File Naming Format
Each audio file name is structured with the following elements separated by underscores (_):
1. Patient number (101, 102, ..., 226)
2. Recording index
3. Chest location:
   - Trachea (Tc)
   - Anterior left (Al)
   - Anterior right (Ar)
   - Posterior left (Pl)
   - Posterior right (Pr)
   - Lateral left (Ll)
   - Lateral right (Lr)
4. Acquisition mode:
   - Sequential/single channel (sc)
   - Simultaneous/multichannel (mc)
5. Recording equipment:
   - AKG C417L Microphone (AKGC417L)
   - 3M Littmann Classic II SE Stethoscope (LittC2SE)
   - 3M Litmmann 3200 Electronic Stethoscope (Litt3200)
   - WelchAllyn Meditron Master Elite Electronic Stethoscope (Meditron)

### Demographic Information
The demographic information file includes the following columns:
- Patient number
- Age
- Sex
- Adult BMI (kg/m²)
- Child Weight (kg)
- Child Height (cm)

### Annotation Files
The annotation text files contain four columns:
- Beginning of respiratory cycle(s)
- End of respiratory cycle(s)
- Presence/absence of crackles (presence=1, absence=0)
- Presence/absence of wheezes (presence=1, absence=0)

#### Diagnosis Abbreviations
- COPD: Chronic Obstructive Pulmonary Disease
- LRTI: Lower Respiratory Tract Infection
- URTI: Upper Respiratory Tract Infection

## Problem Statement

We were provided with lung sound audio recordings categorized into four types representing lung diseases/disorders. Our task was to develop a deep learning model using Librosa for feature extraction and ensemble CNN architectures to classify these sounds accurately.

## Approach

### Data Preprocessing

We extracted respiratory cycles from audio files using start and end times provided in corresponding text files. Standardization of audio length was achieved through padding or clipping. Processed audio files were stored separately, and metadata including patient ID, collection method, and filename were organized into a DataFrame.

### Model Architecture

#### Feature Extraction
Using Librosa, we extracted features from audio files:
- Short Time Fourier Transform
- Mel Spectrogram
- Mel-frequency cepstral coefficients (MFCC)

These features were processed into numpy arrays and fed into separate CNNs.

#### Ensemble Neural Network
The ensemble model consisted of three CNNs, each trained on one of the extracted features. Outputs from these CNNs were combined using another neural network to classify audio into categories: wheezes, crackles, both, or normal.

#### Model Architecture Details
Each CNN included convolutional layers, batch normalization, ReLU activation, and max pooling. The final network concatenated outputs from the three CNNs, passed through dense layers with dropout, and had an output layer with four nodes for classification.

### Training and Evaluation

#### Training Data Preparation
The DataFrame was split into training and testing sets for model training.

#### Training Process
The "Net" model was trained using Adam optimizer and categorical cross-entropy loss for a specified number of epochs.

### Model Architecture
![Screenshot 2024-07-03 015348](https://github.com/Aryansh-kr/Detection-of-Crackles-and-Wheezes-in-Lung-Audio-Samples/assets/127012188/c68e9567-0c70-4d89-b3d3-b872d7c0b5e1)


#### Model Evaluation
The model achieved an impressive accuracy of 92.46% on the test set, demonstrating effective classification of lung sound defects.

## Conclusion

The ensemble deep learning approach successfully identifies lung sound abnormalities with high accuracy. Future improvements may include incorporating additional features to further enhance model performance.

## Usage

To use this project:
1. Download the respiratory sound dataset from [Kaggle link or dataset source](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database/data).
2. Preprocess audio files using the provided methods to extract features.
3. Train the ensemble deep learning model as described.
4. Evaluate model performance and make predictions on new respiratory sound data.

This README file provides a detailed overview of the project, integrating dataset information with the deep learning model development and evaluation process, highlighting the achieved accuracy of 92.46%.
