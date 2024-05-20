# Initial Code Upload

This repository contains the initial code upload for running on a Windows 11 PC equipped with an RTX 3090 GPU. The environment required for running the code is listed in `environment.yml`.

## Prerequisites

Before running the code, ensure you have the following:
- A PC with Nvidia GPU installed, at least 12 GB of GRAM required.
- Anaconda or Miniconda installed
- Python 3.8 or higher

## Setting Up the Environment

 Open a terminal and clone this repository using:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   conda env create -f environment.yml
   conda activate <environment_name>
  ```
  

## To Run the Testing Code

**Edit the Script:**
Open the test_reg_attngan_dpm.py script in a text editor. Update the paths to the model and images to match your local directories.
(In line 19,23,27)

**Run the Script:**
python test_reg_attngan_dpm.py

## Resources
- **Model and Demo Images**: Access the necessary model and demo images through Google Drive: https://drive.google.com/drive/folders/14kA0NlJQSwvDjqYROvEe14UcQsoPn5FW

## Support

For any questions, please email [mikexlyang@ucla.edu](mailto:mikexlyang@ucla.edu).
