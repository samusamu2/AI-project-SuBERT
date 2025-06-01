# AI Project
This repository contains the project for the course `30562 - MACHINE LEARNING AND ARTIFICIAL INTELLIGENCE`, Bocconi University.

**Abstract:**
Recent advances in computer vision and natural language processing open new possibilities
for reading and translating ancient cuneiform texts: in this work, we tackle one of the most
ancient and obscure languages, spoken by the Sumerians, a grandiose civilization that flour-
ished in the Near East around 2000 BCE and vanished millennia ago, leaving us with only
a few scant testimonies. We focus on two main tasks: first, we exploit image-classification
techniques to isolate and recognize individual characters from high-resolution tablet images.
Then, we attempt to translate Sumerian into English via Transformer-like architectures.

## Table of Contents

- [AI Project](#ai-project)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Datasets](#datasets)
  - [Using `rclone` to access to cloud files](#using-rclone-to-access-to-cloud-files)
  - [Installation](#installation)

## Repository Structure
This repository is structured in a way it is possible to create a full pipeline and replicate all the steps we followed while bulding our project.

All folders and Jupyter notebooks are commented, to be as more accessible as possible for the reader.

Folders (and files in each folder) are numbered, so that it is possible to ispect the content of each of them and run specific portion of our code for the desired task:

- `1-dataset_operations`: contains all files necessary to download the datasets from HuggingFace and perform some cheks on data.
- `2-vision_task_glyphs_recognition`: contains our attempts in recognizing through some object detection models.
- `3-glyphs_translitteration`: contains the necessary files to convert glyphs from ASCII encoding to their translitterated version, which is the format on which translation models are trained (so it is useful to potentially create a full pipeline to translate Sumerian text detected from some images).
- `4-translation`: contains all the models trained for translation.
- `datasets`: contains the datasets we used in this project.
- `fonts`: contains a Google Font file to print Sumerian unicode characters.
- `z-temp`: contains some further failed attempts of producing different translation models.

## Datasets
Our project utilizes the SumTablets dataset (Version v1), a dataset curated by Cole Simmons, Richard Diehl Martinez, and Prof. Dan Jurafsky (2024). This resource, available from their [GitHub repository](https://github.com/colesimmons/SumTablets) and distributed though [HuggingFace](https://huggingface.co/datasets/colesimmons/sumtablets), is licensed under the Creative Commons Attribution 4.0 International, allowing for its adaptation and reuse with proper credit. You can read more about their work in the GitHub repository and in the related paper: [SumTablets: A Transliteration Dataset of Sumerian Tablets](https://aclanthology.org/2024.ml4al-1.20/) (Simmons et al., ML4AL 2024).

## Using `rclone` to access to cloud files

Large files may have been moved to a OneDrive folder due to space limitations. You can find them in the [shared folder](https://bocconi-my.sharepoint.com/:f:/g/personal/samuele_straccialini_studbocconi_it/EjLdGCkSWehAq587J8KcJ9YBEsersXTvplLDyZ8OBpLDfA?e=PW31EI).

If you wish to access the shared folder and download datasets using `rclone`, please follow the following passages:
- Copy the folder into your own OneDrive.
- Install `rclone` on your machine: https://rclone.org/install/
- Run `rclone config` in your terminal and connect it to your OneDrive.

**Note:** if you wish to use our code to read and write in the folder, kindly name the remote "`onedrive_bocconi`" when running `rclone` configuration.

## Installation

To install the required dependencies, make sure you have Python 3.7+ installed, then run:

```bash
pip install -r requirements.txt
```

