# Data Folder

Mostly consists of the dataset configurations and the actual MSRVTT dataset that is used for training, which needs to be downloaded and moved here as per the instructions of the home page `README` file.


# Table of Contents

- [Data Folder](#data-folder)
    - [Repository Structure](#repository-structure)
    - [Labels](#labels)
    - [MSRVTT Dataset](#msrvtt-dataset)
    - [Teacher Configs](#teacher-configs)


## Repository Structure

The repository consists of data related to video processing and labeling, structured as follows:

```
data
├── labels
│   ├── encoded_captions.pkl
│   └── labels.csv
└── MSRVTT <-- To be installed, instructions on previous page README.
│   ├── annotation
│   │   └── ...
│   ├── high-quality
│   │   └── ...
│   ├── structured-symlinks
│   │   └── ...
│   └── videos
│       └── ...
└── teacher_configs
│       └──GIT_LARGE_MSRVTT
│               └── parameter.yaml
│
└── README.md <-- You are here.
```


## Labels

Contains the `encoded_captions.pkl` pickle file. Also contains a CSV file called `labels.csv` consisting of `image_id` and caption pairs. The example below illustrates the first few lines of the CSV file for ease of viewing.

```
 --------------------------------------------------------------------------------------------
|   | caption                                                       | id | image_id  | split |
|---|---------------------------------------------------------------|----|-----------|-------|
| 0 | a cartoon animals runs through an ice cave in a video game    | 0  | video2960 | train |
| 1 | a cartoon character runs around inside of a video game        | 1  | video2960 | train |
| 2 | a character is running in the snow                            | 2  | video2960 | train |
| 3 | a person plays a video game centered around ice age the movie | 3  | video2960 | train |
| 4 | a person plays online and records themselves                  | 4  | video2960 | train |
| 5 | a scene from the ice age video game is shown                  | 5  | video2960 | train |
| 6 | a video game character is jumping about in a cave             | 6  | video2960 | train |
| 7 | a video game of a little animal running through an ice tunnel | 7  | video2960 | train |
| 8 | a video game of a small animal                                | 8  | video2960 | train |
 --------------------------------------------------------------------------------------------
```

- You will notice that some of the image ids will have mapping to more than one caption, and if you think about it, it does make sense. This can have very unique benefits during training.

- Splits have been provided for training, testing and validation accordingly. The images will be loaded based on the `image_id` (which is a video) accordingly.


## MSRVTT Dataset

The dataset which contains the data required for training. Consists of videos for you to decide how to sample frames in order to train. We use the same dataset to train that was used in the original `GenerativeImage2Text` research. If installation has not been done yet, follow the instructions below.

Install the `MSRVTT` dataset from this [link](https://cove.thecvf.com/datasets/839). Unzip the contents and move the `MSRVTT` folder to the `data` directory as shown in the repository structure. Its contents should look like this:

```
data
└── MSRVTT
      ├── annotation
      │   └── ...
      ├── high-quality
      │   └── ...
      ├── structured-symlinks (generated on run)
      │   └── ...
      ├── videos
      │   └── ...
```


## Teacher Configs

Contains a `parameter.yaml` file that specifies the teacher model instantiation parameters. Nothing needs to be changed here as such.