# Real Time Video Captioning

This repository contains the source code of the project "Real Time Video Captioning," a course project for the _CSC2231HS — Topics in Computer Systems: Visual and Mobile Computing Systems_ course at the _University of Toronto_. This repo takes inspiration from the [GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) research by Microsoft.

## Setup

### Clone the repository

Clone the repository into your machine using the following command:   
```shell
git clone https://github.com/farazali7/real-time-video-captioning.git
cd real-time-video-captioning
```

### Create and activate a virtual environment (optional but recommended)

To avoid package version conflicts, it's recommended to create and activate a virtual environment. Here are the steps for different operating systems:   
- Windows
```shell
python -m venv venv
venv\Scripts\activate
```

- macOS
```shell
python -m venv venv
source venv/bin/activate
```

### Install the required packages

Install the packages using the following commands:   
```shell
pip install -r requirements.txt
python setup.py build develop
```

### Install azcopy (for data download)

You would also need to install azcopy to download some of the data. Follow the appropriate instructions for your operating system:   

- Windows
Manually install `azcopy` by checking out the source repository [azfuse](https://github.com/microsoft/azfuse), or download the installer from the official Microsoft website.

- macOS
If you are running on macOS and have Homebrew installed, you can install `azcopy` using the following command:
```shell
brew install azcopy
```