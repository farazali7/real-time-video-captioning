# Real Time Video Captioning
This repository contains the source code of the project "Real Time Video Captioning" which is a course project for the _CSC2231HS — Topics in Computer Systems: Visual and Mobile Computing Systems_ course at the University of Toronto. This repo takes inspiration and examples from the 
[GIT: A Generative Image-to-text Transformer for Vision and Language](https://arxiv.org/abs/2205.14100) research by Microsoft.

# Setup
- Clone the repository into your machine using the following command  
  ```shell
  git clone https://github.com/farazali7/real-time-video-captioning.git
  cd real-time-video-captioning
  ```

- Install the package  
  ```shell
  pip install -r requirements.txt
  python setup.py build develop
  ```

- You would also need to install `azcopy` to download some of the data. You can manually install `azcopy` by checking out the source repository here [azfuse](https://github.com/microsoft/azfuse). If you are running on OSX then do the following if you have homebrew    
`brew install azcopy`