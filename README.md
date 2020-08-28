
# U_Net Model
> This projects detects cancer region by U_Net model.

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Dataset](#Dataset)
* [Run](#Run)
* [Contact](#contact)

## General info
In this project a series of train data are available. The train data consists of medical pictures and in their
corresponding mask pictures, the region of cancers are outlined by white color. Using U_Net, we can train a deep learning
model that is capable of detecting cancer regions in medical images.

## Technologies
* Coded on Python 3.7.7 -- keras 2.3.1 -- tensorflow-gpu 1.14.0

## Dataset
* before running the code, unzip the dataset.rar into the python code folder , the n run the code.

## Run
On command line, enter the following:\
&nbsp; &nbsp; &nbsp; Python Main.py\
After that:\
&nbsp; &nbsp; &nbsp; To run the code with default values, just press Enter, \
     or:\
&nbsp; &nbsp; &nbsp; Enter the parameters as following format:\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; -e 2 -b 2 -f 0.1 -p 5 -y 64 -x 64 -z 3 -m 0 -s 1 -l 1 \
     or:\
&nbsp; &nbsp; &nbsp; To get help, type "-h" and press Enter 
     

## Contact
Created by [@raminmehdizad] - feel free to contact me!
