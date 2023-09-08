# CSE 515 - Multimedia and Web Databses

Phase 1 - Image features, vector models, similarity/distance measures

This is an individual project, the contents of this repository were
coded solely by the author with relevant resources used linked via
comments within the code as well as the accompanying project report.


Author Information:

Name: Gokul Vasudeva
Email: gvasude2@asu.edu
Group No: 15


Setup:

The system was developed using Python 3.11.5.

1. It is recommended to setup a virtual environment:

python3 -m venv .venv
source venv/bin/activate

2. Install requirements within venv:

pip install -r requirements.txt

3. Either have an IDE which supports opening Jupyter Notebooks respecting
choosing the virtual environment as the interpreter (such as VSCode), or
start jupyter notebook from within the venv:

jupyter notebook

4. This should open a new browser tab with the codebase. Navigate to
Code/<task_no>.ipynb to open a notebook, and follow the instructions.


Instructions for task 1:

    Implement a program which, given an image ID and one of the
    following feature models, visualizes the image and then extracts and
    prints (in a human readable form) the corresponding feature descriptors:

    1. Color moments (CM10x10)
    2. Histograms of oriented gradients (HOG)
    3. ResNet-AvgPool-1024
    4. ResNet-Layer3-1024
    5. ResNet-FC-1000

Task 1 has been presented in the form of a Jupyter Notebook with input prompts
for accepting the image ID. The outputs for this task are visualized within
the notebook.

Please note that the system will not accept an image ID that corresponds to an
image that does not have RGB channels, and will keep prompting you to provide
a different image ID which does.



Instructions for task 2:

    Implement a program which extracts and stores feature descriptors for
    all the images in the data set.

Task 2 is a Python script, run it in the virtual environment as:

python3 Code/task_2.py

Warning: Running this script takes well over an hour.

Please note that the system will ignore images which do not have RGB channels,
and therefore they cannot show up during image retrieval.



Instructions for task 3:

    Implement a program which, given an image ID and a value “k”, returns
    and visualizes the most similar k images based on each of the visual
    model -you will select the appropriate distance/similarity measure for
    each feature model. For each match, also list the corresponding
    distance/similarity score.

Task 3 has been presented in the form of a Jupyter Notebook with input prompts
for accepting the image ID as well as the value of K. The outputs for this task
are visualized within the notebook.

If you provide an image ID without RGB channels, the system will print a
warning, but will proceed by converting the image an RGB image. Please note
that the task 2 did not store the vector models for non RGB images, so they
will not show up in the results.



Outputs:

The set of 5 image IDs asked to be queried for K similar images has been
stored within Outputs/ directory with relevant names as:

id_2500.pdf

Which signifies that for the Image ID 2500, the task_3.ipynb notebook
was run, and the resulting output was saved as a PDF. The PDF neatly
describes the rankings by each vector model by the chosen similarity
measure, along with the values of the measures for the requested
number of K.



Report:

The report is placed within Report/.