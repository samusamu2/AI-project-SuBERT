# 2-vision_task_glyphs_recognition

In this folder we will implement our vision task. Unfortunately, we would need to label lots of images of Sumerian tablets to obtain a nice mworking classifier, which is out of the scope of this project. Instead, we will just explore the expressive potential of the YOLO model by creating some noisy synthetic sentences from the Sumerian Font. More on this is commented on the report.

Here are the files in this folder:
- `1_create-glyph-images.ipynb`: create images for single glyphs from the `NotoSansCuneiform-Regular.ttf` font.
- `2_generate-sentences.ipynb`: generate sentences by picking a number of random glyphs. Some transformations are applied to glyphs to make sentences noisy, such as small traslations, resizings, blurring and masking.
- `3_yolo-classifier.ipynb`: this notebook contains the actual implementation of the YOLO classifier.
- `3b_yolo-classifier.py`: this file contains the same model as designed in the previous notebook, but in a Python format*.
- `4_F-CNN.ipynb`: implementation of the Faster CNN.
- `5_create-synthetic-tablets.ipynb`: this file creates some synthetic version of complete tablets. It may be useful for testing purposes on our YOLO classifier. Again, we remark that this serves just as a test on the expressive power of this model, and we are aware that the model as it is cannot be used to implement a full translation pipeline from Sumerian tablets (or linart) images to translated text.


\* <b>Note on Python files:</b>
throughout this project, you'll notice that several Jupyter notebooks have corresponding Python script (.py) counterparts. This dual format serves a practical purpose: while notebooks provide an excellent interactive development environment with inline visualizations and step-by-step execution, they are less suitable for automated or long-running processes. The Python scripts enable seamless execution in virtual machines and remote servers, particularly for computationally intensive tasks that need to run unattended (such as overnight training runs or batch processing jobs). This approach allows us to leverage the exploratory advantages of notebooks during development while maintaining the operational flexibility needed for production-like environments.

