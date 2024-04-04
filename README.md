# Here are simple project, the end result of which is a dashboard into which you can upload a picture, and using an image classification model, get the probability that this picture belongs to a certain architectural style

Way to dashboard:
https://dashmosarch-pt6bybfyoq-ew.a.run.app

The model belongs image to one of 11 architectural styles. Mostly (but not at all) dataset was prepared by images of Moscow buildings. There are next styles: Brutalism, Church, Classicism, Constructivism (Bauhaus), Fortification (Russian fortreses of XVI - XVII centuries), Industrial_XIX (Mostly Factorys), Modern (Art Nouveau) , Palaty (Chambers, mostly Russian citisen buildings of XVII century, Panelka (living houses of second half of XX century), Socialist_classicism (close to Art Deco), XXI_century (contemporary architecture).
There are few files and one folder in this repository:
- Folder with files for deploying of dashboard
- 01_organize dataset - in this file I organised dataset to .csv (I decided that this path would be more convenient for me, in addition, at first I planned to add a prediction of the year of construction and that would be even more convenient)
- 02_mos_arch_fitting - it is main file which contains architecture of model and process of fitting model
- 03.1_dash_mos_arch_for_deploy - this file which I use as app.py in dashboard (here second version which adopted to work whith model in onnx format to avoid addition of tensorflow in requirements.txt)
- 03_dash_mos_arch - same file without onnx, wich work with tensorflow
- 04_similarity_images - in this file i made some experiments to visualise similarity of architectural style using my model
- 05_CAM_mos_arch - in this file I tryed to make some visualisation approaches which often uses for interpetate models of image classification (Grad-CAM, Guided backpropogation, Guided Grad-CAM)
