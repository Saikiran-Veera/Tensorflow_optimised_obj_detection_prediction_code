1. create the environment
   conda create -n Checkbox python=3.6

2. Activate environment
   conda activate Checkbox
3.change the directory to "Tensorflow_Optimized_Prediction_Code" and run following command.

   pip install -r requirements.txt
   
   python setup.py build
   
   python setup.py install
   
4. Place the Trained '.pb' file in to 'Trained_File' folder 

5. Place the 'label.pbtxt' file into 'Label_Map' folder

6. Run command: "python object_detection_Prediction.py" for prediction under activated environment
7. Include folder--> object_detection/core
                                      protos
                                      utils

