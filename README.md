# iBitter-GRE
BitterTree-Stacking is a bitter peptides predictor, using ESM-2 and other traditional desriptors to extract features with stacking method.
The websever of iBitter-GRE is freely available at 
Note:
(1) The websever only accepts proteins with a length less than 1024
(2) If you have any questions regarding tasks, please send an email to 


## Reference.


## Webserver.
The webserver we provide can be access at [iBitter-GRE](http://121.36.197.223:45910/)




## Code
### 1. Built virtualenv
   
   (a). The python edition and imported packages with edition is in config.jason
   
### 2. Extract features
   
   (a). extract_features.py contains the codes we used when built our model.
   
   (b). If you don't want to use the websever you need to use the extract_features_use.py to extract 
       specific features to make sure the model can run successfully.
        - You show download the index.csv to make sure this script works, and also change the path where it located in the extract_features_use.py
   
### 3. Prediction

   (a). prediction.py contains the codes how we built this predictor
   
        - You show change the path where the data the predictor need is in your local machine
   
   (b). If you just want to use the model that we've built download the stacking.pkl, scaler.pkl and use_predictor.py
   
        - You show change the path where the stacking.pkl, scaler.pkl and the data the predictor need are in your local machine
   
### 4. Data
   
   (a). File training.fasta and file testing.fasta contains the data used for training and testing, 
       respectively.
   
   (b). File training.csv and file training.csv contains the data after extraction and selected,
        respectively.
   
NOTE:
   All the path in the codes should replaced with your own path
