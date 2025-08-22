# iBitter-GRE
iBitter-GRE is a bitter peptides predictor, using ESM-2 and other traditional desriptors to extract features with stacking method.
The websever of iBitter-GRE is freely available at 
Note:
(1) The websever only accepts proteins with a length less than 1024
(2) If you have any questions regarding tasks, please send an email to 


### Reference
- Lv, J., Geng, A., Pan, Z., Wei, L., Zou, Q., Zhang, Z., & Cui, F. (2025).  
  *iBitter-GRE: A Novel Stacked Bitter Peptide Predictor with ESM-2 and Multi-View Features.*  
  Journal of Molecular Biology, 437(8), 169005.  
  https://doi.org/10.1016/j.jmb.2025.169005


## Webserver.
The webserver we provide can be access at [iBitter-GRE](http://121.36.197.223:45107/)




## Code
### 1. Built virtualenv
   
   (a). The python edition and imported packages with edition is in config.jason
   
### 2. Extract features
   
   (a). extract_features.py contains the codes we used when built our model.
   
   (b). If you don't want to use the websever you need to use the extract_features_use.py to extract 
       specific features to make sure the model can run successfully.
      _  You show download the index.csv to make sure this script works, and also change the path where it located in the extract_features_use.py_
   
### 3. Prediction

   (a). prediction.py contains the codes how we built this predictor
   
       _ You show change the path where the data the predictor need is in your local machine_
   
### 4. Data
   
   (a). File training.fasta and file testing.fasta contains the data used for training and testing, 
       respectively.
   
NOTE:
   All the path in the codes should replaced with your own path


   
