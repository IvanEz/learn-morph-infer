# Usage
To run the code in training mode excute the train.sh 

To run the code in testing mode excute the test.sh

The implementation is adopted from https://github.com/byungsook/deep-fluids

- To train network with no thresholding do:
    1. Set outputparams = 6
    2. In inverse_preprocess_single set thrvolume = data['data']
    3. In inverse_preprocess_single start paramsarray[0] from Dw, removing uth
    4. Adjust paramsarray size in declaration
    
    > more changes needed for test() 
    
- To remove certain output parameters:
    1. Set outputparams = desired number
    2. In inverse_preprocess_single remove unwanted outputparams
    3. Adjust paramsarray size in declaration
    
    > more changes needed for test()

- Inverse parameters:
    * Train on data [starttraindata, endtraindata]
    * Validate on [startvaldata, endvaldata]
    
    *Make sure they do not overlap! Otherwise some validation data in training set!*
    
    * Number of neurons in hidden layer: fcsize
    * Number of fully connected *hidden* layers: fchdepth
