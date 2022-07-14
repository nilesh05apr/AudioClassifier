# AudioClassifier
Binary Classification of Audio recordings of patients as healthy and non healthy  
Used librosa for audio feature extraction   

To Execute
- Download the dataset from here: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge    
- git clone https://github.com/nilesh05apr/AudioClassifier  
- cd AudioClassifier  
- extract Dataset and place it inside inputs folder  
- install requiremnts using : pip install -r requirements.txt  
- python3 main.py   

  
Models   
- ann for Neural Network  
- knn for Kth Nearest Neighbour  
- rfc for Random Forest Classifier  
- sgd for Stochastic Gradient Classifier  
- dtc for Decision Tree Classifier  
- svm for Support Vector Machine  
- mlp Multi Layer Perceptron (Sklearn)  
- all to compare all models  

python3 main.py --model [model name]   
python3 main.py --model [model name] --metric True  