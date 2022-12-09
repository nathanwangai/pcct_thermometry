# Photon-counting CT thermometry via material decomposition and machine learning

Code underlying the results of the above paper (to appear in Springer Visual Computing for Industry, Biomedicine, and Art)

## Key result
A fully connected neural network trained to predict temperature from spectrally-resolved attenuation values of a set of basis materials (water, 50 mM CaCl2, 600 mM CaCl2) generalizes well to other materials (300 mM CaCl2 and a protein shake).

![](https://user-images.githubusercontent.com/98730743/201267743-736cafe2-b889-4b23-bda7-8658eb50ed56.png) | ![](https://user-images.githubusercontent.com/98730743/201267753-2f4f3e8e-ce4a-4e94-bb27-2d4778fa2f4a.png)

## Contents 
- atteunation_csv\: folder of attenuation data at all energies and temperatures collected in this project
- temp_MLP\: saved Tensorflow model of the temperature predicting neural network
- projection_processing.ipynb: Python notebook that contains all data processing and experiments
  - Module scripts: spectral_scan.py, temperature_prediction.py, visualizations.py
