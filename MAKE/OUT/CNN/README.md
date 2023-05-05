# Folder Hierarchy Explanation

CNN
|
|-alpha=0.01
|      |
|      |-kappa=0.09.png
|      |
|      |-kappa=0.9.png
|      |
|      |-kappa=0.99.png
|
|-alpha=0.1
|      |
|      |-kappa=0.09.png
|      |
|      |-kappa=0.9.png
|      |
|      |-kappa=0.99.png
|
|-alpha=1.0
|      |
|      |-kappa=0.09.png
|      |
|      |-kappa=0.9.png
|      |
|      |-kappa=0.99.png

9 total CNN models have been trained, one for each set of alpha lr values={0.01, 0.1, 1.0}={'small', 'good', 'large'} and for each set of mu optimizer lr values={0.09, 0.9, 0.99}={'small', 'good', 'large'}.
