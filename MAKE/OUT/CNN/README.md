# Folder Hierarchy Explanation

.
└── CNN/
    ├── alpha=0.01/
    │   ├── mu=0.09.png
    │   ├── mu=0.9.png
    │   └── mu=0.99.png
    ├── alpha=0.1/
    │   ├── mu=0.09.png
    │   ├── mu=0.9.png
    │   └── mu=0.99.png
    └── alpha=1.0/
        ├── mu=0.09.png
        ├── mu=0.9.png
        └── mu=0.99.png

9 total CNN models have been trained, one for each set of alpha lr values={0.01, 0.1, 1.0}={'small', 'good', 'large'} and for each set of mu optimizer lr values={0.09, 0.9, 0.99}={'small', 'good', 'large'}.
