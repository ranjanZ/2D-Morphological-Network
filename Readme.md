# Code of the paper "Morphological Networks for Image De-raining"

| Input        | De-Rained      | 
|:-------------:|:-------------:|
| ![input](https://raw.githubusercontent.com/ranjanZ/2D-Morphological-Network/master/data/input_images/52_in.png)| ![De-Rained](https://raw.githubusercontent.com/ranjanZ/2D-Morphological-Network/master/data/output/52_4small.png) |

## Dependency
* For Running
    * Python2
    * keras
    * scipy
    * numpy
    * scikit-image
    * matplotlib

## Running
```
$ cd src/
$ python run.py  <rainy_image_dir>   <output_dir>
```
This runs the code in the supplied images.
```
$python run.py ../data/input_images/ ../data/output/

```

## Files
```
├── data
│   ├── input_images
│   │   └── GT_p2.jpg
│   └── output
│       ├── GT_p23cnn.jpg
│       └── GT_p2small_morpho_net_.jpg
├── models
│   ├── model_cnn.h5
│   ├── MorphoN.h5
│   ├── MorphoN_small.h5
│   ├── path1.h5
│   ├── path2.h5
│   ├── weights_cnn.h5
│   ├── weights_morphoN.h5
│   ├── weights_morphoN_small.h5
│   ├── weights_path1.h5
│   └── weights_path2.h5
├── Readme.md
└── src
    ├── generator.py
    ├── init.py
    ├── models.py
    ├── morph_layers2D.py
    ├── run.py
    └── utils.py

```

## Publication
Ranjan Mondal,Pulak Purkiat, Sanchayan Santra and Bhabatosh Chanda. "Morphological Networks for Image De-raining" International Conference on Discrete Geometry for Computer Imagery, 2019, ESIEE Paris, France.

