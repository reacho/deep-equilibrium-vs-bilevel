# Deep Equilibrium Models versus Bilevel Learning - CelebA dataset

Here we train a model on the CelebA dataset for one of two possible tasks: denoising and deblurring. Instead of loading the whole dataset, we provide a smaller sample containing 20 images.
We also provide the implementation for having the inpainting operator, although we do not use it for this dataset.
Instead of using affine layers (as we did in the MNIST folder), we implement Conv2d layers here.

We already provide configuration files in the ```config``` folder. To run the code with different settings, change the configuration files in the folder or look at the MNIST folder for a code that automatically generates these JSON files.

To run the training code, use the following command
```
python DEQ-vs-bilevel-conv2d.py config/[NAME-config_file].json
```

A checkpoint with the model's parameters is automatically saved at the end of the training.
