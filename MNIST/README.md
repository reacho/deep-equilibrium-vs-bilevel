# Deep Equilibrium Models versus Bilevel Learning - MNIST dataset

Here we train a model on the MNIST dataset for one of three possible tasks: denoising, deblurring, and inpainting.

First, generate the config folder and all the config files running
```
python generate_config_json.py
```

Then, run the code with the following command
```
python DEQ-vs-bilevel.py config/[NAME-config_file].json
```

A checkpoint with the model's parameters is automatically saved at the end of the training.
