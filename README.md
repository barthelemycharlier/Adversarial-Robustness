


This is a basic code repository for Assignment on adversarial attack done at PSL.

The repository contains a basic model and a basic training and testing
procedure. It will work on the testing-platform (but it will not
perform well against adversarial examples). The goal of the project is
to train a new model that is as robust as possible.

# Basic usage

Install python dependencies with pip: 

    $ pip install -r requirements.txt

Test the basic model:

    $ ./model.py
    Testing with model from 'models/default_model.pth'. 
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
    100.0%
    Extracting ./data/cifar-10-python.tar.gz to ./data/
    Model natural accuracy (test): 53.07

(Re)train the basic model:

    $ ./model.py --force-train
    Training model
    models/default_model.pth
    Files already downloaded and verified
    Starting training
    [1,   500] loss: 0.576
    [1,  1000] loss: 0.575
    ...

Train/test the basic model and store the weights to a different file:

    $ ./model.py --model-file models/mymodel.pth
    ...

Load the module project and test it as close as it will be tested on the testing plateform:

    $ ./test_project.py

Even safer: do it from a different directory:

    $ mkdir tmp
    $ cd /tmp
    $ ../test_project.py ../

# Modifying the project

You can modify anything inside this git repository, it will work as long as:

- it contains a `model.py` file in the root directory
- the `model.py` file contains a class called `Net` derived from `torch.nn.Module`
- the `Net` class has a function call `load_for_testing()` that initializes the model for testing (typically by setting the weights properly).  The default load_for_testing() loads and store weights from a model file, you will also need to make sure the repos contains a model file that can be loaded into the `Net` architecture using Net.load(model_file).
- You may modify this `README.md` file. 

# Before pushing

When you have made improvements your version of the git repository:

1. Add and commit every new/modified file to the git repository, including your model files in models/.(Check with `git status`) *DO NOT CHECK THE DATA IN PLEASE!!!!*
2. Run `test_project.py` and verify the default model file used by load_for_testing() is the model file that you actually want to use for testing on the platform. 
3. Push your last change

Note: If you want to avoid any problems, it is a good idea to make a local copy of your repos (with `git clone <repos> <repos-copy>`) and to test the project inside this local copy.

Good luck!

# MeanSparse Post-Training Defense

## Approach

The **MeanSparse** method is a post-training defense technique that improves adversarial robustness without retraining the model. It's based on the observation that adversarial attacks often exploit tiny fluctuations in the feature maps that occur close to the mean value of each channel—fluctuations that carry little useful information.

### How it works

For each feature channel in the network:

1. **Compute statistics** on the training set:
   - Channel-wise mean: μ_ch
   - Channel-wise standard deviation: σ_ch

2. **Define a sparsification threshold** using hyperparameter α:
   - Threshold = α × σ_ch

3. **Block small deviations** at inference time:
   - Any activation within the interval [μ_ch - α×σ_ch, μ_ch + α×σ_ch] is replaced with μ_ch
   - Values outside this region remain unchanged

This is implemented as a PyTorch `MeanSparse` module that is inserted before each ReLU activation in our WideResNet model (already trained with adversarial training). The only hyperparameter to tune is the threshold α, which we empirically optimized.

**Note:** We also tested retraining the WideResNet with GeLU activations instead of ReLU, but this did not yield any improvement.

## Experimental Results

| Alpha | Natural Acc. | L∞ PGD | L₂ PGD |
|-------|-------------|--------|--------|
| baseline | 80.6% | 47.7% | 43.9% |
| 0.001 | **80.6%** | 47.9% | 43.8% |
| 0.010 | 80.6% | 47.9% | 43.9% |
| 0.050 | 80.6% | 48.0% | 44.7% |
| 0.100 | 80.6% | 48.7% | 45.8% |
| 0.150 | 80.4% | 49.5% | 46.8% |
| 0.190 | 80.2% | 50.3% | 47.5% |
| 0.200 | 80.2% | 50.7% | 48.1% |
| 0.250 | 80.3% | 53.1% | 49.8% |
| 0.300 | 80.5% | 56.5% | 55.0% |
| **0.320** | 79.8% | **58.3%** | **57.4%** |
| 0.330 | 79.7% | 57.2% | 57.2% |
| 0.350 | 38.9% | 21.6% | 24.5% |

### Key Findings

The MeanSparse post-training approach worked very well on our WideResNet model:

- **+10.6 percentage points** improvement on L∞ PGD attacks (47.7% → 58.3%)
- **+13.5 percentage points** improvement on L₂ PGD attacks (43.9% → 57.4%)
- Natural accuracy remains nearly intact (80.6% → 79.8%) at optimal α = 0.32
- The model was originally trained against L∞ attacks, yet we see significant gains on L₂ as well

### Interpretation

This substantial improvement suggests that our adversarially-trained model had already learned precise and robust representations, but was still vulnerable to noise in its activations. The attacks were exploiting these small fluctuations to corrupt the output. 

MeanSparse effectively addresses this flaw by filtering out the noise that adversarial perturbations introduce near the mean values, allowing the model to leverage its robust learned features more effectively.
