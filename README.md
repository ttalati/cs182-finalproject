# cs182-finalproject
CS 182 Project: Model-Agnostic Pipelines for Latent Bias Detection and Intervention in LLMs

by Anushka Mukhopadhyay, Shraya Pal, William Xu, and Tanush Talati

## File Guide
### Part A, Technique 1: Classifying output embeddings corresponding to biased prompts

`clean_biased_prompts.ipynb`: takes the Crow Pairs dataset and groups by different bias-types for further analysis.
It generates different `csv` files containing paired data with prompts corresponding to bias. Generated `csv` files are of 
form `filtered_prompts_[bias_type].csv`

 `embedding_vis.ipynb`: contains code for taking the data inside the bias specific datasets and generating various classification 
 models on them, also exports the post-PCA projection prompt-response matrix so it can be used to create various classification models. There is also code to visualize
 clusterings of the model using PCA and t-SNE, however those results were not particularly insightful.

 `Classification_MLP.ipynb`: contains the weighted average of mean-hidden-layer token representation model
 that we present as the best-classifying model in the paper. Requires code in  `embedding_vis.ipynb` to be run first since that
 does the job of prompting GPT2 and storing the responses for downstream analysis.

 ### `steering.py`: Bias Direction Extraction and Activation Steering for Religious Bias

This script implements **Part B of the pipeline**: evaluating and modifying the internal representations of LLMs to steer bias. It performs two main tasks:

1. **Bias Direction Extraction** using the [RepE library](https://github.com/andyzoujm/representation-engineering/tree/main/repe):  
   It computes PCA-based representation directions from paired prompts labeled for disability bias. These directions correspond to bias axes in hidden state space and are used to probe model behavior layer-by-layer.

2. **Activation Steering during Inference**:  
   Using the computed bias direction, the script modifies hidden activations of the model at runtime to either **amplify** or **suppress** biased generations. Comparisons of baseline vs. steered generations are printed for qualitative analysis.

The full pipeline includes:
- Preparing training/test examples from a labeled disability bias dataset (`filtered_prompts_religion.csv`),
- Extracting bias directions via PCA over hidden state differences using the `rep-reading` pipeline,
- Probing and plotting accuracy of bias classification layer-by-layer,
- Running the `rep-control` pipeline with steering activations applied at middle transformer layers (-30 to -11),
- Saving and visualizing the test accuracy plot (`religion_bias_accuracy.png`).

#### Requirements

This script relies on the **RepE (Representation Engineering)** library. You must clone and install the RepE repo before running this file:

```bash
git clone https://github.com/andyzoujm/representation-engineering.git
cd representation-engineering/repe
pip install -e .


