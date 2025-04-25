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

