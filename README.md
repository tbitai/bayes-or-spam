# Bayes or Spam?

<img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg" height="16"> [Run in Google Colab](https://colab.research.google.com/github/tbitai/bayes-or-spam/blob/main/bayes_or_spam.ipynb)
&nbsp;
<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" height="16"> [Demo on Hugging Face](https://huggingface.co/spaces/tbitai/bayes-or-spam)

Spam filtering with Bayesian epistemology.

## Development

In the `requirements.txt`, the Intel Optimizations for TensorFlow package (`intel-tensorflow`) is listed. Change that to 
the official `tensorflow` if you're not on an Intel CPU. 

You'll need a Google [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) set in the `GEMINI_API_KEY` 
environment variable.
