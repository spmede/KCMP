
## Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing

This repository contains the complete source code, dataset references, and documentation to facilitate the reproduction, application, and further development of our **KCMP** method.  
The implementation is written in Python. The paper

---
## 🚀 System Requirements & Environment Setup

All dependencies required by this project are listed in the `requirements.txt` file. To ensure compatibility and reproducibility, specific package versions have been included.

### 🧱 Step-by-Step Setup

After cloning the repository and navigating to the project root directory, follow these steps to set up the environment:


```bash
# 1. Create a new Conda environment named 'kcmp' with the desired Python version
conda create -n kcmp python=3.10.16 -y

# 2. Activate the Conda environment
conda activate kcmp

# 3. Install required Python dependencies via pip
pip install -r requirements.txt
```

### 📦 Install OpenAI CLIP

This project requires the OpenAI CLIP package, which is not available via PyPI. Please follow the steps below to install it manually:

```bash
# 1. Clone the CLIP repository
git clone https://github.com/openai/CLIP.git

# 2. Navigate to the CLIP folder (adjust the path if needed)
cd CLIP

# 3. Install CLIP as a local Python package
pip install .
```

---

### 📁 Used Data

We use the [JaineLi/VL-MIA-image](https://huggingface.co/datasets/JaineLi/VL-MIA-image) dataset, which was open-sourced as part of the paper *Membership Inference Attacks against Large Vision-Language Models*.

Please download the dataset and place it under the `VL-MIA-image/` directory in the project root.


### 🧠 Used Models

We use `MiniGPT-4`, `LLaVA 1.5`, and `LLaMA-Adapter V2` as target models in our experiments.  
The corresponding model weights need to be downloaded and properly configured before use.

Please download the required models and update the model loading paths in the following scripts:

- `load_model_utili.py`  
- `minigpt4_infer.py`  
- `llava_infer.py`  
- `llama_adapter_infer.py`

> ⚠️ Ensure that the directory paths point to the correct locations of your downloaded models.


---
### 📂 Source Code Overview

We provide the complete codebase for implementing the KCMP method from scratch.
The following sections explain the purpose and usage of each core script.

#### `ObjColor_image_analysis.py`

This script primarily utilizes SAM (Segment Anything Model) to detect objects in images, and then leverages a multimodal API model (gpt-4o-mini in our experiment) to analyze the **class** and **color** of each detected object.

**Note:**  
You must configure the `client` in `infer_utili/apiCallFunc.py` to successfully connect to the API service.  
Alternatively, you can replace the API with an open-source model (e.g., Qwen-2.5-VL). To do this, modify the `apiCall_img` function to accept image and instruction inputs, and return a text-based response.


#### `ObjColor_confuser_gen_noFilter.py` and `ObjColor_confuser_gen_Filter.py`

These two scripts generate confuser options used to construct multiple-choice questions based on the object/color analysis results from the previous step:


- `ObjColor_confuser_gen_noFilter.py`:  
  Generates confuser options for **all** detected objects and colors without any filtering. All detected instances are used to construct questions, regardless of quality.

- `ObjColor_confuser_gen_Filter.py`:  
  Introduces a **filtering mechanism** to assess the quality and validity of questions and confuser options. Only high-quality, well-formed examples with strong detection potential are retained.


#### `ObjColor_traverse.py`

This script performs model inference using a specified target model and dataset.  
**Note:** A `confuser_json` file is required to construct multiple-choice questions.  
You can provide the result file generated from the previous step (i.e., confuser generation) as input.

#### `get_clipScore.py` and `question_select.ipynb`


These two scripts are designed to further improve the method's performance by selecting questions with the strongest probing ability.


- `get_clipScore.py`:  
  Generates a global caption for each image, and then computes CLIP scores between this caption and each image patch associated with a question.

- `question_select.ipynb`:  
  For each image, it ranks all candidate questions based on the CLIP scores of their associated image patches.  
  A specified number of top-ranked questions are then selected to participate in the final characteristic score computation.  
  The number of selected questions can be customized by the user.
