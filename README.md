## 📄 Prompt Engineering to Inform Large Language Model in Automated Building Energy Modeling

**Gang Jiang**, **Zhihao Ma**, **Liang Zhang**, **Jianli Chen**

This repository provides guidelines for applying Large Language Models (LLMs) to facilitate Auto-Building Energy Modeling (ABEM) through prompt engineering. Unlike fine-tuning, which requires adjusting model weights, prompt engineering enables LLMs to generate Building Energy Models (BEMs) using natural language inputs. This approach allows users without specialized knowledge to create BEMs efficiently using natural language.

![Illustration of Prompting for Auto-building modeling](/figs/graphic.jpg)


**Paper is [HERE](https://doi.org/10.1016/j.energy.2025.134548)**


## 🖇 Key Contributions

Prompt Engineering for ABEM: Six types of prompts were designed to explore 18 open-source LLM capabilities across 648 case studies.
- Exploratory Task 1: Generates a simple building model with basic geometry in IDF format (~1k tokens).
- Exploratory Task 2: Creates a more complex model with multiple windows, internal loads, and varying WWRs (~2k tokens).
- Real-World Task 3: Produces a detailed model for a real-world modular test facility (iUnit) at National Renewable Energy Laboratory (NREL), including customizable materials, occupancy schedules, and energy settings (~5k tokens).

## 📍 Findings

- Few-shot and chain-of-thought prompting strategies improve ABEM generation.
- Compact LLMs with appropriate context windows are suitable for deployment in building applications.
- LLMs are capable of performing ABEM with one-shot learning.
- Excessive demonstrations and over-explanation can worsen ABEM generation and cause issues such as out-of-memory and token limitation. 

## 🚀 Quick Start

This repository includes designed prompts, case studies, and implementation details for prompting LLMs in ABEM.


📂 Repository Structure

```
    ── README.md                           # Project documentation
    ── Task1.py                            # Script for Exploratory Task 1 (Basic IDF model generation)
    ── Task1_test_dataset_example.json     # Example dataset for Task 1
    ── Task2.py                            # Script for Exploratory Task 2 (More complex IDF model)
    ── Task2_test_dataset_example.json     # Example dataset for Task 2
    ── Task3_real-world.py                 # Script for Real-world Task 3 (Detailed IDF model)
    ── Task3_real-world_rest-part.idf      # IDF file for real-world task
    ── GPT4o_Test.py                       # Script for testing ABEM performance using GPT-4o
    ── requirements.txt                    # Dependencies for this project
```

🔧 Installation

- Clone the repository:
```
    git clone https://github.com/Gangjiang1/Prompting-for-Auto-building-Modeling.git
    cd Prompting-for-Auto-building-Modeling
```
- Install required dependencies:
```
    pip install -r requirements.txt
```

▶️ Running the Tasks
- Exploratory Task 1: Run the basic IDF model generation script
`python Task1.py`
- Exploratory Task 2: Generate a more complex IDF model with multiple windows and internal loads:
`python Task2.py`
- Real-world Task 3: Generate a real-world building model in IDF format:
`python Task3_real-world.py`

📊 Testing with GPT-4o

To evaluate ABEM performance using GPT-4o, run:
`python GPT4o_Test.py`

### 📝 Citation

If you find this work useful, please cite our paper:
```
@article{jiang2025prompt,
  author    = {Gang Jiang and Zhihao Ma and Liang Zhang and Jianli Chen},
  title     = {Prompt engineering to inform large language models in automated building energy modeling},
  journal   = {Energy},
  volume    = {316},
  pages     = {134548},
  year      = {2025},
  month     = {Feb},
  doi       = {https://doi.org/10.1016/j.energy.2025.134548}
}
```
