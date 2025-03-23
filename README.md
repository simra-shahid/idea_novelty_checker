# Scientific Idea Novelty Evaluation

Automated scientific idea generation systems have made remarkable progress, yet the automatic evaluation of idea novelty remains a critical and underexplored challenge. Manual evaluation of novelty through literature review is labor-intensive, prone to error due to subjectivity, and impractical at scale. To address these issues, we propose the **Idea Novelty Checker**, an LLM-based retrieval-augmented generation (RAG) framework that leverages a two-stage  retrieve-then-rerank approach. The **Idea Novelty Checker** first collects a broad set of relevant papers using keyword and snippet-based retrieval, then refines this collection through embedding-based filtering followed by facet-based LLM re-ranking. It incorporates expert-labeled examples to guide the system in comparing papers for novelty evaluation and in generating literature-grounded reasoning. This repository includes both the AI Scientist's novelty checker and our custom **Idea Novelty Checker** implementation. 

📄 [Read the paper here!](assets/paper.pdf)

![Figure](assets/image.png)


## Setup Instructions

### Option 1: Using Conda

1. **Create the Environment:**  
   Generate the environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate the Environment:**  
   ```bash
   conda activate nc_env
   ```

### Option 2: Using pip and virtualenv (or venv)

1. **Create a Virtual Environment:**  
   ```bash
   python3 -m venv nc_env
   ```
2. **Activate the Virtual Environment:**
   ```bash
   source nc_env/bin/activate
   ```
3. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

## Setup Keys and Default Parameters

```bash
python noveltychecker/utils/load_env.py
```

## Example Run

To evaluate a scientific idea, run the main script with your input idea, corresponding paper IDs, and any additional parameters. For example:
```bash
python main.py --idea "Hierarchical Topic Models (HTMs) are useful for discovering topic hierarchies in a collection of documents. However, traditional HTMs often produce hierarchies where lower-level topics are unrelated and not specific enough to their higher-level topics. Additionally, these methods can be computationally expensive. We present HyHTM - a Hyperbolic geometry based Hierarchical Topic Models - that addresses these limitations by incorporating hierarchical information from hyperbolic geometry to explicitly model hierarchies in topic models. Experimental results with four baselines show that HyHTM can better attend to parent-child relationships among topics. HyHTM produces coherent topic hierarchies that specialize in granularity from generic higher-level topics to specific lower-level topics. Further, our model is significantly faster and leaves a much smaller memory footprint than our best-performing baseline. We have made the source code for our algorithm publicly accessible." --papers "220046811, 267211735"
```

## Contributing

Contributions are welcome! If you encounter any issues or have ideas for improvement, please open an issue or submit a pull request.

---

