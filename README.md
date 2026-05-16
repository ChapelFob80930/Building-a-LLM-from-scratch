# Building a Large Language Model from Scratch

A comprehensive educational project demonstrating deep understanding of transformer-based language model architecture, implementation, and advanced techniques. This repository showcases hands-on experience with core ML/AI concepts including self-attention mechanisms, model training, parameter-efficient fine-tuning, and production-ready text generation.

---

## 🎯 Project Overview

This project implements a **GPT-style transformer model** from the ground up, providing complete insight into how modern large language models work. The implementation covers the full pipeline from tokenization and embedding to training, evaluation, and fine-tuning with practical applications.

### Key Capabilities
- **Transformer Architecture**: Multi-head self-attention, transformer blocks, positional embeddings
- **Model Training**: Training loops with loss tracking, accuracy metrics, and validation
- **Advanced Fine-tuning**: Parameter-efficient methods (LoRA), instruction tuning, classification tasks
- **Text Generation**: Token-based text generation with sampling strategies
- **Evaluation**: Custom metrics (GLEU), accuracy tracking, and loss analysis

---

## 📚 Core Components & Learning Outcomes

### 1. **Foundation & Tokenization**
- **`working_with_text_data.ipynb`** - Text preprocessing and tokenization strategies
- **`token_converter.py`** - Token-to-text and text-to-token conversion utilities
- **Concepts Demonstrated**: 
  - Character-level and subword tokenization
  - Vocabulary management
  - Sequence padding and batching

### 2. **Transformer Architecture**
- **`multi_head_attention.py`** - Multi-head self-attention mechanism with causal masking
  - Implements Q, K, V projections
  - Head splitting and recombination
  - Attention score computation with scaled dot-product
  - Dropout regularization
- **`transformer_block.py`** - Transformer block combining attention and feed-forward layers
- **`layer_normalization.py`** - Layer normalization implementation
- **`feed_forward_network.py`** - Feed-forward sub-layer with activation functions
- **`gpt_model.py`** - Complete GPT model architecture
  - Token and positional embeddings
  - Stacked transformer blocks
  - Output projection to vocabulary

### 3. **Model Implementation & Architecture**
- **`GPT_implmentation.ipynb`** - Step-by-step GPT model construction
- **`GPT_CONFIG_124M.py`** - Configuration for 124M parameter model
- **`self_attention.ipynb`** - Detailed self-attention mechanism walkthrough

### 4. **Training Pipeline**
- **`GPT_Model_Training_and_Evaluation.ipynb`** - Complete training workflow
  - Loss calculation and backpropagation
  - Validation on held-out data
  - Training visualization and monitoring
- **`train_model_simple.py`** - Simplified training script
- **`calc_loss_loader.py`** - Loss computation utilities
- **`evaluate_model.py`** - Model evaluation functions

### 5. **Advanced Fine-tuning Techniques**

#### Parameter-Efficient Fine-tuning
- **`Parameter-efficient-Finetuning-with-LoRA.ipynb`** - Low-Rank Adaptation (LoRA)
  - Reduces trainable parameters dramatically
  - Maintains model performance
  - Memory-efficient training

#### Instruction Tuning
- **`finetuing_for_instructions.ipynb`** - Instruction-following capability
- **`finetuning_for_instructions_ANNOTATED.ipynb`** - Annotated version with detailed explanations
- **`instruction-data.json` & `instruction-data-with-response.json`** - Training datasets
- **Concepts**: Prompt-response formatting, supervised fine-tuning (SFT)

#### Classification Fine-tuning
- **`finetuning_for_classification.ipynb`** - Text classification task adaptation
- **`train_classifier_simple.py`** - Classification training script
- **`calc_loss_loader_classification.py`** - Classification-specific loss computation
- **`calc_accuracy_loader.py`** - Accuracy metrics
- **Dataset**: SMS spam collection (5,574 messages)
  - **`sms_spam_collection.zip`** - Raw dataset
  - **`spam_dataset_prep.py`** - Data preprocessing pipeline
  - **`train.csv`, `validation.csv`, `test.csv`** - Train/val/test splits

### 6. **Text Generation**
- **`generate.py`** - Core text generation with configurable strategies
- **`generate_text_simple.py`** - Simplified generation interface
- **`generate_and_print_sample.py`** - Demonstration script
- **Techniques**: Token sampling, temperature scaling, greedy decoding

### 7. **Pre-trained Model Integration**
- **`gpt_download.py`** - Download pre-trained GPT-2 weights
- **`load_weights_into_gpt.py`** - Transfer learning: loading official weights into custom architecture
- **`gpt2-medium355M-sft.pth`** - Fine-tuned model checkpoint

### 8. **Evaluation & Visualization**
- **`GLEU.py`** - GLEU metric implementation (modified BLEU for generation)
- **`plot_losses.py`** - Loss curve visualization
- **`plot_values.py`** - Training metrics visualization
- **`loss-plot.pdf` & `accuracy-plot.pdf`** - Training results

### 9. **Training Loop Enhancements**
- **`Adding_Bells_and_Whistles_to_the_Training_Loop.ipynb`** - Advanced training techniques
  - Gradient accumulation
  - Learning rate scheduling
  - Checkpoint management
  - Model monitoring and logging

---

## 🔧 Technical Skills Demonstrated

### Deep Learning & ML
- ✅ Transformer architecture fundamentals
- ✅ Attention mechanisms (multi-head self-attention, causal masking)
- ✅ Neural network design and implementation
- ✅ Backpropagation and gradient-based optimization
- ✅ Loss functions and regularization techniques

### Training & Fine-tuning
- ✅ Supervised learning workflows
- ✅ Transfer learning and pre-trained models
- ✅ Parameter-efficient fine-tuning (LoRA)
- ✅ Multi-task learning (generation, classification, instruction-following)
- ✅ Hyperparameter tuning and model selection

### NLP Techniques
- ✅ Tokenization and vocabulary management
- ✅ Text preprocessing and data preparation
- ✅ Evaluation metrics (loss, accuracy, GLEU)
- ✅ Instruction tuning and prompt engineering
- ✅ Text classification and sequence modeling

### Software Engineering
- ✅ Modular code design (separate modules for attention, blocks, etc.)
- ✅ Configuration management (GPT_CONFIG_124M.py)
- ✅ Data pipeline development (spam_dataset_prep.py)
- ✅ Experiment tracking and visualization
- ✅ Documentation and reproducibility

### Tools & Libraries
- ✅ **PyTorch** - Deep learning framework
- ✅ **Jupyter Notebooks** - Exploratory analysis and education
- ✅ **Python** - Core programming language
- ✅ Data handling and visualization

---

## 📊 Project Highlights

### Architectural Insights
- **Multi-head Attention**: Implements scaled dot-product attention with 100+ line documentation and comments explaining each computation step
- **Modular Design**: Clean separation between attention, feed-forward, and transformer blocks
- **Configuration-driven**: Flexible model sizing through configuration dictionaries

### Training Results
- Successfully trained models on custom datasets
- Fine-tuned on instruction-following and classification tasks
- Generated coherent text sequences
- Achieved measurable improvements through advanced training techniques

### Data & Datasets
- **Text Generation**: "The Verdict" text file (~20K characters)
- **Classification**: SMS Spam Collection (5,574 samples, 80-10-10 train/val/test split)
- **Instruction Data**: Custom instruction-response pairs

---

## 🚀 How to Use

### Setup
```bash
# Clone the repository
git clone https://github.com/ChapelFob80930/Building-a-LLM-from-scratch.git
cd Building-a-LLM-from-scratch

# Install dependencies
pip install torch numpy pandas jupyter matplotlib
```

### Running Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open and run notebooks in order:
# 1. working_with_text_data.ipynb - learn tokenization
# 2. self_attention.ipynb - understand attention
# 3. GPT_implmentation.ipynb - build the model
# 4. GPT_Model_Training_and_Evaluation.ipynb - train it
# 5. Parameter-efficient-Finetuning-with-LoRA.ipynb - advanced techniques
```

### Training from Scratch
```bash
python train_model_simple.py
```

### Classification Task
```bash
python train_classifier_simple.py
```

### Text Generation
```bash
python generate_and_print_sample.py
```

---

## 📈 Learning Path & Progression

This project is structured to build understanding progressively:

1. **Fundamentals** → Tokenization and data handling
2. **Core Concepts** → Self-attention and transformer blocks
3. **Implementation** → Complete model architecture
4. **Training** → End-to-end training pipeline
5. **Advanced** → LoRA, instruction tuning, classification
6. **Production** → Text generation and deployment

Each notebook and script builds on previous concepts, creating a comprehensive learning journey.

---

## 💡 Key Learnings & Concepts

- **Why multi-head attention?** - Multiple representations of the same input allow the model to focus on different aspects simultaneously
- **Causal masking** - Prevents tokens from "cheating" by attending to future tokens during training
- **Positional embeddings** - Encodes token position information since attention is position-independent
- **Layer normalization** - Stabilizes training and improves convergence
- **LoRA** - Reduces fine-tuning parameters from millions to thousands while maintaining performance
- **Transfer learning** - Pre-trained weights initialize the model with general language understanding

---

## 🎓 Educational Value

This repository is ideal for:
- **Learning**: Understand transformer internals through code and detailed comments
- **Reference**: Well-documented implementations of key components
- **Experimentation**: Modify and test different architectures and training strategies
- **Portfolio**: Demonstrates deep understanding of modern NLP

---

## 📁 Repository Statistics

- **Jupyter Notebooks**: 95.8% - Exploratory analysis, education, and experimentation
- **Python Scripts**: 4.2% - Modular utilities and training pipelines
- **Total Components**: 40+ files including configs, data, and visualizations
- **Created**: February 2026
- **Last Updated**: May 2026

---

## 🔍 Code Quality

- ✅ Heavily commented code (especially in `multi_head_attention.py` with 50+ explanatory comments)
- ✅ Clear variable naming (queries, keys, values, attention_weights, etc.)
- ✅ Modular architecture (separate files for attention, layers, models)
- ✅ Configuration-based design for easy experimentation
- ✅ Type-aware tensor operations with shape documentation

---

## 🎯 Concepts Mastered

**Attention Mechanism**
- Scaled dot-product attention
- Multi-head attention
- Causal masking for auto-regressive generation

**Model Architecture**
- Transformer encoder/decoder blocks
- Embeddings (token + positional)
- Layer normalization
- Feed-forward networks

**Training Techniques**
- Loss computation and backpropagation
- Validation and model evaluation
- Learning rate scheduling
- Gradient accumulation
- Checkpoint management

**Fine-tuning Methods**
- Full parameter fine-tuning
- Parameter-efficient LoRA
- Multi-task learning
- Transfer learning from pre-trained models

**Applications**
- Text generation (language modeling)
- Text classification
- Instruction following
- Spam detection

---

## 📞 Next Steps

To extend this project:
- Experiment with different model sizes
- Implement quantization for inference
- Add more evaluation metrics
- Explore prompt engineering
- Deploy as a web service

---

**This project demonstrates a comprehensive understanding of modern language model architecture, training, and deployment — key skills for any ML/AI role.**
