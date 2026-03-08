# Basil Stress Detection: The Journey to 96% Accuracy and 0.94 f1-score for the minority class

This repository contains the source code for my machine learning project focused on identifying stress in basil plants using the **VGG16 architecture**. 

## Project Overview
The core of this project was an exploration of how to push a deep learning model beyond a "stagnant" baseline. 
The breakthrough wasn't just in the hyperparameters; it came from tackling the data directly through class weighting and augmentation.

### Detailed Documentation
I have documented every experiment, from the initial "92% Ceiling" phase to the final "Breakthrough," in a series of articles on my **Medium profile**:

👉 **[https://medium.com/@nacholucero](https://medium.com/@nacholucero)**

### The Roadmap to 96%
| Stage | Strategy | Best Accuracy | Technical Lesson Learned |
| :--- | :--- | :--- | :--- |
| **I. The 92% Ceiling** | VGG16 Initial Benchmark | 92% | Imbalanced data (61/39) causes a recall gap in the minority class. |
| **II. Why more training isn't always better** | Extending Epochs (LR 0.001) | 85–90% | LR 0.001 proves to be unstable; more epochs can lead to "violent" weight updates. |
| **III. The Fine Line Between Convergence and Chaos** | Chasing Stability (LR 0.0005) | 92% | Lowering LR doesn’t fix the underlying instability problem. |
| **IV. Surpassing the Limit via Data Balancing** | Applying Class Weights | 93% | Penalizing minority class errors more heavily (1.27 vs 0.82) breaks the 92% accuracy ceiling and improves the F1-Score. |
| **V. Breakthrough: Reaching 96% through Data Augmentation**| Data Augmentation | **96%** | Visual diversity prevents memorization and improves generalization. |



### Repository Structure
The scripts are located in the `.py files` folder:
* **`Medium - VGG16 -learning from scratch- final.py`**: Base code with focus on exploring the learning rate effect on the outcome of training the model from scratch.
* **`Medium - VGG16 - learning from scratch - balanced data -.py`**: Implementation of class weights to balance the cost function impact.
* **`Medium - VGG16 - learning from scratch - balanced data - augmentation.py`**: The final optimized setup combining balancing with random visual transformations for better generalization.

### What's Next?
To follow the 96% accuracy achieved here my next experiments will involve:
1. **Species Universality**: Testing this exact setup on other crops to see if the lessons learned are universal.
2. **Real-World Inference**: Taking my own photos of basil plants to test the model's performance in field conditions.
