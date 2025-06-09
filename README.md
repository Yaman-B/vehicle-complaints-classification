# Vehicle Complaints Classification

## Overview

This project focuses on classifying consumer vehicle complaints from the National Highway Traffic Safety Administration (NHTSA) database into key component categories (e.g., brakes, steering, airbags). 

By analyzing these complaint summaries, the goal is to assist manufacturers and safety regulators in identifying patterns and trends related to vehicle component issues.

## Key Findings

**Model Performance:**
- **DistilBERT:** Outperforms classical models (Logistic Regression, SVM, Random Forest) in accuracy and macro F1 score. Shows clear improvements in harder-to-classify components.
- **Classical Models:** Perform surprisingly well with TF-IDF features. Logistic Regression and SVM achieve solid baseline performance (~75% accuracy).

**Resource Efficiency:**
- **DistilBERT:** Provides the best performance but at a significantly higher computational cost (3.5 hours training time on GPU).
- **Classical Models:** Train in minutes on CPU, with very competitive performance for practical use.

## Practical Considerations

**Model Tradeoffs:** While DistilBERT offers the highest accuracy, the improvement (~3% macro F1 gain) comes at a much higher computational cost. Classical models offer a strong balance of speed and accuracy for this task.

**Component Distinguishability:** Components with strong, specific language (e.g., *airbags*, *steering*) are well-classified across all models. More ambiguous components (e.g., *vehicle speed control*) remain challenging, though DistilBERT improves these cases.

**Model Deployment:** Classical models are highly suitable for lightweight applications where resources are constrained. DistilBERT is more appropriate where maximum accuracy is required and computational resources allow.

## Future Work

- **Model Deployment:** Explore deploying the best models via an interactive Streamlit application for public demonstration.
- **Further Tuning:** Experiment with advanced techniques (e.g., data augmentation, class balancing, DistilBERT hyperparameter tuning) to improve classification of weak classes.
- **Multi-label Classification:** Many complaints list multiple components. Future iterations could explore multi-label modeling to better reflect this reality.

## Recommendations

- **For Production:** Logistic Regression or SVM are suitable first choices due to their efficiency and solid accuracy.
- **For Research or High-Accuracy Use:** DistilBERT is recommended, particularly for improving performance on difficult classes.
- **For Future Projects:** Consider combining classical and deep learning approaches, or leveraging more recent Transformer architectures for further gains.

## Model Comparison

**Performance Metrics:**

| Model                  | Accuracy | Macro F1 | Notes                                           |
|------------------------|----------|----------|-------------------------------------------------|
| Logistic Regression    | 75%      | 0.75     | Strong baseline using TF-IDF features           |
| Support Vector Machine | 75%      | 0.74     | Comparable to Logistic Regression               |
| Random Forest          | 74%      | 0.73     | Performs surprisingly well given the sparse text data |
| DistilBERT             | 78%      | 0.78     | Best performance; improved on weaker classes, though at high computational cost |

## Hardware, Time, and Model Size

| Model                  | Hardware              | Time      | Model Size |
|------------------------|-----------------------|-----------|------------|
| Logistic Regression    | Google Colab CPU      | 41 seconds | Small      |
| Support Vector Machine | Google Colab CPU      | ~1 minute | Small      |
| Random Forest          | Google Colab CPU      | ~14 minutes | Medium     |
| DistilBERT             | Google Colab GPU      | ~3.5 hours | ~256MB     |

## Tools and Technologies

- **Pandas & scikit-learn:** Used for data wrangling, classical modeling, and evaluation.
- **Huggingface Transformers:** Used to fine-tune DistilBERT on the complaint text.
- **Google Colab:** Used for model training, with GPU support for DistilBERT.
- **Matplotlib & Seaborn:** Used for visualization and analysis.

## Conclusion

This project demonstrates that while deep learning models like DistilBERT can yield the highest accuracy for text classification tasks, classical models still perfrom great, especially when balancing resource efficiency and performance.

In this particular use case, the marginal gains from DistilBERT (~3% macro F1) must be weighed against the large increase in training time and compute requirements. For many practical applications, classical models will offer a better balance of efficiency and accuracy.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Credits

This project was inspired by best practices in text classification projects and follows a modeling approach similar to the [Amazon Product Reviews Sentiment Analysis Project](https://github.com/tuback/amazon-products-sentiment-analysis).
