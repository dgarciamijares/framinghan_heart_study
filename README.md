# Framingham Heart Study Analysis

## Project Overview

This project involves analyzing data from the Framingham Heart Study to predict the risk of Coronary Heart Disease (CHD) within the next 10 years using logistic regression. The project includes training a logistic regression model on a training dataset and evaluating its performance on a test dataset. Key metrics such as accuracy, True Positive Rate (TPR), and False Positive Rate (FPR) are calculated to assess the model's effectiveness. Additionally, the project includes a cost-benefit analysis to determine an optimal threshold for prescribing preventive medication based on the predicted CHD risk.

## Datasets

- `framingham_train.csv`: Training dataset with 2560 data points.
- `framingham_test.csv`: Test dataset with 1098 data points.

## Project Structure

- `README.md`: Overview of the project.
- `analysis.md`: Detailed analysis and results.
- `code`: Directory containing the Jupyter notebook and other scripts used for analysis.
- `data`: Directory containing the training and test datasets.
- `images`: Directory containing images used in the markdown files.

## Key Files

- `framingham_train.csv`: Training data used for building the logistic regression model.
- `framingham_test.csv`: Test data used for evaluating the model.

## Technologies Used

- **Python**:
  - **Pandas**: For data manipulation and analysis.
  - **Statsmodels**: For building the logistic regression model.
  - **Scikit-learn**: For model evaluation and metrics.
  - **Matplotlib**: For plotting the ROC curve and other visualizations.
- **Jupyter Notebook**: For interactive coding and documenting the analysis.

## How to Use

1. Clone the repository.
2. Navigate to the `code` directory and open the Jupyter notebook `framingham_heart_study.ipynb`.
3. Run the notebook to see the analysis and results.
4. Refer to the `analysis.md` file for a detailed explanation of the analysis and findings.

## Results Summary

The logistic regression model was trained on the Framingham dataset to predict the 10-year risk of CHD. The model identified significant risk factors and provided a probability threshold for prescribing preventive medication. Key metrics such as accuracy, TPR, and FPR were used to evaluate the model's performance. Additionally, a cost-benefit analysis was conducted to estimate the economic impact of prescribing medication based on the model's predictions.

## License

This project was developed as part of a course in the Industrial Engineering and Operations Research (IEOR) department at UC Berkeley. Please credit the course, school, and professor as follows:

- Course: IEOR 242A - Machine Learning and Data Analytics
- School: University of California, Berkeley
- Professor: Prof. Paul Grigas
