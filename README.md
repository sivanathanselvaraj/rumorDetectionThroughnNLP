# Rumor Mill: Tracking Viral Rumors through Textual Analysis

## Description

This Streamlit app is designed to detect whether a news article is likely fake or real based on its content using Natural Language Processing (NLP) techniques. It allows users to input a news article, select a vectorizer and classifier, and then predicts the authenticity of the article.

## Modules

### Module 1: Import Necessary Packages
- **streamlit**: For creating the web application.
- **numpy**: For numerical computations.
- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For machine learning functionalities.
- **tensorflow**: For backend deep learning functionalities.
- **matplotlib**: For plotting and visualization.
- **seaborn**: For statistical data visualization built on top of matplotlib.
- **warnings**: For ignoring warnings.

### Module 2: Load the Dataset
- Loads the dataset containing fake and real news articles from a CSV file.
- Converts the labels to binary format (0 for real, 1 for fake).

### Module 3: Select Vectorizer and Classifier
- Allows users to select a vectorizer (TF-IDF or Bag of Words) and a classifier (Linear SVM or Naive Bayes) via the sidebar.

### Module 4: Train the Model
- Trains the selected classifier model using the chosen vectorizer and the loaded dataset.
- Caches the trained model for faster access.

### Module 5: Streamlit App
- Sets page configuration including title, icon, and layout.
- Displays the title and a PNG image for visualization.
- Hides the Streamlit style for a cleaner interface.
- Provides a text area for users to input news articles.
- Upon clicking the "Check" button, predicts the authenticity of the input news article using the trained model and displays the result.
- ![Rumor_output](https://github.com/Sri22082/rumorDetectionThroughNLP/assets/92198693/ab1e89a9-bca8-46e2-a3ae-98007c07d2ba)


## Usage

1. **Clone the Repository**: `git clone https://github.com/Sri22082/rumorDetectionThroughNLP.git`
2. **Navigate to Project Directory**: `cd rumorDetectionThroughNLP`
3. **Set Up Virtual Environment** (Optional but recommended):
   - **Create Virtual Environment**: `python -m venv env`
   - **Activate Virtual Environment**: 
     - On Windows: `.\env\Scripts\activate`
     - On macOS and Linux: `source env/bin/activate`
4. **Install Dependencies**: `pip install -r requirements.txt`
5. **Run the Streamlit App**: `streamlit run app.py --client.showErrorDetails=false`
   - This command removes cache error messages on the Streamlit interface.
6. **Input News Article**: Input a news article into the text area.
7. **Select Vectorizer and Classifier**: Choose a vectorizer and classifier from the sidebar.
8. **Check Prediction**: Click the "Check" button to see the prediction result.

## Credits

- This project was created by:
-  M.Srimanjunadh[Team Leader]
-  Sivanathan S
-  Name of The University : Puducherry Technological University
- Dataset source: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news

## License

This project is licensed under the [MIT License](LICENSE).
