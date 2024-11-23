Project Structure Summary:


## Description

- **/app**: Contains the web application's components, including static files, templates, utility scripts, route definitions, and initialization script.

- **/data**: Stores the datasets used for training, testing, and the results of data analysis.

- **/final_analysis**: Contains files related to the final analysis phase of the project, potentially including sentiment analysis results.

- **/models**: Includes scripts for various machine learning models applied to analyze the data.

- **/tests**: Contains scripts for testing the web application's functionality.

- **/mynotes**: Contains logs and project structure for progress on analysis and models.

- **/utils**: Contains scripts for collecting, cleaning, preprocessing, and analyzing data before putting it into the models. Current Scripts: 1. TF-IDF scores that can be used as features for further analysis, such as clustering (e.g., K-Means) or sentiment analysis, 2. Silhoutte Score method to help determine the best number of clusters for kmeans, 3. A scraping tool to collect the data (steam reviews), 4. A way to summarize the sentiment distribution, 5. Tokenizer and train/test script for advanced and thorough preprocessing pipeline for text data, specifically tailored to gaming reviews.



- **Root Directory Files**: Essential files for project configuration, documentation, and execution.

This structure is designed to facilitate the development, analysis, and deployment of the Steam SentimentAnalysis project.


Secondary overview as things adapt:
# Steam_SentimentAnalysis Project Structure

## Overview
This structure outlines the organization of the Steam_SentimentAnalysis project, including directories for the web application, data storage, model scripts, and analysis results.

### Application Directory (`/app`)
- `static/`: Contains static files such as images, CSS, and JavaScript.
  - `me.jpg`: An example static image.
- `templates/`: Stores HTML templates for the web application.
  - `home.html`: The home page template.
- `utils/`: Utility scripts for various tasks.
  - `steam_scrape.py`: Script for scraping Steam reviews.
  - `tokenize_csv.py`: Script for tokenizing and processing review text.
- `routes.py`: Defines the web application's routes.
- `__init__.py`: Initializes the application as a Python package.

### Data Directory (`/data`)
- Stores datasets and results from analyses.
- Includes raw Steam reviews, test and training datasets, and KMeans clustering test results.

### Final Analysis Directory (`/final_analysis`)
- Contains outputs from the final phase of analysis, such as sentiment analysis results.

### Models Directory (`/models`)
- Scripts for machine learning models and training.
- Includes scripts for hierarchical clustering, KMeans clustering, LDA topic modeling, and a general model training script.

### Tests Directory (`/tests`)
- Contains test scripts for the application's functionality.

### Root Directory Files
- `.gitignore`: Specifies files to be ignored by version control.
- `README.md`: Project description and documentation.
- `run.py`: Script to run the web application.
- `setup.py`: Setup script for the project.

**Toolset:**
- Flask
- Python 
- NLTK
- Scikit-Learn
- Gensim
- Pandas
- Transformers
- Feather/CSV Files
- Git and GitHub