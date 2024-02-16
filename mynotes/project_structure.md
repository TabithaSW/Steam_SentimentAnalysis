Project Structure Summary:


## Description

- **/app**: Contains the web application's components, including static files, templates, utility scripts, route definitions, and initialization script.

- **/data**: Stores the datasets used for training, testing, and the results of data analysis.

- **/final_analysis**: Contains files related to the final analysis phase of the project, potentially including sentiment analysis results.

- **/models**: Includes scripts for various machine learning models applied to analyze the data.

- **/tests**: Contains scripts for testing the web application's functionality.

- **Root Directory Files**: Essential files for project configuration, documentation, and execution.

This structure is designed to facilitate the development, analysis, and deployment of the Steam SentimentAnalysis project.

Steam_SentimentAnalysis/
│
├── app/ # Web application directory
│ ├── static/ # Static files (images, CSS, JS)
│ │ └── me.jpg # Example static image
│ ├── templates/ # HTML templates
│ │ └── home.html # Home page template
│ ├── utils/ # Utility scripts
│ │ ├── steam_scrape.py # Steam reviews scraping script
│ │ └── tokenize_csv.py # Review text tokenization script
│ ├── routes.py # Defines web app routes
│ └── init.py # Initializes the app as a package
│
├── data/ # Data directory for datasets and analysis results
│ ├── clustered_reviews_analysis.csv # Clustering and analysis results
│ ├── steam_reviews.csv # Raw Steam reviews dataset
│ ├── TEST_KMEANS_RESULTS.txt # KMeans clustering test results
│ ├── test_reviews.csv # Test review dataset
│ └── train_reviews.csv # Training review dataset
│
├── final_analysis/ # Directory for final analysis outputs
│ └── kmeans_vader_sentiment # Final sentiment analysis results
│
├── models/ # Machine learning models
│ ├── hier_cluster.py # Hierarchical clustering model script
│ ├── kmeans_cluster.py # KMeans clustering model script
│ ├── LDA_Model.py # LDA topic modeling script
│ └── train_model.py # Script for training models
│
├── tests/ # Test scripts for the application
│ └── test_app.py # Test script for app functionality
│
├── .gitignore # Specifies untracked files to ignore
├── README.md # Project description and documentation
├── requirements.txt # List of project dependencies
├── run.py # Script to run the web application
└── setup.py # Setup script for the project


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
- `requirements.txt`: Lists project dependencies.
- `run.py`: Script to run the web application.
- `setup.py`: Setup script for the project.
