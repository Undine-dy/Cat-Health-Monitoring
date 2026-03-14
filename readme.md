Cat-Health-Monitoring
An electronic cat can use an SVC model to evaluate the health indicators of people carrying electronic devices such as blood pressure and mobile phone gyroscopes, detect their movement status, and link LLM to provide medical advice

Download the required database through setup. py

Quick Start Steps
1. Environmental preparation
First, create the Conda environment and install the dependencies:
# 1. Install/update dependencies (remember to replace local path)
conda create -n "moon_project" python=3.9
#Enter the project directory
cd C:\Users\26710\Desktop\work\moon\TeamResearch\fall_detection_backend
#Install dependencies (using Tsinghua Source Acceleration)
conda run -n "moon_project" pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

# 2. Verify import
conda run -n "moon_project" python -c "import main, fusion_main, requests, pandas; print('import success')"

# 3. Launch the old backend (new window)
cd C:\Users\26710\Desktop\work\moon\TeamResearch\fall_detection_backend
conda run -n "moon_project" python manage.py serve-legacy
The service will be available http://localhost:8000 Start up.

# 4. Interface testing (new window) Testing APIs in a new terminal window:
cd C:\Users\26710\Desktop\work\moon\TeamResearch\fall_detection_backend
conda run -n "moon_project" python manage.py test-api

# 5. Real data testing (new window) using real data for end-to-end testing:
cd C:\Users\26710\Desktop\work\moon\TeamResearch\fall_detection_backend
conda run -n "moon_project" python manage.py test-real
