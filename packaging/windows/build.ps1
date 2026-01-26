python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --onefile my_ai_package\main.py --name my-ai

