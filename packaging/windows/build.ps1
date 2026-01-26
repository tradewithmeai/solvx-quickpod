python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --onefile solvx_quickpod\main.py --name solvx-quickpod
