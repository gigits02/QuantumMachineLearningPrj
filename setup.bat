@echo off

echo Creating virtual environment...
python -m venv qaoa_env

echo Activating environment...
call qaoa_env\Scripts\activate.bat

echo Upgrading pip...
pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete!
echo To activate the environment later, run:
echo qaoa_env\Scripts\activate.bat