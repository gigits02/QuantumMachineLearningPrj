#!/bin/bash

echo "Creazione dell'ambiente virtuale..."
python3 -m venv qaoa_env

echo "Attivazione dell'ambiente virtuale..."
source qaoa_env/bin/activate

echo "Aggiornamento di pip..."
pip install --upgrade pip

echo "Installazione delle dipendenze..."
pip install -r requirements.txt

echo ""
echo "Setup completato con successo!"
echo "Per attivare l'ambiente in futuro, esegui:"
echo "source qaoa_env/bin/activate"