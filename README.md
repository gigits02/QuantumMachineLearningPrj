Graph Coloring con QAOA
=========================

Questa repository contiene diverse implementazioni del protocollo QAOA (Quantum Approximate Optimization Algorithm) applicato al problema del **graph coloring**. Le varianti presenti utilizzano differenti codifiche dei qubit (one-hot e binaria) e un'estensione concettuale basata su **qudit**, ovvero stati quantistici con d livelli.

Contenuto della repository
--------------------------
I seguenti file sono presenti nella repository:

1. `graphColOne.py`    - Versione compatta del codice *one-hot*. Restituisce il numero cromatico dato un grafo in input.
2. `graphColBin.py`    - Versione compatta del codice *binario*. Calcola il numero cromatico del grafo.
3. `graphColQd.py`     - Implementazione compatta del metodo basato su qudit. Ritorna il numero cromatico stimato.

Caratteristiche principali
--------------------------
- **QAOA** come strategia di risoluzione del problema combinatorio.
- **Codifiche multiple** per confrontare efficienza e consumo di risorse:
  - One-hot (più qubit, codifica semplice)
  - Binaria (meno qubit, codifica più compatta)
  - Qudit (versatilità teorica, ancora esplorativa)
- **Node-fixing** per migliorare ulteriormente l'efficienza della risoluzione.
- Ogni file `.py` è progettato per l'esecuzione diretta su un grafo definito, restituendo il **numero cromatico** stimato.

Requisiti
---------
- Python 3.8+
- [PennyLane](https://pennylane.ai/) (per circuiti quantistici e ottimizzazione)
- NumPy, NetworkX, Matplotlib (per grafo e visualizzazione)
- (Opzionale) Jupyter Notebook per l'esecuzione interattiva

Esecuzione
----------
Nella cartella vi è il file "requirements.txt" che contiene tutto il necessario per far girare i codici.
È inoltre presente uno script pronto all'utilizzo per l'installazione: in questo modo verrà creato direttamente da terminale l'ambiente virtuale "qaoa_env" con tutto il necessario.
Per Linux/macOS:
chmod +x setup.sh
./setup.sh
Per Windows:
setup.bat
