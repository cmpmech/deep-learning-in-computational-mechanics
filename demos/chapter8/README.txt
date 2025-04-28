- Code zur Evaluation der 4 Modelle: naives FNN, naives RNN, visco-PANN und GSM.

- Benötigte packages: Tensorflow (V. 2.17.1 oder neuer), numpy und matplotlib 

- In der file 'main.py' kann man eins der 4 Modelle auswählen, trainieren und für Testfälle sich predictions plotten lassen und speichern.
- Die Ergebnisse werden als png und txt files in dem Ordner 'data/[MODELTYPE]' abgespeichert

- Der Code für die 4 Modelle befindet sich jeweils in den files 'FFNN.py', 'RNN.py', 'viscoNN.py' und 'GSM.py'

- Die file 'data.py' beinhaltet ein analytisches Modell zur Erzeugen der Trainings- und Testdatensätze.

- Die file 'CustomNN.py' beinhaltet eine Funktion um FNNs basierend auf den input Parametern schnell zu generieren. Diese Funktion wird jeweils unterschiedlich in den einzelnen Modellen 'FFNN.py', 'RNN.py', 'viscoNN.py' und 'GSM.py' verwendet.

- In den Ordnern 'Weights_FNN', 'Weights_RNN', 'Weights_PANN' und 'Weights_GSM' werden die zuletzt trainierten Modelle gespeichert.



	