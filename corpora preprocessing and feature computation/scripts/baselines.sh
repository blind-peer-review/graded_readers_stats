# indices are provided because the random split was already done in RStudio and we are matching the data
env/bin/python main.py baselines "data/readers/readers.csv" "Inicial,Intermedio,Avanzado" "1,2,6,7,9,21,23,26,29,30,32,33,34,36,41,42,46"
env/bin/python main.py baselines "data/literature/literature.csv" "Infantil,Juvenil,Adulta" "0,3,8,14,17,21,23,25,26,31,34,35,42,45,46"
