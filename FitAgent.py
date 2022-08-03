from MARK import MARK
from Parameters import lr, dataset, epochs


m = MARK()
m.fit(textdir=dataset, epochs=epochs, lr=lr)
