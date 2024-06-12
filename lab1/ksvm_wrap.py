import numpy as np
import torch
from sklearn.svm import SVC

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        """
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre
        """
        self.model = SVC(C=param_svm_c, gamma=param_svm_gamma).fit(X, Y_)
        
    def predict(self, X):
        # Predviđa i vraća indekse razreda podataka X
        return self.model.predict(X)

    def get_scores(self, X):
        # Vraća klasifikacijske mjere
        # (engl. classification scores) podataka X;
        # ovo će vam trebati za računanje prosječne preciznosti.
        return self.model.decision_function(X)
    
    def support(self):
        # Indeksi podataka koji su odabrani za potporne vektore
        return self.model.support_
    
def ksvmwrap_decfun(model, X):
    def classify(X):
        return model.predict(X)
    return classify