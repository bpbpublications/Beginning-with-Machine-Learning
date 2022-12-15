from sklearn.datasets import load_iris

import pandas as pd
iris = load_iris()
ir= pd.DataFrame(iris.data)
ir.column = iris.feature_names
ir['Class']=iris.target
print(ir.head())