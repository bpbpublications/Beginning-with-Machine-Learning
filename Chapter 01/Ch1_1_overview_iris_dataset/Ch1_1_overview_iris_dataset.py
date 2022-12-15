from sklearn.datasets import load_iris

import pandas as pd
iris = load_iris()
ir= pd.DataFrame(iris.data)
ir.columns = iris.feature_names
ir['Class']=iris.target
print(ir.head())