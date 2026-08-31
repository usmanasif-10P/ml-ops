from kfp import dsl, compiler
from kfp.dsl import Dataset, Input, Output

@dsl.component(
    base_image='python:3.12-slim',
    packages_to_install=['scikit-learn>=1.9.0', 'pandas']
)
def load_data(
    X_out: Output[Dataset],
    y_out: Output[Dataset]
):
    import pandas as pd
    from sklearn.datasets import load_iris
    
    X, y = load_iris(return_X_y=True)
    
    # Save the arrays as CSV files to the designated KFP artifact paths
    pd.DataFrame(X).to_csv(X_out.path, index=False)
    pd.DataFrame(y).to_csv(y_out.path, index=False)

@dsl.component(
    base_image='python:3.12-slim',
    packages_to_install=['scikit-learn>=1.9.0', 'pandas']
)
def train_model(
    X_in: Input[Dataset], 
    y_in: Input[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # Load the datasets back into memory
    X = pd.read_csv(X_in.path).values
    y = pd.read_csv(y_in.path).values.ravel() # Flatten to 1D array
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {score}")

@dsl.pipeline
def iris_pipeline():
    # KFP components with Output[Dataset] parameters return placeholders automatically
    data = load_data()
    
    # Pass the output paths directly to the training component
    train_model(X_in=data.outputs['X_out'], y_in=data.outputs['y_out'])

if __name__ == '__main__':
    compiler.Compiler().compile(iris_pipeline, 'pipelines/iris.yaml')
