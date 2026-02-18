import numpy as np
import model
import pandas as pd

if __name__ == "__main__":
    x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
    y = np.array([[3, 6, 9, 12, 15, 18, 21, 24, 27]]).T

    model.learn(x, y)

    df = pd.DataFrame({
        'true': np.squeeze(y), 'pred': np.squeeze(model.calculatePredict(x)),
    })

    print(df)
