from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_test, axis=-3): 
    # X: (S, N, T, 1)
    for j in range(X_train.shape[axis]):
          scaler = StandardScaler()
          scaler.fit(X_train[:, j, :, 0])
          X_train[:, j, :, 0] = scaler.transform(X_train[:, j, :, 0])
          X_test[:, j, :, 0] = scaler.transform(X_test[:, j, :, 0])

    return X_train, X_test