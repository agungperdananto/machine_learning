# machine_learning
machine learning data

    # Visualizing test set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(
            np.arange(
                    start = X_set[:, 0].min() - 1, 
                    stop = X_set[:, 0].max() + 1, 
                    step = 0.01),
            np.arange(
                    start = X_set[:, 1].min() - 1, 
                    stop = X_set[:, 1].max() + 1, 
                    step = 0.01))
    plt.contourf(
            X1, 
            X2, 
            classifier.predict(
                    np.array([X1.ravel(), 
                            X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.5, 
            cmap = ListedColormap(('blue', 'red'))
            )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
            
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('blue', 'red'))(i), label = j)
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()