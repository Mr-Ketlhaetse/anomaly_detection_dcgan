try:
    import pandas as pd
    import numpy as np
    import os
    from ctgan import CTGAN
    from preprocessing import detect_features, remove_null_rows, combine_datasets
    from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation
    import torch
    from cnn import CNNTransferLearning
    from dcgan import DCGAN
    from torchvision import transforms
    from custom_helpers import ImageDatasetLoader
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.dataset import Subset
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.preprocessing import Binarizer
    import matplotlib.pyplot as plt 
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.model_selection import learning_curve
    import tempfile
    from config import Parameters as prm
except ImportError as e:
    print(f"An error occurred while importing modules: {e}")


def false_alarm_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if fp + tn > 0 else 0
    return far


def false_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if fn + tp > 0 else 0
    return fnr


def true_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp) if tn + fp > 0 else 0
    return tnr


def main():
    
    print("Initializing")

    # Load original data
    # URL_ONLINE = 'https://www.kaggle.com/datasets/ekkykharismadhany/csecicids2018-cleaned/download?datasetVersionNumber=1'
    LOCAL_FOLDER = 'cleaned_ids2018_sampled.csv'
    
    # Check if the file exists
    if not os.path.exists(LOCAL_FOLDER):
        raise FileNotFoundError(f"The file {LOCAL_FOLDER} does not exist.")
    
    # Open and read the file safely
    try:
        with open(LOCAL_FOLDER, 'r') as file:
            print("Loading dataset")
            original_data = pd.read_csv(LOCAL_FOLDER).iloc[:prm.ORIGINAL_SAMPLES]
            print("data loading done")
            # Process the data as needed
    except OSError as e:
        print(f"An error occurred while handling the file: {e}")
     

    # Remove rows with null values
    print("removing null values")
    original_data = remove_null_rows(original_data)

    # Detect continuous and discrete features
    print("detecting features")
    continuous_features, discrete_features = detect_features(original_data)

    # Train CTGAN model
    print("Training CTGAN model")
    ctgan = CTGAN(epochs=prm.CTGAN_EPOCHS)
    ctgan.fit(original_data, discrete_features)

    # Generate synthetic data
    print("Generating synthetic data")
    synthetic_data = ctgan.sample(prm.CTGAN_SAMPLES)
    # print(synthetic_data.head(10))
    # 
    # Iterate based on a range prm.syn_ratio values with an increment of 0.2
    # Specify the increment value
    increment = 0.2  # Change this value to your desired increment

    # Iterate based on a range with the specified increment
    for i in [0.2, 0.4, 0.6, 0.8, 1.0]:
       
        prm.SYN_RATIO = i
        # Combine real and synthetic data
        print("Combining real and synthetic data")
        sample_data, sampled_filepath = combine_datasets(original_data, synthetic_data, prm.SYN_RATIO, prm.SAMPLED_DATA_FILE)

        # Select features based on variation
        print("Selecting features based on variation")
        num_row = 26
        num_col = 3
        num = num_row * num_col

        data = pd.read_csv(sampled_filepath)
        os.remove(sampled_filepath)  # Remove the sampled data file after reading
        id = select_features_by_variation(data, variation_measure='var', num=num)
        data = data.iloc[:, id]
        norm_data = min_max_transform(data.values)
        norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

        # Separate real data and target column
        print("Separating real data and target column")
        real_target = norm_data['Label']
        real_data = norm_data.drop(columns=['Label'])

        # Convert table data to images
        print("Converting table data to images")    
        fea_dist_method = 'Euclidean'
        image_dist_method = 'Euclidean'
        error = 'abs'
        result_dir = 'Results/Table_To_Image_Conversion/Test_1'
        os.makedirs(name=result_dir, exist_ok=True)  # Create the result directory if it doesn't exist
    
        save_image_size = 64  # Define the variable "save_image_size"
        max_step = 30000  # Define the variable "max_step"
        val_step = 300  # Define the variable "val_step"

        print("generate image samples")
        generated_images = table_to_image(real_data, [num_row, num_col], fea_dist_method, image_dist_method,
                                      save_image_size,
                                      max_step, val_step, result_dir, error)
        # os.close(result_dir)

    
        print("Load images from file")
        # Load image dataset
        print("Load images from file")
        folder_path = 'Results/Table_To_Image_Conversion/Test_1/data'
        transform = transforms.Compose([
            transforms.Resize((64)),
            transforms.ToTensor(),
        ])
        image_dataset = ImageDatasetLoader(folder_path, image_type='png', transform=transform)
        # os.close(folder_path)
        for i in [10, 50, 100]: #Iterate through the DCGAN epochs
            prm.dcgan_epochs = i
            print("Train the DCGAN")
            # Train DCGAN model
            if prm.dcgan_train:
                dcgan_model = DCGAN(image_dataset, prm.latent_dim, prm.img_channels, prm.img_size)
                # Train DCGAN model
                dcgan_model.train()

            # Load pretrained DCGAN discriminator
            print("Transfer learning using pretrained DCGAN discriminator")
            pretrained_dcgan_discriminator = DCGAN.Discriminator(prm.img_channels, 64)
            pretrained_dcgan_discriminator.load_state_dict(torch.load('dcgan_discriminator_weights.pth'))
            pretrained_dcgan_discriminator.eval()
            os.remove('dcgan_discriminator_weights.pth')	# Remove the weights file after loading

            # Initialize CNN model for transfer learning
            num_classes = len(real_target.unique())
            cnn_model = CNNTransferLearning(pretrained_dcgan_discriminator, num_classes, learning_rate=0.01)

            # Define hyperparameters for grid search
            param_grid = {
                'learning_rate': [0.001, 0.01, 0.1]
            }

            # Define the scoring metrics for the model
            far_scorer = make_scorer(false_alarm_rate)  # custom function to calculate the false alarm rate
            fnr_scorer = make_scorer(false_negative_rate)  # custom function to calculate the false negative rate (miss rate
            tnr_scorer = make_scorer(true_negative_rate)  # custom function to calculate the true negative rate
            
            if prm.cross_validate:
                scoring = {
                    'precision': make_scorer(precision_score, average='macro', zero_division=1),
                    'recall': make_scorer(recall_score, average='macro'),
                    'f1': make_scorer(f1_score, average='macro'),
                    'accuracy': make_scorer(accuracy_score),
                    # 'roc_uac_ovr': 'roc_auc_ovr',
                    # 'far': far_scorer,
                    # 'fnr': fnr_scorer,
                    # 'tnr': tnr_scorer
                }

                # Create GridSearchCV object
                print("GridSearchCv instantiation and fitting")
                grid_search = GridSearchCV(cnn_model, param_grid, cv=prm.CV, scoring=scoring, refit='f1', error_score='raise')

                # Fit GridSearchCV object to the data
                grid_search.fit(image_dataset, torch.tensor(real_target).long())

                # Print best parameters and best score
                print("Best Parameters:", grid_search.best_params_)
                print("Best Score:", grid_search.best_score_)

                # best_model = grid_search.best_estimator_
                # y_pred = best_model.predict(image_dataset)  # Replace `X_test` with your actual test data
                # y_true = real_target  # Replace `y_test` with your actual test labels

                # if y_pred.dtype.kind in 'fc':  # 'f' for float, 'c' for complex
                #     # Binarize y_pred (example threshold: 0.5)
                #     binarizer = Binarizer(threshold=0.5)
                #     y_pred = binarizer.fit_transform(y_pred.reshape(-1, 1)).ravel()

                # cm = confusion_matrix(y_true, y_pred)   # Compute confusion matrix

                # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                # disp.plot(cmap=plt.cm.Blues)
                # plt.title('Confusion Matrix')
                # plt.savefig(f"plots/{prm.SYN_RATIO}/{prm.dcgan_epochs}/confusion_matrix.png")
                # plt.show()

                # Create a SummaryWriter object
                writer = SummaryWriter()

                # Preview scorer names
                print("GridSearchCV results keys:", grid_search.cv_results_.keys())
                # Define the directory path
                grid_directory = f"plots/{prm.SYN_RATIO}/{prm.dcgan_epochs}"

                # Create the directory if doesn't exist
                os.makedirs(grid_directory, exist_ok=True)
                # Open a file to write the GridSearchCV results
                with open(f"{grid_directory}/grid_search_results.csv", "w") as file:
                    # Preview scorer names
                    print("GridSearchCV results keys:", grid_search.cv_results_.keys())
                    for key, value in grid_search.cv_results_.items():
                        # Write the results to the file
                        file.write(f"{key}, {value}\n")
                        # Log the results to TensorBoard
                        writer.add_text(key, str(value))


                # Print scores for all metrics
                for scorer_name in scoring.keys():
                    print(f"{scorer_name.capitalize()} Score: {grid_search.cv_results_['mean_test_' + scorer_name]}")
                    plt.plot(grid_search.cv_results_['mean_test_' + scorer_name])
                    plt.xlabel('Parameter Combination')
                    plt.ylabel(f"{scorer_name.capitalize()} Score")
                    plt.title(f"{scorer_name.capitalize()} Score vs Parameter Combination")
                    # Set y-axis limits here
                    plt.ylim([0, 1])  # Adjust as needed

                    directory = f"plots/{prm.SYN_RATIO}/{prm.dcgan_epochs}"
                    os.makedirs(directory, exist_ok=True)
                    plt.savefig(f"{directory}/{scorer_name}.png")
                    plt.clf()

                    # Generate learning curve
                    print("Generating learning curve")
                    train_sizes, train_scores, test_scores = learning_curve(
                        grid_search.best_estimator_, image_dataset, torch.tensor(real_target).long(), cv=prm.CV, scoring=scorer_name,
                        n_jobs=1)

                    train_scores_mean = np.mean(train_scores, axis=1)
                    train_scores_std = np.std(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)
                    test_scores_std = np.std(test_scores, axis=1)

                    plt.figure()
                    plt.title(f"Learning Curve ({scorer_name.capitalize()} Score)")
                    plt.xlabel("Training examples")
                    plt.ylabel("Score")
                    plt.grid()

                    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
                    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
                    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
                    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

                    plt.legend(loc="best")

                    plt.savefig(f"plots/{prm.SYN_RATIO}/{prm.dcgan_epochs}/learning_curve_{scorer_name}.png")
                    plt.clf()

                    # implement tensorboard
                    print("Implementing tensorboard")
                    try:
                        mean_score = np.mean(grid_search.cv_results_['mean_test_' + scorer_name])
                        # include other charts for the other metrics in the tensorboard
                        writer.add_scalar(f"{scorer_name.capitalize()} Score", mean_score)
                        writer.add_figure(f"{scorer_name.capitalize()} Score vs Parameter Combination", plt.figure())
                        writer.add_histogram(f"{scorer_name.capitalize()} Score", grid_search.cv_results_['mean_test_' + scorer_name])
                        writer.add_hparams(param_grid, {scorer_name: mean_score})
                        writer.add_graph(grid_search.best_estimator_, image_dataset)
                    except Exception as e:
                        print(f"An error occurred while processing {scorer_name}: {e}")

                # writer.close()

            else:
                cnn_model, history = cnn_model.fit(image_dataset, torch.tensor(real_target).long())

                try:
                    # Extract performance results
                    print("Extracting performance results")
                    train_loss = history['loss']
                    train_accuracy = history['accuracy']
                    val_loss = history['val_loss']
                    val_accuracy = history['val_accuracy']

                    # Print or visualize the results
                    print("Printing or visualizing the results")
                    print("Training Loss:", train_loss)
                    print("Training Accuracy:", train_accuracy)
                    print("Validation Loss:", val_loss)
                    print("Validation Accuracy:", val_accuracy)
                except KeyError as e:
                    print(f"An error occurred while extracting performance results: {e}")


if __name__ == '__main__':
    main()
