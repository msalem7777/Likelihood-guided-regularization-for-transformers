class Dataset_gen(Dataset):
    def __init__(self, root_path, data_path='gen_dat.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=False, smooth = 0, fpca_comp = 0, scale_statistic=None):
        # info
        self.in_len = size[0]  # Length of the input sequence
        self.out_len = size[1]  # Length of the output sequence

        self.flag = flag
        
        self.scale = scale
        self.smooth = smooth
        self.fpca_comp = fpca_comp
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        # Read the dataset
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Split the data for train, val, test
        if self.data_split[0] > 1:
            train_num = self.data_split[0]
            val_num = self.data_split[1]
            test_num = self.data_split[2]
        else:
            train_num = int(len(df_raw) * self.data_split[0])
            test_num = int(len(df_raw) * self.data_split[2])
            val_num = len(df_raw) - train_num - test_num

        if self.flag == 'train':
            start_idx, end_idx = 0, train_num
        elif self.flag == 'val':
            start_idx, end_idx = train_num, train_num + val_num
        else:
            start_idx, end_idx = train_num + val_num, len(df_raw)
        
        # Adjust the split for multiple individuals
        individual_data = []
        individual_labels = []

        if self.fpca_comp > 0:

            # Extract the classification status (labels) from the first column
            labels_train = df_raw.iloc[0:train_num, 0].values
            labels_val = df_raw.iloc[train_num:(train_num + val_num), 0].values
            labels_test = df_raw.iloc[(train_num + val_num):, 0].values
            
            # Initialize a list to store the reconstructed data for each covariate
            reconstructed_train_data_list = []
            reconstructed_val_data_list = []
            reconstructed_test_data_list = []
            
            for i in range(1, df_raw.shape[1]): # Loop through the dataframe columns
                
                # Extract the current time series data
                X_train = df_raw.iloc[0:train_num, 1:].values.reshape((train_num // self.in_len), self.in_len, (df_raw.shape[1]-1))
                X_val = df_raw.iloc[train_num:(train_num+val_num), 1:].values.reshape((val_num // self.in_len), self.in_len, (df_raw.shape[1]-1))
                X_test = df_raw.iloc[(train_num+val_num):len(df_raw), 1:].values.reshape((test_num // self.in_len), self.in_len, (df_raw.shape[1]-1))
                
                time_series_data_train = X_train[:, :, (i-1)]
                time_series_data_val = X_val[:, :, (i-1)]
                time_series_data_test = X_test[:, :, (i-1)]
                
                # Create FDataGrid for the current time series
                fdata_train = skfda.FDataGrid(time_series_data_train, grid_points=np.linspace(0, 1, self.in_len))
                fdata_val = skfda.FDataGrid(time_series_data_val, grid_points=np.linspace(0, 1, self.in_len))
                fdata_test = skfda.FDataGrid(time_series_data_test, grid_points=np.linspace(0, 1, self.in_len))
                
                # Perform FPCA
                self.fpca = FPCA(n_components=self.fpca_comp)
                self.fpca.fit(fdata_train)
            
                # Transform the data (scores)
                X_train_transformed = self.fpca.transform(fdata_train)
                X_val_transformed = self.fpca.transform(fdata_val)
                X_test_transformed = self.fpca.transform(fdata_test)
            
                # Reconstruct the approximate functions using FPCA scores
                reconstructed_train = reconstruct_fpca(self.fpca, X_train_transformed, time_series_data_train.shape[0], self.in_len)
                reconstructed_val = reconstruct_fpca(self.fpca, X_val_transformed, time_series_data_val.shape[0], self.in_len)
                reconstructed_test = reconstruct_fpca(self.fpca, X_test_transformed, time_series_data_test.shape[0], self.in_len)
                    
                # Append the reconstructed covariate to the list
                reconstructed_train_data_list.append(reconstructed_train)
                reconstructed_val_data_list.append(reconstructed_val)
                reconstructed_test_data_list.append(reconstructed_test)
            
            # Stack the reconstructed covariates along the second axis (features/covariates)
            # Resulting shape will be: (num_observations, t_length, num_covariates)
            reconstructed_train_data = np.stack(reconstructed_train_data_list, axis=-1)
            reconstructed_val_data = np.stack(reconstructed_val_data_list, axis=-1)
            reconstructed_test_data = np.stack(reconstructed_test_data_list, axis=-1)
            
            # Reshape each set to the original format
            reconstructed_train_data = reconstructed_train_data.reshape(-1, df_raw.shape[1] - 1)
            reconstructed_val_data = reconstructed_val_data.reshape(-1, df_raw.shape[1] - 1)
            reconstructed_test_data = reconstructed_test_data.reshape(-1, df_raw.shape[1] - 1)

            # Concatenate the labels as the first column
            reconstructed_train_data = np.column_stack((labels_train, reconstructed_train_data))
            reconstructed_val_data = np.column_stack((labels_val, reconstructed_val_data))
            reconstructed_test_data = np.column_stack((labels_test, reconstructed_test_data))
            
            # Concatenate the reconstructed datasets
            df_raw_reconstructed = np.concatenate((reconstructed_train_data, reconstructed_val_data, reconstructed_test_data), axis=0)
            # Convert 'df_raw_reconstructed' to a pandas DataFrame
            df_raw = pd.DataFrame(df_raw_reconstructed, columns=df_raw.columns)
        
        # Process data for the current split (train/val/test)
        for i in range(start_idx // self.in_len, end_idx // self.in_len):
            start = i * self.in_len
            end = start + self.in_len
            if end <= len(df_raw):  # Ensure that the end index does not go out of bounds
                data_segment = df_raw.iloc[start:end, 1:].values  # Exclude the label column
        
                # Apply smoothing if self.smooth > 1.0
                if self.smooth > 1:
                    window_size = int(self.smooth)
                    smoothed_segment = np.zeros_like(data_segment)
                    
                    # Apply smoothing independently for each column
                    for col in range(data_segment.shape[1]):  # Apply per column
                        for kl in range(0, len(data_segment), self.smooth):
                            chunk = data_segment[kl:kl + self.smooth, col]  # Get the chunk for this step
                            mean_value = chunk.max()  # Compute the mean for this chunk
                            smoothed_segment[kl:kl + self.smooth, col] = mean_value  # Apply the mean to the chunk
            
                    data_segment = smoothed_segment  # Replace with smoothed data

                # Scale the data if self.scale is enabled
                if self.scale:
                    self.scaler = StandardScaler()
                    self.scaler.fit(data_segment.reshape(-1, data_segment.shape[-1]))
                    data_segment = self.scaler.transform(data_segment)
                
                individual_data.append(data_segment)
                individual_labels.append(df_raw.iloc[end-1, 0])  # Assume label is in the first column
                
        self.data_x = np.array(individual_data)
        self.labels = np.array(individual_labels)

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        label = self.labels[index]

        return seq_x, label
    
    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)