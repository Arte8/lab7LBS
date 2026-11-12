# lab7LBS
lab7 capstoneprep humber college aigc5005

### Linear Relationship Assessment

The relationship between hour of day and bike rentals is nonlinear. Rental demand tends to peak during morning and evening commuting hours and decrease during quieter periods.

A straight-line model would not accurately capture these fluctuations. However, linear regression remains useful as a baseline for comparing the performance of more advanced nonlinear models.

### One-Epoch Training and Evaluation Function

A reusable `run_epoch()` function was implemented to support both training and evaluation of the PyTorch neural network.

When an optimizer is provided, the function enables training mode, calculates gradients using backpropagation, and updates the model weights.

When no optimizer is provided, the function switches to evaluation mode and disables gradient calculations, allowing performance to be measured without modifying model parameters.

The function returns the mean loss and Mean Absolute Error (MAE). MAE measures the average absolute difference between predicted and actual bike rentals, expressed in the original units of rentals.

Both metrics are aggregated across observations rather than averaged equally across batches.

### Multi-Epoch Neural Network Training

A `train_model()` function was implemented to train the neural network over multiple epochs using the Adam optimizer and Mean Squared Error (MSE) loss.

For every epoch, the function records the training loss, validation loss, and validation Mean Absolute Error (MAE).

Validation performance is monitored throughout training. Whenever the validation loss improves, a deep copy of the model's weights is saved.

After all epochs are completed, the weights corresponding to the lowest validation loss are restored. This ensures that the final model uses the best-performing epoch on the validation dataset rather than automatically using the last epoch.

Training and validation loss curves are plotted to assess model convergence and potential overfitting. Validation MAE is also plotted to measure prediction error in units of bike rentals.

The test dataset remains untouched during model training and selection.