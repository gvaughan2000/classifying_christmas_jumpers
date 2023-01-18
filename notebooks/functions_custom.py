import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K

def load_data(train_dir, test_dir, batch_size, img_height, img_width):
    img_height = 180
    img_width = 180

    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, seed=123, image_size=(img_height, img_width), batch_size=batch_size
    )

    test = tf.keras.utils.image_dataset_from_directory(
        test_dir, seed=123, image_size=(img_height, img_width), batch_size=batch_size
    )

    class_names = train.class_names
    num_classes = len(class_names)

    return train, test, num_classes


def train_model(model, train, test, folder_path, epochs):
    #Compile
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Save model summary
    file_path = folder_path/'model_summary.txt'

    with open(file_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    K.set_value(model.optimizer.learning_rate, 0.0005)

    # Run model
    history = model.fit(train, validation_data=test, epochs=epochs)

    return history

def plot_output(history, epochs, batch_size, folder_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Save accuracy and loss to txt aswell as batch number and number of epochs
    with open(folder_path/'details.txt', "w") as f:
        f.write('batch_size:'+  str(batch_size) + "\n")
        f.write('epochs:' + str(epochs) + "\n")
        f.write('test_loss:' + str(val_loss) + "\n")
        f.write('train_accuracy:'+ str(acc) + "\n")
        f.write('test_accuracy:' + str(val_acc) + "\n")
        f.write('train_loss:' + str(loss) + "\n")
        f.write('test_loss:' + str(val_loss) + "\n")

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Test Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Test Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Test Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Test Loss")

    graph_path = folder_path/'graph.png'
    plt.savefig(graph_path, bbox_inches='tight')








