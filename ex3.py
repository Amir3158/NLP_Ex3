import torch
import torch.nn as nn
import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import data_loader
import pickle

# ------------------------------------------- Constants ----------------------------------------
POLAR = "polar"
RARE = "rare"

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training_oh on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training_oh or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training_oh the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    output = np.zeros(embedding_dim, dtype='float32')
    num_of_words = len(sent.text)
    for w in sent.text:
        output += word_to_vec.get(w, np.zeros(embedding_dim, dtype='float32'))
    return output / num_of_words


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    output = np.zeros(size, dtype='float32')
    output[ind] = 1.0
    return output


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    output = np.zeros(len(word_to_ind), dtype='float32')
    num_of_words = len(sent.text)
    for w in sent.text:
        output += get_one_hot(len(word_to_ind), word_to_ind[w])
    return output / num_of_words


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    return {w: i for i, w in enumerate(words_list)}


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    length = len(sent)

    if (length > seq_len):
        sent = sent[:seq_len]
        res = [word_to_vec[word] for word in sent]

    elif (length < seq_len):
        res = [word_to_vec[word] for word in sent]
        for i in range(seq_len - length):
            res.append(np.zeros(embedding_dim))         ##TODO: check dim for zeros

    else:
        res = [word_to_vec[word] for word in sent]

    return np.array(res)



class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training_oh and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training_oh and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training_oh data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()
        self.sentences["rare"] = get_rare_polar(self, "rare")
        self.sentences["polar"] = get_rare_polar(self, "polar")
        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators

        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}




    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.module = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * hidden_dim, 1)


    def forward(self, text):
        forward_lstm = self.module(text)
        regular = forward_lstm[:, -1, : self.hidden_dim]
        reverse = forward_lstm[:, 0, : self.hidden_dim]

        return self.linear(torch.cat([regular, reverse], dim=1))


    def predict(self, text):
        return torch.sigmoid(self.linear(self.moduletext))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.module = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.module.forward(x)

    def predict(self, x):
        return torch.sigmoid(self.module(x))


# ------------------------- training_oh functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    assert preds.size() == y.size()
    correct = 0
    for i, p in enumerate(preds):
        if torch.round(torch.sigmoid(p)) == y[i]:
            correct += 1
    return correct / len(y)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training_oh of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training_oh
    :param data_iterator: an iterator, iterating over the training_oh data for the model.
    :param optimizer: the optimizer object for the training_oh process.
    :param criterion: the criterion object for the training_oh process.
    """
    model.train()
    for (X, y) in data_iterator:
        optimizer.zero_grad()
        predict = model.forward(X).reshape(y.shape)
        loss = criterion(predict, y)
        loss.backward()
        optimizer.step()


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    loss = 0
    accuracy = 0
    num_of_batches = len(data_iterator)
    model.eval()
    for (X, y) in data_iterator:
        pred = model.forward(X)
        reshapedY = torch.reshape(y, pred.shape)
        loss += criterion(pred, reshapedY)
        accuracy += binary_accuracy(pred, reshapedY)
    loss /= num_of_batches
    accuracy /= num_of_batches

    return loss, accuracy


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:nd.array: the models predictions as a numpy ndarray or torch tensor (or list if you prefer). the
    prediction should be in the same order of the examples returned by data_iter
    """
    predictions = []

    for (X, y) in data_iter:
        predict = model.predict(X).reshape(y.shape)
        predictions.append(predict)

    predictions = np.array(predictions)
    return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0., t='oh'):
    """
    Runs the full training_oh procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training_oh set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    train_loss, train_accuracy = np.zeros(n_epochs, dtype='float32'), np.zeros(n_epochs, dtype='float32')
    val_loss, val_accuracy = np.zeros(n_epochs, dtype='float32'), np.zeros(n_epochs, dtype='float32')
    for r in range(n_epochs):
        if os.path.exists(f"training_{t}/epoch_{str(r)}"):
            model, optimizer, _ = load(model, f"training_{t}/epoch_{str(r)}", optimizer)
        else:
            train_epoch(model, data_manager.get_torch_iterator(), optimizer, loss_function)
            save_model(model, f"training_{t}/epoch_{str(r)}", r, optimizer)
        train_loss[r], train_accuracy[r] = evaluate(model, data_manager.get_torch_iterator(), loss_function)
        print(f"Train Error for epoch-{r}: \n Accuracy: {(100 * train_accuracy[r]):>0.1f}%, Avg loss:"
              f" {train_loss[r]:>8f} \n")
        val_loss[r], val_accuracy[r] = evaluate(model, data_manager.get_torch_iterator(VAL), loss_function)
        print(f"Val Error for epoch-{r}: \n Accuracy: {(100 * val_accuracy[r]):>0.1f}%, Avg loss: {val_loss[r]:>8f} \n")

    return train_loss, train_accuracy, val_loss, val_accuracy



def train_log_linear_with_one_hot():
    """
    Here comes your code for training_oh and evaluation of the log linear model with one hot representation.
    """
    dm = DataManager(batch_size=64)
    m = LogLinear(dm.get_input_shape()[0])
    train_loss, train_accuracy, val_loss, val_accuracy = train_model(m, dm, 20, 0.01, 0.001, "oh")
    n_epochs = len(train_loss)
    epochs = np.arange(1,n_epochs + 1)

    plt.plot(epochs, train_loss, color='r', label='Train loss')
    plt.plot(epochs, val_loss, color='g', label='Validation loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation loss over number epochs")
    plt.legend()
    plt.show()

    plt.plot(epochs, train_accuracy, color='r', label='Train accuracy')
    plt.plot(epochs, val_accuracy, color='g', label='Validation accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train and Validation accuracy over number epochs")
    plt.legend()
    plt.show()

    test_loss, test_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=TEST), nn.BCEWithLogitsLoss())
    rare_loss, rare_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=RARE), nn.BCEWithLogitsLoss())
    polar_loss, polar_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=POLAR), nn.BCEWithLogitsLoss())
    print(f"the test loss is- {test_loss}, the test accuracy is- {test_accuracy}")
    print(f"the rare loss is- {rare_loss}, the rare accuracy is- {rare_accuracy}")
    print(f"the polar loss is- {polar_loss}, the polar accuracy is- {polar_accuracy}")




def get_rare_polar(dm, subset):
    test_data = dm.sentiment_dataset.get_test_set()

    if subset == RARE:
        ind = data_loader.get_rare_words_examples(test_data, dm.sentiment_dataset)

    elif subset == POLAR:
        ind = data_loader.get_negated_polarity_examples(test_data)

    res = []
    for i in ind:
        res.append(test_data[i])

    return res


def train_log_linear_with_w2v():
    """
    Here comes your code for training_oh and evaluation of the log linear model with word embeddings
    representation.
    """
    dm = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    m = LogLinear(dm.get_input_shape()[0])
    train_loss, train_accuracy, val_loss, val_accuracy = train_model(m, dm, 20, 0.01, 0.001, "w2v")
    n_epochs = len(train_loss)
    epochs = np.arange(1, n_epochs + 1)

    plt.plot(epochs, train_loss, color='r', label='Train loss')
    plt.plot(epochs, val_loss, color='g', label='Validation loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation loss over number epochs")
    plt.legend()
    plt.show()

    plt.plot(epochs, train_accuracy, color='r', label='Train accuracy')
    plt.plot(epochs, val_accuracy, color='g', label='Validation accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train and Validation accuracy over number epochs")
    plt.legend()
    plt.show()

    test_loss, test_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=TEST), nn.BCEWithLogitsLoss())
    rare_loss, rare_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=RARE), nn.BCEWithLogitsLoss())
    polar_loss, polar_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=POLAR), nn.BCEWithLogitsLoss())
    print(f"the test loss is- {test_loss}, the test accuracy is- {test_accuracy}")
    print(f"the rare loss is- {rare_loss}, the rare accuracy is- {rare_accuracy}")
    print(f"the polar loss is- {polar_loss}, the polar accuracy is- {polar_accuracy}")


def train_lstm_with_w2v():
    dm = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=300)
    m = LSTM(W2V_EMBEDDING_DIM, 100, 2, 0.5)
    train_loss, train_accuracy, val_loss, val_accuracy = train_model(m, dm, 4, 0.01, 0.0001, "lstm")
    n_epochs = len(train_loss)
    epochs = np.arange(1, n_epochs + 1)

    plt.plot(epochs, train_loss, color='r', label='Train loss')
    plt.plot(epochs, val_loss, color='g', label='Validation loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation loss over number epochs")
    plt.legend()
    plt.show()

    plt.plot(epochs, train_accuracy, color='r', label='Train accuracy')
    plt.plot(epochs, val_accuracy, color='g', label='Validation accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train and Validation accuracy over number epochs")
    plt.legend()
    plt.show()

    test_loss, test_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=TEST), nn.BCEWithLogitsLoss())
    rare_loss, rare_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=RARE), nn.BCEWithLogitsLoss())
    polar_loss, polar_accuracy = evaluate(m, dm.get_torch_iterator(data_subset=POLAR), nn.BCEWithLogitsLoss())
    print(f"the test loss is- {test_loss}, the test accuracy is- {test_accuracy}")
    print(f"the rare loss is- {rare_loss}, the rare accuracy is- {rare_accuracy}")
    print(f"the polar loss is- {polar_loss}, the polar accuracy is- {polar_accuracy}")


if __name__ == '__main__':
    # train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()
    # dm = DataManager()
    # m = LogLinear(dm.get_input_shape()[0])
    # optemizer = torch.optim.Adam(m.parameters(), lr=0.05)
    # criterion = nn.BCEWithLogitsLoss()
    # train_epoch(m, dm.get_torch_iterator(), optemizer, criterion)

    train_lstm_with_w2v()

