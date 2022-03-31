import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, LeakyReLU, BatchNormalization, Dropout, Reshape
from tensorflow.keras import losses, regularizers


class Discriminator(tf.keras.Model):
    """
    The Discriminator part of the music generating GAN.

    Technically consists of two discriminators, one that discriminates between real and fake/generated samples
    and one that classifies the genre of the sample.
    """
    def __init__(self, nb_of_genres):
        """
        Initializes the Discriminator's layers using the given argument

        Args:
            nb_of_genres: The number of genres present in the data / that should be classified
        """
        super(Discriminator, self).__init__()
        self.nb_genres = nb_of_genres

        self.layer_list_dr = [
            Bidirectional(LSTM(64, return_sequences=True, activation='tanh')),
            Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
            Dense(256),
            LeakyReLU(alpha=0.01),
            Dense(1, activation='sigmoid', name='gan_output')
        ]

        self.layer_list_dc = [
            Bidirectional(LSTM(128, return_sequences=True, activation='tanh', dropout=0.1, recurrent_dropout=0.1)),
            BatchNormalization(),
            Bidirectional(LSTM(128, return_sequences=False, activation='tanh', dropout=0.1, recurrent_dropout=0.1)),
            BatchNormalization(),
            Dropout(0.1),
            Dense(512, activation='sigmoid'),
            Dense(self.nb_genres, activation='softmax', name='style_output',
                  kernel_regularizer=regularizers.l2(0.01),
                  activity_regularizer=regularizers.l1(0.01))
        ]

    def call(self, x):
        """
        Performs the forward computation for the Discriminator network, going through both parts of the discriminator
        and returning their outputs.

        Args:
            x: The input sample(s). Expected shape is (batch_size, sequence_len, nb_possible_notes).

        Returns:
            The output of the Dr Discriminator, a tensor of shape (batch_size, 1) with values from 0 to 1 indicating
            whether the sample is fake(generated) or real.
            The output of the Dc Discriminator, a tensor of shape (batch_size, nb_genres) with values from 0 to 1
            indicating how likely the sample belongs to the respective genre.

        """
        gan_output = x
        genre_output = x

        for layer in self.layer_list_dr:
            gan_output = layer(gan_output)

        for layer in self.layer_list_dc:
            genre_output = layer(genre_output)

        return gan_output, genre_output


class Generator(tf.keras.Model):
    """
    The Generator part of the music generating GAN.

    Generates new music samples.
    """
    def __init__(self, sequence_length=32, nb_of_notes=128):
        """
        Initializes the Generator and its layers with the given arguments.

        Args:
            sequence_length: The length that the generated music sequence should have
            nb_of_notes: The number of possible notes that the generated samples(s) should contain.
        """
        super(Generator, self).__init__()
        self.nb_notes = nb_of_notes
        self.seq_len = sequence_length

        self.layer_list = [
            Dense(self.seq_len*self.seq_len/2),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.9),
            Dense(self.seq_len*self.seq_len),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.9),
            Reshape((self.seq_len, self.seq_len)),
            Dropout(0.1),

            LSTM(128, return_sequences=True, activation='tanh'),
            LSTM(128, return_sequences=True, activation='tanh'),
            LSTM(self.nb_notes, return_sequences=True, activation='sigmoid')
        ]

    def call(self, z):
        """
        Performs a forward pass.

        Args:
            z: The random input vector, a tensor of shape (batch_size, lenght), which is used to generate the output and
            whose batch_size indicates the batch size of the generated output (the number of samples).

        Returns:
            Generated sample(s) as a tensor of shape (batch_size, sequence_length, nb_of_notes).

        """
        for layer in self.layer_list:
            z = layer(z)
        return z

    def generate_noise(self, batch_size, length):
        """
        Generates a tensor of normally distributed noise with shape (batch_size, length).

        Args:
            batch_size: The batch_size that the generated noise should have.
            length: The length that the generated noise should have.

        Returns:
            A tensor of generated noise with shape (batch_size, length).

        """
        noise = tf.random.normal(shape=[batch_size, length], mean=0, stddev=0.5)
        return noise


class GAN(tf.keras.Model):
    def __init__(self, nb_of_genres, sequence_len, nb_of_notes=128):
        """
        Initializes the GAN.

        Args:
            nb_of_genres: The number of genres in the samples that will be presented to the Discriminator.
            sequence_len: The length that the generated music samples should have.
            nb_of_notes: The number of possible notes that the generated sample should use.
        """
        super(GAN, self).__init__()
        self.nb_genres = nb_of_genres
        self.nb_notes = nb_of_notes
        self.seq_len = sequence_len

        self.d = Discriminator(nb_of_genres)
        self.g = Generator(sequence_len, nb_of_notes)

    def call(self, z):
        """
        Performs a forward pass generating new samples given the random input and returning the discriminator output for
        the generated samples.

        Args:
            z: The random input vector with shape (batch_size, length).

        Returns:
            The output of the Discriminator which consists of the Dr and Dc output.

        """
        img_gan = self.g(z)
        prediction_gan = self.d(img_gan)
        return prediction_gan

    def style_ambiguity_loss(self, y_true, y_pred):
        """
        Computes the style ambiguity loss for y_pred. y_true is not used for the computation and only exists as a
        parameter so that this can be used like a tensorflow loss.

        Args:
            y_true: The true/actual values.
            y_pred: The predicted genre values as given as output by the dc part of the Discriminator as a tensor of
                    shape (batch_size, nb_of_genres)

        Returns:
            The style ambiguity loss value.

        """
        even_dist = tf.ones_like(y_pred) * 1.0 / float(self.nb_genres)
        return losses.categorical_crossentropy(even_dist, y_pred, from_logits=False)
   

# this is here for use in reloaded models, as the model's style_ambiguity_loss function is lost when the model is saved
def style_ambiguity_loss(y_true, y_pred):
    """
    Computes the style ambiguity loss for y_pred. y_true is not used for the computation and only exists as a
    parameter so that this can be used like a tensorflow loss.

    Args:
        y_true: The true/actual values.
        y_pred: The predicted genre values as given as output by the dc part of the Discriminator as a tensor of
                shape (batch_size, nb_of_genres)

    Returns:
        The style ambiguity loss value.

    """
    even_dist = tf.ones_like(y_pred) * 1.0 / float(NB_GENRES)
    return losses.categorical_crossentropy(even_dist, y_pred, from_logits=False)
