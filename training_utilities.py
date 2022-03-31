import tensorflow as tf
from tensorflow.keras import utils
import matplotlib.image
import pretty_midi
import music21
import numpy as np
from model import Generator


def matrix_to_midi(one_sample):
    sequence_len = tf.shape(one_sample)[0]

    # MIDI Instance
    new_mid = pretty_midi.PrettyMIDI()
    new_inst = pretty_midi.Instrument(program=42)
    new_mid.instruments.append(new_inst)

    # look for note starts
    time, pitch = np.where(one_sample == 1)

    time = list(time)
    pitch = list(pitch)

    for t, note in zip(time, pitch):
        start = t * 0.125
        p = note
        end = None

        # end rausfinden
        for s in range(t + 1, sequence_len - 1):
            # Is it end
            if one_sample[s, p] == 3:
                end = s * 0.125
                break

            # nur einmal angeschlagen
            elif one_sample[s, p] != 2:
                end = start + 0.125
                break

        if end is None:
            end = (float(sequence_len) - 1) * 0.125

        new_note = pretty_midi.Note(127, p, start, end)
        new_inst.notes.append(new_note)

    new_mid.write("matrix2midi.mid")

    return new_mid


def plot_and_play_notes(matrix):
    # multiply and round to get from values between 0 and 1 to int values from 0 to 3
    matrix = tf.squeeze(matrix)
    matrix = matrix * 3
    matrix = tf.math.round(matrix)

    matrix_to_midi(matrix)
    test = music21.converter.parse("matrix2midi.mid")
    sheet = test.write(fmt="musicxml.png")
    music = pretty_midi.PrettyMIDI(midi_file="matrix2midi.mid")
    waveform = music.synthesize()

    return sheet, waveform


def training(dataset, gan, NB_GENRES, BATCH_SIZE, NB_EPOCHS, K_UNROLLED, MAX_LOSS_RATIO, ones, zeros, train_summary_writer, start_epoch=0, save_path_training="./models", save_path_best="./models/gan"):
    data = dataset.unbatch()

    train_d = True
    train_g = True

    # keep track of min total loss
    min_total_loss = 100000.
    min_total_loss_epoch = 0

    # only used for printing steps
    nb_batches = len(dataset)

    m_d_loss = 0.0
    m_d_accuracy = 0.0
    m_d_accuracy_fake = 0.0
    m_d_accuracy_real = 0.0
    m_d_loss_cat = 0.0
    m_d_cat_accuracy = 0.0

    # iterate each epoch, going thought the dataset once per epoch
    for epoch in range(start_epoch, NB_EPOCHS):
        # iterate through dataset batches
        for repeat, (notes, labels) in enumerate(dataset):

            step = nb_batches * epoch + repeat

            if train_d:

                # training data
                d = data.shuffle(10000)
                d = d.batch(BATCH_SIZE)
                d = d.take(K_UNROLLED)
                for notes_train, labels_train in d:
                    labels_train_categorical = utils.to_categorical(labels_train, num_classes=NB_GENRES)

                    # generated samples
                    noise = Generator.generate_noise(None, BATCH_SIZE, 100)
                    drum_fake = gan.g(noise)

                    # training D
                    _, d_loss_real, d_loss_cat, d_acc_real, cat_accuracy = gan.d.train_on_batch(notes_train,
                                                                                                [ones,
                                                                                                 labels_train_categorical])
                    _, d_loss_fake, _, d_acc_fake, _ = gan.d.train_on_batch(drum_fake,
                                                                            [zeros,
                                                                             labels_train_categorical])

                    m_d_loss += 0.5 * (d_loss_real + d_loss_fake)
                    m_d_loss_cat += d_loss_cat
                    m_d_accuracy += 0.5 * (d_acc_real + d_acc_fake)
                    m_d_accuracy_fake += d_acc_fake
                    m_d_accuracy_real += d_acc_real
                    m_d_cat_accuracy += cat_accuracy

                m_d_loss /= float(K_UNROLLED)
                m_d_accuracy /= float(K_UNROLLED)
                m_d_accuracy_fake /= float(K_UNROLLED)
                m_d_accuracy_real /= float(K_UNROLLED)
                m_d_loss_cat /= float(K_UNROLLED)
                m_d_cat_accuracy /= float(K_UNROLLED)

                # store value
                with train_summary_writer.as_default():
                    tf.summary.scalar("D loss", m_d_loss, step)
                    tf.summary.scalar("D accuracy", m_d_accuracy, step)
                    tf.summary.scalar("D accuracy - fake", m_d_accuracy_fake, step)
                    tf.summary.scalar("D accuracy - real", m_d_accuracy_real, step)

                    tf.summary.scalar("D category loss", m_d_loss_cat, step)
                    tf.summary.scalar("D category accuracy", m_d_cat_accuracy, step)
                    tf.summary.scalar("D total loss", m_d_loss + m_d_loss_cat, step)

            # training G
            if train_g:
                gan.d.trainable = False
                noise = Generator.generate_noise(None, BATCH_SIZE, 100)

                # Get a batch of random labels
                labels_random = tf.random.uniform(shape=(BATCH_SIZE, 1), minval=0, maxval=NB_GENRES,
                                                  dtype=tf.dtypes.int32)
                labels_random_categorical = utils.to_categorical(labels_random, num_classes=NB_GENRES)

                _, m_a_loss, m_a_cat_loss, m_a_accuracy, _ = gan.train_on_batch(noise,
                                                                                [ones, labels_random_categorical])

                with train_summary_writer.as_default():
                    # store value
                    tf.summary.scalar("G loss", m_a_loss, step)
                    tf.summary.scalar("G accuracy", m_a_accuracy, step)
                    tf.summary.scalar("style ambiguity loss", m_a_cat_loss, step)
                    tf.summary.scalar("G total loss", m_a_loss + m_a_cat_loss, step)

                gan.d.trainable = True

            if train_d and train_g:

                total_los = m_a_loss + m_a_cat_loss + m_d_loss + m_d_loss_cat
                if total_los < min_total_loss:
                    min_total_loss = total_los
                    min_total_loss_epoch = epoch
                    gan.save(save_path_best+"/gan")

                if m_a_loss / m_d_loss > MAX_LOSS_RATIO:
                    train_d = False
                    train_g = True
                    print("Pausing D")
                elif m_d_loss / m_a_loss > MAX_LOSS_RATIO:
                    train_g = False
                    train_d = True
                    print("Pausing G")
            else:
                train_d = True
                train_g = True

        # show results every epoch, save audio and sheet examples
        print("epoch", epoch, repeat)
        print("d_loss", m_d_loss, "d_cat_acc", m_d_cat_accuracy,
              "a_loss", m_a_loss, 'a_cat_loss', m_a_cat_loss)  # print mean loss)
        print("d_accuracy", m_d_accuracy)

        # sample output
        noise = Generator.generate_noise(None, 1, 100)
        notes_generated = gan.g(noise)

        img_path, wav = plot_and_play_notes(notes_generated)

        image = matplotlib.image.imread(img_path)

        audio_data = tf.convert_to_tensor(wav, dtype=tf.dtypes.float32)
        audio_data = tf.expand_dims(audio_data, axis=0)
        audio_data = tf.expand_dims(audio_data, axis=-1)

        image = tf.convert_to_tensor(image, dtype=tf.dtypes.float32)
        image = tf.expand_dims(image, axis=0)

        with train_summary_writer.as_default():
            tf.summary.audio(
                "audio sample", audio_data, sample_rate=44100, step=step
            )
            tf.summary.image("sheet sample", image, step=step)

        print()
        print()

        # store temporary models of g every 5 epochs
        if (epoch + 1) % 1 == 0:
            gan.save(save_path_training+"%03d-gan" % epoch)
