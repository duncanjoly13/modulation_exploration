"""
Description: This script implements QAM modulation and demodulation for a given modulation order.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def generateQAM(mod_order):
    """
    Generates QAM symbols for the given modulation order.

    :param mod_order: modulation order (e.g., 16 for 16-QAM)
    :return: array of QAM symbols
    """
    # Creates an array of complex numbers representing the QAM symbols
    return np.array([complex((2 * i - 1), (2 * j - 1)) for i in range(1, int(np.sqrt(mod_order)) + 1) for j in range(1, int(np.sqrt(mod_order)) + 1)])

def generateBits(mod_order, signal, qam):
    """
    Demodulates the QAM signal into bits based on the nearest QAM symbols.

    :param mod_order: modulation order
    :param signal: QAM-modulated signal array
    :param qam: array of QAM symbols
    :return: array of unpacked bits
    """
    # Calculate the number of bits per symbol
    k = int(np.log2(mod_order))
    # Calculate Euclidean distances from signal points to each QAM symbol
    distances = np.array([np.abs(signal - qam[i]) ** 2 for i in range(mod_order)])
    # Find the closest QAM symbol for each signal point
    symbols = np.argmin(distances, axis=0)
    # Unpack symbols into individual bits and return them as a flat array
    return np.unpackbits(symbols.astype(np.uint8).reshape(-1, 1), axis=1)[:, -k:].flatten()

def modulate(bits, mod_order, oversample=1):
    """
    Modulates a bit array into a QAM signal.

    :param bits: array of bits to be modulated
    :param mod_order: modulation order
    :param oversample: oversampling factor (repeats each symbol in the signal)
    :return: QAM-modulated signal array
    """
    # Calculate number of bits per symbol
    k = int(np.log2(mod_order))
    # Calculate number of symbols from bit array length
    N = len(bits) // k
    # Convert groups of bits into symbol indices
    symbols = np.sum(bits.reshape(N, k) << np.arange(k - 1, -1, -1), axis=1)
    # Generate QAM symbol array and normalize its power
    qam = generateQAM(mod_order)
    qam = qam[:mod_order] / np.sqrt(np.mean(np.abs(qam) ** 2))
    # Map symbols to QAM signal and oversample
    signal = qam[symbols]
    signal = np.repeat(signal, oversample)
    return signal

def demodulate(signal, mod_order, oversample=1):
    """
    Demodulates a QAM signal back into bits.

    :param signal: QAM signal array
    :param mod_order: modulation order
    :param oversample: oversampling factor
    :return: demodulated bit array
    """
    # Downsample signal based on the oversampling factor
    signal = signal[::oversample]
    # Generate normalized QAM symbol array
    qam = generateQAM(mod_order)
    qam = qam[:mod_order] / np.sqrt(np.mean(np.abs(qam) ** 2))
    # Convert QAM signal into bits
    return generateBits(mod_order, signal, qam)

def plot(signal, mod_order, oversample=1, save_plot=False):
    """
    Plots the QAM constellation of the signal.

    :param signal: QAM-modulated signal array
    :param mod_order: modulation order
    :param oversample: oversampling factor
    """
    # Downsample the signal based on oversampling factor
    signal = signal[::oversample]
    # Generate normalized QAM symbol array
    qam = generateQAM(mod_order)
    qam = qam[:mod_order] / np.sqrt(np.mean(np.abs(qam) ** 2))

    # Plot the signal points and QAM constellation
    plt.figure()
    plt.plot(signal.real, signal.imag, 'o', label="Signal Points")
    plt.plot(qam.real, qam.imag, 'x', label="QAM Constellation")

    # Position the legend outside the plot area
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()  # Adjust layout to make room for the legend

    # Save the plot as an image if specified
    if save_plot:
        filename = "qam_constellation.png"
        base, ext = os.path.splitext(filename)
        counter = 1
        unique_filename = filename

        # Check if the file exists and increment the counter until we find a unique name
        while os.path.exists(unique_filename):
            unique_filename = f"{base}_{counter}{ext}"
            counter += 1

        plt.savefig(unique_filename)
        print(f"Plot saved as {unique_filename}")
        plt.savefig(unique_filename)

    plt.show()

def strToBits(input_string: str):
    """
    Converts a string into a bit array.

    :param input_string: string to be converted
    :return: bit array representation of the input string
    """
    byte_array = bytearray(input_string, 'utf-8')
    return np.array([int(bit) for byte in byte_array for bit in format(byte, '08b')], dtype=int)

def bitsToStr(bit_array):
    """
    Converts a bit array back to a string.

    :param bit_array: array of bits
    :return: decoded string
    """
    # Split bit array into 8-bit chunks and convert each to a character
    byte_chunks = [bit_array[i:i+8] for i in range(0, len(bit_array), 8)]
    chars = [chr(int("".join(map(str, byte)), 2)) for byte in byte_chunks]
    return ''.join(chars)

def randomBitsTest(mod_order, oversample, save_plot=False):
    """
    Runs a test with random bits for QAM modulation and demodulation.

    :param mod_order: modulation order
    :param oversample: oversampling factor
    """
    bits = np.random.randint(0, 2, 100)  # Generate random bit array
    print("Original data:\n" + str(bits), end='\n\n')
    signal = modulate(bits, mod_order, oversample)
    print("Modulated signal:\n" + str(signal), end='\n\n')
    decoded_bits = demodulate(signal, mod_order, oversample)
    print("Demodulated data:\n" + str(decoded_bits), end='\n\n')
    print("Original data equals demodulated data: " + str(np.array_equal(bits, decoded_bits)), end='\n\n')
    plot(signal, mod_order, oversample, save_plot)

def stringInputTest(input_string, mod_order, oversample, save_plot=False):
    """
    Encodes and decodes a string using QAM modulation and demodulation.

    :param input_string: string to encode
    :param mod_order: modulation order
    :param oversample: oversampling factor
    """
    print("Original data:\n" + input_string, end='\n\n')
    bits = strToBits(input_string)
    print("Bit representation of data:\n" + str(bits), end='\n\n')
    signal = modulate(bits, mod_order, oversample)
    print("Modulated signal:\n" + str(signal), end='\n\n')
    decoded_bits = demodulate(signal, mod_order, oversample)
    print("Demodulated data as bits:\n" + str(decoded_bits), end='\n\n')
    string_representation = bitsToStr(decoded_bits)
    print("Demodulated data as string:\n" + string_representation, end='\n\n')
    print("Original data equals demodulated data: " + str(input_string == string_representation), end='\n\n')
    plot(signal, mod_order, oversample, save_plot)

if __name__ == '__main__':
    mod_order = 16  # Set modulation order (e.g., 16-QAM)
    oversample = 4  # Set oversampling factor
    save_plot = False  # Set to True to save the plot as an image

    randomBitsTest(mod_order, oversample, save_plot)
    stringInputTest("Hello, World!", mod_order, oversample, save_plot)
