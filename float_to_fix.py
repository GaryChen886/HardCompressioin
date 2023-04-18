import glob
import numpy as np
import numpy as np

def float_to_fixed(floating_point, fractional_bits=8):
    scaling_factor = 2 ** fractional_bits
    fixed_point = round(floating_point * scaling_factor)
    return fixed_point

def fixed_to_hex(fixed_point, fractional_bits=8):
    scaling_factor = 2 ** fractional_bits
    integer_part = fixed_point // scaling_factor
    fractional_part = fixed_point % scaling_factor
    hex_string = hex(integer_part)[2:].zfill(2) + hex(fractional_part)[2:].zfill(2)
    return hex_string.upper()

for file in glob.glob('*.np[yz]'):
	# Load the floating-point weights from the .npy file
	weights = np.load(file)

	# Convert the weights to fixed-point representation with 8 bits for the fractional part
	fixed_weights = np.vectorize(float_to_fixed)(weights, fractional_bits=8)

	# Convert the fixed-point weights to hexadecimal format
	hex_weights = np.vectorize(fixed_to_hex)(fixed_weights, fractional_bits=8)

	# Write the hexadecimal weights to a new file
	with open('weights_hex.txt', 'a') as f:
		for row in hex_weights:
			for hex_val in row:
				f.write(hex_val + ' ')
				f.write('\n')

	# Load the 2D array from the numpy file
	#my_weights = np.load(file)
	print(file)
	# Convert the array to a fixed-point number format
	fixed_point_weights = np.round(weights * 2**8).astype(np.int32)

	# Print the original and fixed-point weights
	print('Original weights:')
	print(weights)
	print('Fixed-point weights:')
	print(fixed_point_weights)
	print('Hex-point weights:')
	print(hex_weights)
