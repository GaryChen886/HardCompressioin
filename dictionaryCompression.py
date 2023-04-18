import math

def compress(data):
    dictionary = {}
    next_code = 0
    result = []
    w = ""

    for c in data:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            # Find closest entry in dictionary using Euclidean distance
            closest_dist = math.inf
            closest_entry = None
            for entry in dictionary:
                dist = math.sqrt(sum((ord(entry[i]) - ord(wc[i]))**2 for i in range(len(wc))))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_entry = entry
            result.append(dictionary[closest_entry])
            dictionary[wc] = next_code
            next_code += 1
            w = c

    if w:
        result.append(dictionary[w])

    return result

def decompress(data):
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256
    result = ""

    w = dictionary[data.pop(0)]
    result += w

    for code in data:
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = w + w[0]
        result += entry

        dictionary[next_code] = w + entry[0]
        next_code += 1

        w = entry

    return result

def calculate_loss(original_data, compressed_data):
    original_bits = len(original_data) * 8
    compressed_bits = len(compressed_data) * math.ceil(math.log2(len(set(compressed_data))))
    return (compressed_bits / original_bits) * 100
