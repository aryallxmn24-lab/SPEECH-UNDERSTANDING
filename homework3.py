def words2characters(words):
    characters = []

    for word in words:
        word_str = str(word)
        for char in word_str:
            characters.append(char)

    return characters


# Test case
words = ['hello', 1.234, True]
result = words2characters(words)
print(result)