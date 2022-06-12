split1_train = open("split1_train.SRC", "w")
split1_dev = open("split1_dev.SRC", "w")
split1_test = open("split1_test.SRC", "w")

inflection1 = {'left': ' s', 'twice': ' x', 'thrice': ' x', 'right': ' s', 'walk': ' o', 'look': ' o', 'turn': ' o', 'jump': ' o', 'run': ' o', 'and': ' a', 'after': ' a', 'around': ' s', 'opposite': ' s'}


train = open("train.SRC", "r")
dev = open("dev.SRC", "r")
test = open("test.SRC", "r")

for line in train:
    words_train = []
    for word in line.split():
        suffix = inflection1[word]
        words_train.append(word + suffix)
    words_train.append("\n")
    command = " ".join(words_train)
    split1_train.write(command)


for line in dev:
    words_dev = []
    for word in line.split():
        suffix = inflection1[word]
        words_dev.append(word + suffix)
    words_dev.append("\n")
    command = " ".join(words_dev)
    split1_dev.write(command)


for line in test:
    words_test = []
    for word in line.split():
        suffix = inflection1[word]
        words_test.append(word + suffix)
    words_test.append("\n")
    command = " ".join(words_test)
    split1_test.write(command)

train.close()
dev.close()
test.close()
split1_train.close()
split1_dev.close()
split1_test.close()

train = open("train.SRC", "r")
dev = open("dev.SRC", "r")
test = open("test.SRC", "r")

split2_train = open("split2_train.SRC", "w")
split2_dev = open("split2_dev.SRC", "w")
split2_test = open("split2_test.SRC", "w")

inflection2 = {'left': ' s', 'twice': ' x', 'thrice': ' x', 'right': ' s', 'walk': ' a', 'look': ' o', 'turn': ' o', 'jump': ' o', 'run': ' s', 'and': ' a', 'after': ' a', 'around': ' s', 'opposite': ' s'}

for line in train:
    words_train = []
    for word in line.split():
        suffix = inflection2[word]
        words_train.append(word + suffix)
    words_train.append("\n")
    command = " ".join(words_train)
    split2_train.write(command)


for line in dev:
    words_dev = []
    for word in line.split():
        suffix = inflection2[word]
        words_dev.append(word + suffix)
    words_dev.append("\n")
    command = " ".join(words_dev)
    split2_dev.write(command)


for line in test:
    words_test = []
    for word in line.split():
        suffix = inflection2[word]
        words_test.append(word + suffix)
    words_test.append("\n")
    command = " ".join(words_test)
    split2_test.write(command)

train.close()
dev.close()
test.close()
split2_train.close()
split2_dev.close()
split2_test.close()

train = open("train.SRC", "r")
dev = open("dev.SRC", "r")
test = open("test.SRC", "r")

jumpo_train = open("jumpo_train.SRC", "w")
jumpo_dev = open("jumpo_dev.SRC", "w")
jumpo_test = open("jumpo_test.SRC", "w")

verbs = ["jump", "look", "walk", "turn", "run"]

for line in train:
    words_train = []
    for word in line.split():
        if word in verbs:
            words_train.append(word + " o")
        else:
            words_train.append(word)

    words_train.append("\n")
    command = " ".join(words_train)
    jumpo_train.write(command)


for line in dev:
    words_dev = []
    for word in line.split():
        if word in verbs:
            words_dev.append(word + " o")
        else:
            words_dev.append(word)
    words_dev.append("\n")
    command = " ".join(words_dev)
    jumpo_dev.write(command)


for line in test:
    words_test = []
    for word in line.split():
        if word in verbs:
            words_test.append(word + " o")
        else:
            words_test.append(word)
    words_test.append("\n")
    command = " ".join(words_test)
    jumpo_test.write(command)

train.close()
dev.close()
test.close()
jumpo_train.close()
jumpo_dev.close()
jumpo_test.close()

train = open("train.SRC", "r")
dev = open("dev.SRC", "r")
test = open("test.SRC", "r")

unique_train = open("uniquesuffix_train.SRC", "w")
unique_dev = open("uniquesuffix_dev.SRC", "w")
unique_test = open("uniquesuffix_test.SRC", "w")

inflection_unique = {'left': ' s', 'twice': ' x', 'thrice': ' y', 'right': ' u', 'walk': ' o', 'look': ' o', 'turn': ' o', 'jump': ' o', 'run': ' o', 'and': ' a', 'after': ' b', 'around': ' w', 'opposite': ' v'}

for line in train:
    words_train = []
    for word in line.split():
        suffix = inflection_unique[word]
        words_train.append(word + suffix)
    words_train.append("\n")
    command = " ".join(words_train)
    unique_train.write(command)


for line in dev:
    words_dev = []
    for word in line.split():
        suffix = inflection_unique[word]
        words_dev.append(word + suffix)
    words_dev.append("\n")
    command = " ".join(words_dev)
    unique_dev.write(command)


for line in test:
    words_test = []
    for word in line.split():
        suffix = inflection_unique[word]
        words_test.append(word + suffix)
    words_test.append("\n")
    command = " ".join(words_test)
    unique_test.write(command)

train.close()
dev.close()
test.close()
unique_train.close()
unique_dev.close()
unique_test.close()