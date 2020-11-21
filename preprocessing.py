import re


def preprocess_sentence(sentence, person1, person2):

    if person1 in sentence:
        index = sentence.find(person1) + len(person1) + 2
        sentence = sentence[index:]
    if person2 in sentence:
        index = sentence.find(person2) + len(person2) + 2
        sentence = sentence[index:]
    sentence = sentence.lower().strip()

    # replace all links with link token
    if ".com" in sentence or ".org" in sentence:
        sentence = ''
    if "<media omitted>" in sentence:
        sentence = ''

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    # sentence = re.sub(r"[^a-zA-Z]+", " ", sentence) #removing punctuation to test
    sentence = sentence.strip()
    # adding a start and an end token to the sentence

    # remove small sentences
    if len(sentence) < 3:
        sentence = ''

    return sentence


def extract_sentences_list(person1, person2):

    sentences_list = []
    with open("traindata.txt") as mytxt:
        rawtexts = mytxt.readlines()
    if "Messages and calls are end-to-end encrypted" in rawtexts[0]:
        rawtexts.pop(0)

    line = rawtexts[0]

    # Check who the conversation starts with
    if person1 in line:
        p1 = person1
        p2 = person2
    else:
        p2 = person1
        p1 = person1

    state = True  # If true, processing first persons dialogues

    tempsentence = preprocess_sentence(rawtexts[0], person1, person2)
    for i in range(1, len(rawtexts)):
        if state:
            if p2 in rawtexts[i]:
                state = not state

                if tempsentence:
                    sentences_list.append(tempsentence)
                tempsentence = preprocess_sentence(
                    rawtexts[i], person1, person2)
            else:
                if tempsentence:
                    tempsentence = tempsentence + " " + \
                        preprocess_sentence(rawtexts[i], person1, person2)
                else:
                    tempsentence = tempsentence + \
                        preprocess_sentence(rawtexts[i], person1, person2)

        else:
            if p1 in rawtexts[i]:
                state = not state

                if tempsentence:
                    sentences_list.append(tempsentence)
                tempsentence = preprocess_sentence(
                    rawtexts[i], person1, person2)
            else:
                if tempsentence:
                    tempsentence = tempsentence + " " + \
                        preprocess_sentence(rawtexts[i], person1, person2)
                else:
                    tempsentence = tempsentence + \
                        preprocess_sentence(rawtexts[i], person1, person2)

    return sentences_list


def load_conversations(sentence_list):
    inputs, outputs = [], []
    for i in range(len(sentence_list) - 1):
        inputs.append(sentence_list[i])
        outputs.append(sentence_list[i+1])
    return inputs, outputs
