

def _predict_for_word(sent_list, word):
    model = load_model(word)

    # Apply same preprocessing used during training
    processed = [preprocess(s, word) for s in sent_list]

    # Use the scikit-learn pipeline to predict
    preds = model.predict(processed)

    # Make sure we return a list of plain Python ints
    return [int(p) for p in preds]


def WSD_Test_director(list):
    return _predict_for_word(list, "director")

def WSD_Test_rubbish(list):
    return _predict_for_word(list, "rubbish")

def WSD_Test_overtime(list):
    return _predict_for_word(list, "overtime")