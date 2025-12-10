# run_wsd.py

from cs5322f25prog3 import (
    WSD_Test_director,
    WSD_Test_overtime,
    WSD_Test_rubbish,
)


def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        # keep only non-empty lines
        return [line.strip() for line in f if line.strip()]


def write_predictions(path, preds):
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(str(p) + "\n")


def main():
    # === DIRECTOR ===
    
    director_in = "director_testdata.txt"
    director_out = "director_results.txt"  # change name
    director_sents = load_sentences(director_in)
    director_preds = WSD_Test_director(director_sents)
    assert len(director_preds) == len(director_sents)
    write_predictions(director_out, director_preds)
    

    # === OVERTIME ===
    overtime_in = "overtime_testdata.txt"
    overtime_out = "overtime_results.txt"
    overtime_sents = load_sentences(overtime_in)
    overtime_preds = WSD_Test_overtime(overtime_sents)
    assert len(overtime_preds) == len(overtime_sents)
    write_predictions(overtime_out, overtime_preds)

    # === RUBBISH ===
    
    rubbish_in = "rubbish_testdata.txt"
    rubbish_out = "rubbish_results.txt"
    rubbish_sents = load_sentences(rubbish_in)
    rubbish_preds = WSD_Test_rubbish(rubbish_sents)
    assert len(rubbish_preds) == len(rubbish_sents)
    write_predictions(rubbish_out, rubbish_preds)
    
    print("All result files written.")


if __name__ == "__main__":
    main()
