import easyocr
import os
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def ocr_checker(final_state):
    reader = easyocr.Reader(['en'])  # add more languages if needed
    ocr_results = {}
    processed_folder = "./processed_certificates"

    for file in final_state["accepted_certi"]:
        if file.endswith(".png"):
            img_path = os.path.join(processed_folder, file)
            result = reader.readtext(img_path, detail=0)  # get only text
            ocr_results[file] = " ".join(result)

    final_state["ocr_texts"] = ocr_results
    return final_state
