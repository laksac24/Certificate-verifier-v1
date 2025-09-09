
from PIL import Image
import imagehash

def similarity_checker(final_state):

    img1 = Image.open("./similar_certificates/image.png")

    for i in final_state["human"]:
        img2 = Image.open(f"./processed_certificates/{i}")

        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)

        distance = hash1 - hash2
        similarity = 1 - distance / (len(hash1.hash) ** 2)

        if similarity > 0.58:
            final_state["accepted_certi"].append(i)
        else:
            final_state["rejected_certi"].append(i)

    for i in final_state["ecerti"]:
        img2 = Image.open(f"./processed_certificates/{i}")

        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)

        distance = hash1 - hash2
        similarity = 1 - distance / (len(hash1.hash) ** 2)

        if similarity > 0.9:
            final_state["accepted_certi"].append(i)
        else:
            final_state["rejected_certi"].append(i)

    return final_state

# img1 = Image.open("./similar_certificates/image.png")
# img2 = Image.open(f"./processed_certificates/06_detected_1.png")

# hash1 = imagehash.phash(img1)
# hash2 = imagehash.phash(img2)

# distance = hash1 - hash2
# similarity = 1 - distance / (len(hash1.hash) ** 2)

# print(similarity)