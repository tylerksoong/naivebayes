from email.parser import BytesParser
from email import policy
import os
from tqdm import tqdm

common_words = set()

email_path = "TRAINING/"

try:
    for file_name in tqdm(os.listdir(email_path)):
        with open(os.path.join(email_path, file_name), 'rb') as fp:
            msg = BytesParser(policy=policy.default).parse(fp)

        body = msg.get_body(preferencelist=('plain'))
        if body:
            words = body.get_content().split()
            for word in words:
                common_words.add(word)
except:

    with open('corpus.txt', 'w') as f:
        mew = list(common_words)
        f.writelines(common_words)
finally:
    with open('corpus.txt', 'w') as f:
        mew = list(common_words)
        for word in mew:
            f.write(word + "\n")
