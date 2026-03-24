from email.parser import BytesParser
from email import policy
import os
from tqdm import tqdm


def load_emails(email_path, label_file):
    labels = {}
    with open(label_file, 'r') as f:
        for line in f.readlines():
            splitted = line.split()
            labels[splitted[1]] = splitted[0]

    rows = []
    for file_name in tqdm(os.listdir(email_path)):
        try:
            with open(os.path.join(email_path, file_name), 'rb') as fp:
                msg = BytesParser(policy=policy.compat32).parse(fp)
                
                body = None
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True)
                        break
                
                if body:
                    text = body.decode('utf-8', errors='replace')
                    text = text.replace(">", "")
                    rows.append((text, labels[file_name]))
        except Exception as e:
            print(f"Skipping {file_name}: {e}")

    return rows
