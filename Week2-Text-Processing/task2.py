# Task 2
# Implement a program that removes the URLs, email addresses and hashtags from a given textand save them in separate text files. emails.txt, urls.txt, hashtags.txt (one per line)

import re

text = "I have so much fun. yahoo! kwan@scitech.com"

email_regex = '[a-zA-Z0-9_]+[a-zA-Z0-9_.]*@[a-zA-Z0-9]+.[a-zA-Z0-9]+'
cleaned_email_text = re.findall(email_regex, text) # find email
print(cleaned_email_text)

url_text = "http://www.wikipedia.com and www.google.com and https://youtube.com"
url_regex = '(?:https?://)?[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)+'
cleaned_url_text = re.findall(url_regex, url_text) # find url
print(cleaned_url_text)

hashtag_text = "I love #icecream and #chocolate"
hashtag_regex = '#[a-zA-Z0-9_]+'
hashtags = re.findall(hashtag_regex, hashtag_text) # find all hashtags
print(hashtags)


with open('emails.txt', 'w') as email_file:
    email_file.write('\n'.join(cleaned_email_text))

with open('urls.txt', 'w') as url_file:
    url_file.write('\n'.join(cleaned_url_text))
    
with open('hashtags.txt', 'w') as hashtag_file:
    hashtag_file.write(str(hashtags))
