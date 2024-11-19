# Task 1
# Implement a program that removes the URLs, email addresses and hashtags using Regex

import re

text = "I have so much fun. yahoo! kwan@scitech.com"

email_regex = '[a-zA-Z0-9_]+[a-zA-Z0-9_.]*@[a-zA-Z0-9]+.[a-zA-Z0-9]+'
cleaned_email_text = re.sub(email_regex, '', text) # remove email
print(cleaned_email_text)

url_text = "http://www.wikipedia.com and www.google.com and https://youtube.com"
url_regex = '(?:https?://)?[a-zA-Z0-9]+(?:\\.[a-zA-Z0-9]+)+'
cleaned_url_text = re.sub(url_regex,'', url_text) # find url
print(cleaned_url_text)

hashtag_text = "I love #icecream and #chocolate"
hashtag_regex = '#[a-zA-Z0-9_]+'
hashtags = re.sub(hashtag_regex,'', hashtag_text) # remove all hashtags
print(hashtags)
