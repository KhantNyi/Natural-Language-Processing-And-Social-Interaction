# Task 3
# Implement a program that extracts and print all course codes from a given text.

import re

course_text = "I like CSX4210 CSX3001 BGA1402 BBA1005"
course_regex = '[a-zA-Z0-9.]+[0-9]+'
courses = re.findall(course_regex, course_text) # find all course codes

for course in courses:
    print(course)
