# Task 4
# Implement a program that extracts, saves all course codes and their frequency from the given text in a CSV file.

import re

course_text = "I take CSX4210, CSX3009 and CSX3010 but my best friend takes CSX3007, CSX4210, and CSX4110 this semester."
course_regex = '[a-zA-Z0-9.]+[0-9]+'
courses = re.findall(course_regex, course_text) # find all course codes
print(courses)

course_list = list(set(courses))
print(course_list)

sorted_course = sorted(course_list) # sort course codes
print(sorted_course)

sorted_counts = []
for course in sorted_course: # count the frequency of each course code
    count = courses.count(course)
    sorted_counts.append(f'{course},{count}') 

# save course codes and their frequency in a CSV file
with open('course_code.csv', 'w') as course_file:
    course_file.write('\n'.join(sorted_counts))
