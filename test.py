import csv

f = open('hello.csv', 'w+', encoding='utf-8', newline='')

csv_writer = csv.writer(f)

csv_writer.writerow(["姓名","年龄","性别"])

f.close()